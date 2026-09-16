"""
qcentroid.py - Solver Híbrido MWIS + QAOA para QCentroid Quantum Platform
Basado en la formulación de IQM & Deutsche Bahn (arXiv:2606.11383)

Punto de entrada oficial: run(input_data, solver_params, extra_arguments)
"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ============================================================
# IMPORTACIONES OPCIONALES (Qiskit, Aer, IQM)
# ============================================================
QISKIT_AVAILABLE = False
AER_AVAILABLE = False
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    QISKIT_AVAILABLE = True
    try:
        from qiskit_aer import AerSimulator
        AER_AVAILABLE = True
    except ImportError:
        AER_AVAILABLE = False
except ImportError:
    QISKIT_AVAILABLE = False

IQM_AVAILABLE = False
IQM_IMPORT_ERROR: Optional[str] = None
try:
    from iqm.qiskit_iqm import IQMProvider
    IQM_AVAILABLE = True
except ImportError:
    IQM_AVAILABLE = False
except RuntimeError as exc:
    IQM_AVAILABLE = False
    IQM_IMPORT_ERROR = str(exc)

# ============================================================
# LOGGER OFICIAL DE QCENTROID
# ============================================================
logger = logging.getLogger("qcentroid-user-log")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================
# SANITIZACIÓN DE URL DE IQM
# ============================================================
class URLSanitizer:
    """Limpia la URL del servidor IQM para evitar sufijos de procesadores."""
    KNOWN_QPUS = {
        "emerald",
        "sirius",
        "garnet",
        "sapphire",
        "ruby",
        "diamond",
    }

    @staticmethod
    def sanitize_iqm_url(raw_url: str) -> str:
        if not raw_url:
            return "https://resonance.iqm.tech/"

        url = raw_url.rstrip("/")
        parts = url.split("/")

        if parts and parts[-1].lower() in URLSanitizer.KNOWN_QPUS:
            url = "/".join(parts[:-1])

        if not url.endswith("/"):
            url += "/"

        return url


# ============================================================
# OBJETIVO Y FORMULACIÓN MWIS / QUBO / ISING
# ============================================================
class MWISObjective:
    """Define la función objetivo MWIS, evaluación QUBO/Ising y Ground Truth exacto."""

    def __init__(
        self,
        graph: nx.Graph,
        penalty_override: Optional[float] = None,
    ):
        self.graph = graph
        self.nodes = sorted(list(graph.nodes()))
        self.num_nodes = len(self.nodes)

        # Pesos de los nodos (w_i)
        self.weights = {
            n: float(graph.nodes[n].get("weight", 1.0))
            for n in self.nodes
        }

        self.max_weight = max(
            self.weights.values(),
            default=1.0,
        )

        # Penalización estricta del paper: lambda = 4 * max(w_i)
        self.lambda_penalty = (
            float(penalty_override)
            if penalty_override is not None
            else 4.0 * self.max_weight
        )

        # Mapeo determinista: nodo -> índice de qubit
        self.node_index = {
            node: i
            for i, node in enumerate(self.nodes)
        }

    def bitstring_to_bits(self, bitstring: str) -> List[int]:
        """Convierte bitstring de Qiskit a bits alineados (invirtiendo orden little-endian)."""
        clean = bitstring.replace(" ", "")
        if len(clean) != self.num_nodes:
            raise ValueError(
                f"Bitstring inválido: longitud {len(clean)}, se esperaban {self.num_nodes} bits."
            )
        return [
            1 if clean[-(i + 1)] == "1" else 0
            for i in range(self.num_nodes)
        ]

    def qubo_energy(self, bitstring: str) -> float:
        """Calcula H(x) = -sum(w_i * x_i) + lambda * sum(x_i * x_j). Mínima energía = Mejor solución."""
        bits = self.bitstring_to_bits(bitstring)
        energy = 0.0

        # Término lineal de beneficio
        for i, node in enumerate(self.nodes):
            if bits[i]:
                energy -= self.weights[node]

        # Penalización por conflictos activos
        for u, v in self.graph.edges():
            i = self.node_index[u]
            j = self.node_index[v]
            if bits[i] and bits[j]:
                energy += self.lambda_penalty

        return energy

    def solution_weight(self, selected_nodes: List[str]) -> float:
        return sum(self.weights[n] for n in selected_nodes if n in self.weights)

    def is_feasible(self, selected_nodes: List[str]) -> bool:
        selected = set(selected_nodes)
        return all(
            not (u in selected and v in selected)
            for u, v in self.graph.edges()
        )

    def solve_exact_ground_truth(self) -> Tuple[List[str], float]:
        """Calcula el MWIS exacto mediante Maximum Weight Clique sobre el complemento."""
        complement_graph = nx.complement(self.graph)

        # Convertir obligatoriamente a int para evitar el ValueError de NetworkX
        for node in complement_graph.nodes():
            complement_graph.nodes[node]["weight"] = int(round(self.weights[node]))

        clique, _ = nx.max_weight_clique(
            complement_graph,
            weight="weight",
        )

        real_weight = sum(self.weights[node] for node in clique)
        return sorted(clique), float(real_weight)


# ============================================================
# PODA / REPARACIÓN DETERMINISTA (PRUNING)
# ============================================================
class MWISPruner:
    """Repara soluciones conflictivas eliminando iterativamente el nodo con menor w_i / deg(i)."""

    @staticmethod
    def prune_solution(
        graph: nx.Graph,
        initial_selected: List[str],
    ) -> List[str]:
        selected = set(initial_selected)

        while True:
            subgraph = graph.subgraph(selected)
            conflicts = list(subgraph.edges())

            if not conflicts:
                break

            conflict_nodes = set()
            for u, v in conflicts:
                conflict_nodes.add(u)
                conflict_nodes.add(v)

            worst_node = min(
                conflict_nodes,
                key=lambda n: (
                    float(graph.nodes[n].get("weight", 0.0))
                    / max(1, subgraph.degree(n))
                ),
            )

            selected.remove(worst_node)

        return sorted(list(selected))


# ============================================================
# GENERADOR DE ASSETS VISUALES
# ============================================================
class VisualizationAssetGenerator:
    """Genera la representación gráfica del grafo de conflictos en additional_output/."""

    @staticmethod
    def generate_conflict_graph(graph: nx.Graph, selected_nodes: List[str]) -> Optional[str]:
        try:
            output_dir = Path("additional_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "conflict_graph.png"

            plt.figure(figsize=(9, 7), dpi=150)
            pos = nx.spring_layout(graph, seed=42)

            selected_set = set(selected_nodes)
            node_colors = []
            node_sizes = []

            for node in graph.nodes():
                if node in selected_set:
                    node_colors.append("#2ecc71")  # Verde: Seleccionado
                    node_sizes.append(900)
                else:
                    node_colors.append("#95a5a6")  # Gris: No seleccionado
                    node_sizes.append(600)

            nx.draw_networkx_nodes(
                graph, pos, node_color=node_colors, node_size=node_sizes, edgecolors="#2c3e50"
            )
            nx.draw_networkx_edges(graph, pos, edge_color="#e74c3c", width=1.5, alpha=0.7)

            labels = {
                node: f"{node}\n(w={graph.nodes[node].get('weight', 0):.0f})"
                for node in graph.nodes()
            }
            nx.draw_networkx_labels(
                graph, pos, labels=labels, font_size=8, font_weight="bold", font_color="black"
            )

            plt.title("Grafo de Conflictos de Material Rodante (MWIS)", fontsize=12, fontweight="bold")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            logger.info("Visualización guardada en %s", path)
            return str(path)

        except Exception as exc:
            logger.warning("No se pudo generar la imagen del grafo: %s", exc)
            return None


# ============================================================
# CONSTRUCCIÓN Y VALIDACIÓN DEL GRAFO DE CONFLICTOS
# ============================================================
def build_conflict_graph(input_data: Dict[str, Any]) -> nx.Graph:
    """Construye el grafo de conflictos leyendo nodos y aristas con validación estricta."""
    graph = nx.Graph()

    nodes_raw = input_data.get("nodes", [])
    edges_raw = input_data.get("edges", [])

    for node_data in nodes_raw:
        node_id = node_data["id"]
        trips = node_data.get("trips", [])
        passenger_km = float(node_data.get("passenger_km", 0.0))
        empty_km = float(node_data.get("empty_km", 0.0))

        if "weight" in node_data:
            weight = float(node_data["weight"])
        else:
            weight = (2.0 * passenger_km) - empty_km

        operating_cost = float(node_data.get("operating_cost", 0.0))

        graph.add_node(
            node_id,
            weight=weight,
            trips=trips,
            passenger_km=passenger_km,
            empty_km=empty_km,
            operating_cost=operating_cost,
        )

    if edges_raw:
    for edge in edges_raw:

        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            logger.warning(
                "Arista ignorada (formato inválido): %r",
                edge,
            )
            continue

        u = edge[0]
        v = edge[1]

        if u not in graph or v not in graph:
            logger.warning(
                "Arista ignorada (nodo inexistente): %s - %s",
                u,
                v,
            )
            continue

        if u == v:
            logger.warning(
                "Autoconflicto ignorado: %s",
                u,
            )
            continue

        graph.add_edge(u, v)

    else:
        node_list = list(graph.nodes())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                u, v = node_list[i], node_list[j]
                trips_u = set(graph.nodes[u]["trips"])
                trips_v = set(graph.nodes[v]["trips"])
                if trips_u & trips_v:
                    graph.add_edge(u, v)

    return graph


# ============================================================
# SÍNTESIS DEL CIRCUITO QAOA
# ============================================================
def build_qaoa_circuit(
    graph: nx.Graph,
    objective: MWISObjective,
    qaoa_depth: int,
) -> QuantumCircuit:
    """Sintetiza el circuito QAOA variacional usando rotaciones RZ, RZZ y RX."""
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit no está instalado o disponible.")

    n_qubits = objective.num_nodes
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qr, cr)

    for i in range(n_qubits):
        qc.h(qr[i])

    gamma_values = [0.05 / (layer + 1) for layer in range(qaoa_depth)]
    beta_values = [0.25 / (layer + 1) for layer in range(qaoa_depth)]

    for layer in range(qaoa_depth):
        gamma = gamma_values[layer]
        beta = beta_values[layer]

        for i, node in enumerate(objective.nodes):
            deg = graph.degree(node)
            h_i = (objective.weights[node] / 2.0) - (
                objective.lambda_penalty * deg / 4.0
            )
            qc.rz(2.0 * gamma * h_i, qr[i])

        for u, v in graph.edges():
            i = objective.node_index[u]
            j = objective.node_index[v]
            coupling = objective.lambda_penalty / 4.0
            qc.rzz(2.0 * gamma * coupling, qr[i], qr[j])

        for i in range(n_qubits):
            qc.rx(2.0 * beta, qr[i])

    qc.measure(qr, cr)
    return qc


# ============================================================
# DECODIFICACIÓN POR MÍNIMA ENERGÍA QUBO
# ============================================================
def select_best_qaoa_bitstring(
    counts: Dict[str, int],
    objective: MWISObjective,
) -> Tuple[str, float]:
    """Selecciona el bitstring de menor energía QUBO real (desempatando por shots)."""
    if not counts:
        raise ValueError("El conteo de mediciones QAOA está vacío.")

    best_bitstring = min(
        counts.keys(),
        key=lambda bitstring: (
            objective.qubo_energy(bitstring),
            -counts[bitstring],
        ),
    )
    best_energy = objective.qubo_energy(best_bitstring)
    return best_bitstring, best_energy


def bitstring_to_selected_nodes(
    bitstring: str,
    objective: MWISObjective,
) -> List[str]:
    bits = objective.bitstring_to_bits(bitstring)
    return [objective.nodes[i] for i, bit in enumerate(bits) if bit == 1]


# ============================================================
# CÁLCULO DE MÉTRICAS OPERATIVAS Y COBERTURA
# ============================================================
def calculate_coverage(
    nodes_raw: List[Dict[str, Any]],
    selected_cycles: List[str],
) -> Tuple[int, int, float]:
    all_trips = set()
    covered_trips = set()
    selected_set = set(selected_cycles)

    for node_data in nodes_raw:
        trips = node_data.get("trips", [])
        all_trips.update(trips)
        if node_data["id"] in selected_set:
            covered_trips.update(trips)

    scheduled = len(all_trips)
    covered = len(covered_trips)
    rate = covered / scheduled if scheduled > 0 else 1.0
    return scheduled, covered, round(rate * 100.0, 2)


def calculate_operational_metrics(
    nodes_raw: List[Dict[str, Any]],
    selected_cycles: List[str],
) -> Dict[str, float]:
    selected_set = set(selected_cycles)
    passenger_km = 0.0
    empty_km = 0.0
    cost = 0.0

    for node_data in nodes_raw:
        if node_data["id"] in selected_set:
            passenger_km += float(node_data.get("passenger_km", 0.0))
            empty_km += float(node_data.get("empty_km", 0.0))
            cost += float(node_data.get("operating_cost", 0.0))

    return {
        "total_passenger_km": passenger_km,
        "total_empty_km": empty_km,
        "total_operating_cost": cost,
    }


# ============================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN (ENTRYPOINT)
# ============================================================
def run(
    input_data: Dict[str, Any],
    solver_params: Optional[Dict[str, Any]] = None,
    extra_arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Punto de entrada estándar para QCentroid Platform."""
    solver_params = solver_params or {}
    extra_arguments = extra_arguments or {}

    start_time = time.perf_counter()

    iqm_token = (
        solver_params.get("iqm_token")
        or extra_arguments.get("iqm_token")
        or extra_arguments.get("api_token")
        or os.environ.get("IQM_TOKEN")
        or os.environ.get("QCENTROID_TOKEN")
    )

    quantum_computer = solver_params.get("quantum_computer", "emerald")
    raw_server_url = solver_params.get("server_url", "https://resonance.iqm.tech/")
    server_url = URLSanitizer.sanitize_iqm_url(raw_server_url)

    qaoa_depth = int(solver_params.get("qaoa_depth", 2))
    shots = int(solver_params.get("shots", 2048))
    penalty_param = solver_params.get("penalty")

    nodes_raw = input_data.get("nodes", [])
    if not nodes_raw:
        raise ValueError("El dataset de entrada debe contener al menos un nodo.")

    graph = build_conflict_graph(input_data)
    logger.info(
        "Grafo construido con éxito: %d nodos, %d conflictos",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    objective = MWISObjective(graph, penalty_override=penalty_param)
    logger.info("Penalización QUBO calculada: %.4f", objective.lambda_penalty)

    exact_selected, exact_weight = objective.solve_exact_ground_truth()
    logger.info(
        "Ground Truth Exacto Clásico: %s | Peso Máximo: %.4f",
        exact_selected,
        exact_weight,
    )

    backend = None
    backend_used = "Solución exacta clásica (fallback)"
    job_id = "local"

    if iqm_token and IQM_AVAILABLE:
        try:
            logger.info("Conectando con IQM Resonance (%s) en %s...", quantum_computer, server_url)
            provider = IQMProvider(url=server_url, quantum_computer=quantum_computer, token=iqm_token)
            backend = provider.get_backend()
            backend_used = f"IQM Resonance ({quantum_computer})"
        except Exception as exc:
            logger.warning("Fallo de conexión con IQM: %s. Activando AerSimulator.", exc)

    if backend is None and QISKIT_AVAILABLE and AER_AVAILABLE:
        backend = AerSimulator()
        backend_used = "Qiskit AerSimulator (CPU)"

    if QISKIT_AVAILABLE and backend is not None:
        qc = build_qaoa_circuit(graph, objective, qaoa_depth)
        qc_transpiled = transpile(qc, backend=backend, optimization_level=2)

        logger.info("Enviando circuito QAOA p=%d a %s (%d shots)...", qaoa_depth, backend_used, shots)
        q_job = backend.run(qc_transpiled, shots=shots)

        job_id_attr = getattr(q_job, "job_id", None)
        if callable(job_id_attr):
            job_id = job_id_attr()
        elif job_id_attr is not None:
            job_id = str(job_id_attr)

        counts = q_job.result().get_counts()

        initial_bitstring, initial_qubo_energy = select_best_qaoa_bitstring(counts, objective)
        initial_selected = bitstring_to_selected_nodes(initial_bitstring, objective)
    else:
        initial_bitstring = "N/A"
        initial_qubo_energy = -exact_weight
        initial_selected = exact_selected

    selected_cycles = MWISPruner.prune_solution(graph, initial_selected)
    is_feasible = objective.is_feasible(selected_cycles)
    total_weight = objective.solution_weight(selected_cycles)

    if exact_weight != 0:
        optimality_gap_percent = ((exact_weight - total_weight) / abs(exact_weight)) * 100.0
    else:
        optimality_gap_percent = 0.0

    if abs(optimality_gap_percent) < 1e-12:
        optimality_gap_percent = 0.0

    scheduled_trips, covered_trips, coverage_percent = calculate_coverage(nodes_raw, selected_cycles)
    operational = calculate_operational_metrics(nodes_raw, selected_cycles)
    execution_time_seconds = time.perf_counter() - start_time

    asset_path = VisualizationAssetGenerator.generate_conflict_graph(graph, selected_cycles)
    nodes_pruned = max(len(initial_selected) - len(selected_cycles), 0)

    result = {
        "total_weight": float(total_weight),
        "optimality_gap_percent": float(optimality_gap_percent),
        "coverage_percent": float(coverage_percent),
        "execution_time_seconds": float(execution_time_seconds),
        "total_operating_cost": float(operational["total_operating_cost"]),

        "selected_cycles": selected_cycles,
        "is_feasible": bool(is_feasible),

        "total_passenger_km": float(operational["total_passenger_km"]),
        "total_empty_km": float(operational["total_empty_km"]),
        "scheduled_trips": int(scheduled_trips),
        "covered_trips": int(covered_trips),

        "backend_used": backend_used,
        "execution_metrics": {
            "execution_time_seconds": float(execution_time_seconds),
            "final_solution_size": int(len(selected_cycles)),
            "initial_solution_size": int(len(initial_selected)),
            "initial_bitstring_min_qubo": initial_bitstring,
            "initial_qubo_energy": float(initial_qubo_energy),
            "penalty": float(objective.lambda_penalty),
            "qaoa_depth": int(qaoa_depth),
            "shots": int(shots),
            "job_id": job_id,
        },
        "ground_truth_exact": {
            "selected_cycles": exact_selected,
            "total_weight": float(exact_weight),
            "optimality_gap_percent": float(optimality_gap_percent),
        },
        "nodes_pruned": int(nodes_pruned),
        "pruning_stats": {
            "nodes_pruned_count": int(nodes_pruned),
            "pruned_nodes_list": [],
        },
        "assets": {
            "conflict_graph_png": asset_path,
        },
    }

    logger.info("Ejecución finalizada con éxito. Ciclos seleccionados: %s", selected_cycles)
    return result
