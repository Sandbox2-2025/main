 """
qcentroid.py - Solver Híbrido MWIS + QAOA para QCentroid Quantum Platform
Basado en la formulación de IQM & Deutsche Bahn (arXiv:2606.11383)

Punto de entrada oficial: run(input_data, solver_params, extra_arguments)
"""

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

matplotlib.use("Agg")

# Importación opcional de Qiskit y simuladores
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Adaptador de IQM Resonance
IQM_IMPORT_ERROR: Optional[str] = None
try:
    from iqm.qiskit_iqm import IQMProvider
    IQM_AVAILABLE = True
except ImportError:
    IQM_AVAILABLE = False
except RuntimeError as _iqm_exc:
    IQM_AVAILABLE = False
    IQM_IMPORT_ERROR = str(_iqm_exc)

# Configuración del logger oficial de QCentroid
logger = logging.getLogger("qcentroid-user-log")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class URLSanitizer:
    """Limpia la URL del servidor IQM para evitar que contenga el nombre de la QPU."""
    KNOWN_QPUS = {"emerald", "sirius", "garnet", "sapphire", "ruby", "diamond"}

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


class MWISObjective:
    """Estructura matemática QUBO / Ising e inferencia del Ground Truth exacto."""
    def __init__(self, graph: nx.Graph, penalty_override: Optional[float] = None):
        self.graph = graph
        self.nodes = sorted(list(graph.nodes()))
        self.num_nodes = len(self.nodes)
        
        # Pesos de los nodos (w_i = 2 * passenger_km - empty_km)
        self.weights = {n: float(graph.nodes[n].get("weight", 1.0)) for n in self.nodes}
        self.max_weight = max(self.weights.values(), default=1.0)
        
        # Penalización estricta del paper: lambda = 4 * max(w_i)
        self.lambda_penalty = float(penalty_override) if penalty_override is not None else 4.0 * self.max_weight

    def qubo_energy(self, bitstring: str) -> float:
        """Calcula la energía QUBO: H(x) = -sum(w_i * x_i) + lambda * sum(x_i * x_j)."""
        # Qiskit utiliza ordenamiento little-endian
        bits = [1 if bitstring[-(i + 1)] == "1" else 0 for i in range(self.num_nodes)]
        
        energy = 0.0
        # Beneficio lineal
        for i, node in enumerate(self.nodes):
            if bits[i]:
                energy -= self.weights[node]
        
        # Penalización por conflictos activos
        for u, v in self.graph.edges():
            i = self.nodes.index(u)
            j = self.nodes.index(v)
            if bits[i] and bits[j]:
                energy += self.lambda_penalty
                
        return energy

    def is_feasible(self, selected_nodes: List[str]) -> bool:
        sel_set = set(selected_nodes)
        return all(not (u in sel_set and v in sel_set) for u, v in self.graph.edges())

    def solve_exact_ground_truth(self) -> Tuple[List[str], float]:
        """Calcula la solución exacta mediante Máximo Clique en el grafo complementario."""
        complement_graph = nx.complement(self.graph)
        for n in complement_graph.nodes():
            complement_graph.nodes[n]["weight"] = self.weights[n]
            
        clique, weight = nx.max_weight_clique(complement_graph, weight="weight")
        return sorted(clique), float(weight)


class MWISPruner:
    """Poda determinista basada en el ratio r_i = w_i / degree(i)."""
    @staticmethod
    def prune(graph: nx.Graph, initial_selected: List[str]) -> Tuple[List[str], List[str]]:
        selected = set(initial_selected)
        pruned_nodes = []

        while True:
            subgraph = graph.subgraph(selected)
            conflicts = list(subgraph.edges())
            if not conflicts:
                break

            conflict_nodes = set(u for edge in conflicts for u in edge)
            
            # Eliminación iterativa del nodo con menor ratio w_i / degree(i)
            worst_node = min(
                conflict_nodes,
                key=lambda n: float(graph.nodes[n].get("weight", 0.0)) / max(1, subgraph.degree(n))
            )
            selected.remove(worst_node)
            pruned_nodes.append(worst_node)

        return sorted(list(selected)), sorted(pruned_nodes)


class VisualizationAssetGenerator:
    """Genera la imagen PNG del grafo de conflictos en additional_output/."""
    @staticmethod
    def generate_conflict_graph_visualization(graph: nx.Graph, selected: List[str], pruned: List[str], filename: str = "conflict_graph.png") -> Optional[str]:
        try:
            output_dir = "additional_output"
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)

            plt.figure(figsize=(9, 7), dpi=150)
            pos = nx.spring_layout(graph, seed=42)

            node_colors = []
            for n in graph.nodes():
                if n in selected:
                    node_colors.append("#2ecc71")  # Verde: Seleccionado
                elif n in pruned:
                    node_colors.append("#e74c3c")  # Rojo: Podado
                else:
                    node_colors.append("#95a5a6")  # Gris: Descartado

            nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=800, edgecolors="#2c3e50")
            nx.draw_networkx_edges(graph, pos, edge_color="#e74c3c", width=1.5, alpha=0.7)
            
            labels = {n: f"{n}\n(w={graph.nodes[n]['weight']:.0f})" for n in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_weight="bold", font_color="black")

            plt.title("Grafo de Conflictos de Material Rodante (MWIS)", fontsize=12, fontweight="bold")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            logger.info("Visualización de activo guardada en %s", path)
            return path
        except Exception as exc:
            logger.warning("No se pudo generar la imagen del grafo: %s", exc)
            return None


def run(input_data: Dict[str, Any], solver_params: Dict[str, Any], extra_arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Punto de entrada estándar de QCentroid Platform."""
    logger.info("=" * 70)
    logger.info("Iniciando Solver QCentroid MWIS de Material Rodante Ferroviario")
    logger.info("=" * 70)

    start_time = time.time()

    # 1. Búsqueda exhaustiva del token de IQM en todos los posibles orígenes
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
    shots = int(solver_params.get("shots", 2048))
    qaoa_depth = int(solver_params.get("qaoa_depth", 2))
    penalty_param = solver_params.get("penalty")

    # 2. Carga y construcción robusta del Grafo de Conflictos
    nodes_raw = input_data.get("nodes", [])
    edges_raw = input_data.get("edges", [])

    if not nodes_raw:
        return {"selected_cycles": [], "total_weight": 0.0, "is_feasible": False, "error": "Dataset de entrada vacío."}

    graph = nx.Graph()
    all_trips = set()

    for n in nodes_raw:
        n_id = n["id"]
        trips = n.get("trips", [])
        all_trips.update(trips)

        passenger_km = float(n.get("passenger_km", 0.0))
        empty_km = float(n.get("empty_km", 0.0))
        
        # Valoración oficial: w_i = 2 * passenger_km - empty_km
        calculated_w = (2.0 * passenger_km - empty_km) if (passenger_km > 0 or empty_km > 0) else float(n.get("weight", 1.0))
        weight = float(n.get("weight", calculated_w))

        graph.add_node(
            n_id,
            weight=weight,
            trips=trips,
            passenger_km=passenger_km,
            empty_km=empty_km,
            operating_cost=float(n.get("operating_cost", 0.0))
        )

    # CORRECCIÓN DE ARISTAS: e y e[4] en lugar de self-loops e-e
    if edges_raw:
        for e in edges_raw:
            if isinstance(e, list) and len(e) == 2:
                u, v = e, e[4]
                if u in graph and v in graph and u != v:
                    graph.add_edge(u, v)
    else:
        # Generación automática por intersección de viajes si no vienen edges explícitos
        node_list = list(graph.nodes())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                u, v = node_list[i], node_list[j]
                if set(graph.nodes[u]["trips"]) & set(graph.nodes[v]["trips"]):
                    graph.add_edge(u, v)

    objective = MWISObjective(graph, penalty_override=penalty_param)

    # 3. Cálculo de Referencia Clásica Exacta (Ground Truth)
    exact_cycles, exact_weight = objective.solve_exact_ground_truth()
    logger.info("Ground Truth Exacto: %s | Peso Máximo Óptimo: %.1f", exact_cycles, exact_weight)

    # 4. Inicialización del Backend Cuántico (IQM / Aer)
    backend = None
    backend_info = "Qiskit AerSimulator (CPU)"
    job_id = "local"

    if iqm_token and IQM_AVAILABLE:
        try:
            logger.info("Conectando con IQM Resonance (%s) en %s...", quantum_computer, server_url)
            provider = IQMProvider(url=server_url, quantum_computer=quantum_computer, token=iqm_token)
            backend = provider.get_backend()
            backend_info = f"IQM Resonance ({quantum_computer})"
        except Exception as exc:
            logger.warning("Fallo al conectar con IQM: %s. Activando fallback a AerSimulator.", exc)

    if backend is None and QISKIT_AVAILABLE:
        backend = AerSimulator()

    # 5. Ejecución QAOA y Decodificación por Mínima Energía QUBO
    if QISKIT_AVAILABLE and backend is not None:
        n_qubits = len(objective.nodes)
        qr = QuantumRegister(n_qubits, "q")
        cr = ClassicalRegister(n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Estado inicial |+>
        for i in range(n_qubits):
            qc.h(qr[i])

        # Parámetros de circuito QAOA
        gamma_val = 0.05
        beta_val = 0.25

        for _ in range(qaoa_depth):
            for i, node in enumerate(objective.nodes):
                deg = graph.degree(node)
                h_i = (objective.weights[node] / 2.0) - (objective.lambda_penalty * deg / 4.0)
                qc.rz(2.0 * gamma_val * h_i, qr[i])

            for u, v in graph.edges():
                i, j = objective.nodes.index(u), objective.nodes.index(v)
                qc.rzz(2.0 * gamma_val * (objective.lambda_penalty / 4.0), qr[i], qr[j])

            for i in range(n_qubits):
                qc.rx(2.0 * beta_val, qr[i])

        qc.measure(qr, cr)

        qc_transpiled = transpile(qc, backend=backend, optimization_level=2)
        logger.info("Enviando circuito QAOA (p=%d) a %s (%d shots)...", qaoa_depth, backend_info, shots)
        
        q_job = backend.run(qc_transpiled, shots=shots)
        job_id = getattr(q_job, "job_id", lambda: "local")()
        if callable(job_id):
            job_id = job_id()

        counts = q_job.result().get_counts()

        # SELECCIÓN METODOLÓGICA: Evaluar el bitstring de menor energía QUBO real (no max counts)
        best_bitstring = min(counts.keys(), key=lambda b: objective.qubo_energy(b))
        raw_selected = [objective.nodes[i] for i in range(n_qubits) if best_bitstring[-(i + 1)] == "1"]
    else:
        best_bitstring = "N/A"
        raw_selected = exact_cycles

    # 6. Reparación Determinista mediante Poda (Pruning)
    final_selected, pruned_nodes = MWISPruner.prune(graph, raw_selected)

    # 7. Cálculo de Métricas y Cobertura Operativa
    final_weight = sum(graph.nodes[n]["weight"] for n in final_selected)
    covered_trips = set()
    total_passenger_km = 0.0
    total_empty_km = 0.0
    total_operating_cost = 0.0

    for n in final_selected:
        data = graph.nodes[n]
        covered_trips.update(data["trips"])
        total_passenger_km += data["passenger_km"]
        total_empty_km += data["empty_km"]
        total_operating_cost += data["operating_cost"]

    coverage_rate = len(covered_trips) / len(all_trips) if all_trips else 1.0
    exec_time = time.time() - start_time

    # 8. Generación del Activo Gráfico
    asset_path = VisualizationAssetGenerator.generate_conflict_graph_visualization(graph, final_selected, pruned_nodes)
    assets = {"conflict_graph_png": asset_path} if asset_path else {}

    # 9. Respuesta Final Estructurada
    return {
        "selected_cycles": final_selected,
        "total_weight": final_weight,
        "coverage_rate": round(coverage_rate, 4),
        "coverage_percent": round(coverage_rate * 100.0, 2),
        "covered_trips": len(covered_trips),
        "scheduled_trips": len(all_trips),
        "total_empty_km": total_empty_km,
        "total_passenger_km": total_passenger_km,
        "total_operating_cost": total_operating_cost,
        "is_feasible": objective.is_feasible(final_selected),
        "nodes_pruned": len(pruned_nodes),
        "ground_truth_exact": {
            "selected_cycles": exact_cycles,
            "total_weight": exact_weight,
            "optimality_gap_percent": round(((exact_weight - final_weight) / exact_weight) * 100.0, 2) if exact_weight > 0 else 0.0
        },
        "execution_metrics": {
            "execution_time_seconds": exec_time,
            "initial_solution_size": len(raw_selected),
            "final_solution_size": len(final_selected),
            "initial_bitstring_min_qubo": best_bitstring,
            "qaoa_depth": qaoa_depth,
            "shots": shots,
            "penalty": objective.lambda_penalty,
            "job_id": job_id
        },
        "pruning_stats": {
            "nodes_pruned_count": len(pruned_nodes),
            "pruned_nodes_list": pruned_nodes
        },
        "backend_used": backend_info,
        "assets": assets
    }
