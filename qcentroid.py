"""
qcentroid.py - Solver Híbrido MWIS + QAOA para QCentroid Quantum Platform

Problema:
    Selección de ciclos de material rodante ferroviario como
    Maximum Weighted Independent Set (MWIS).

Formulación:
    Minimizar:
        H(x) = -sum(w_i * x_i)
               + lambda * sum(x_i * x_j)

    donde:
        x_i = 1 si el ciclo i es seleccionado
        x_i = 0 si no es seleccionado
        w_i = peso/valor del ciclo
        lambda = penalización por conflicto

La función objetivo utiliza explícitamente el campo "weight".
Los campos passenger_km, empty_km y operating_cost se utilizan
como métricas operativas de la solución.

Incluye:
    - Ground Truth exacto mediante Maximum Weight Clique
      sobre el grafo complementario.
    - QAOA mediante Qiskit.
    - Ejecución opcional en IQM Resonance.
    - Fallback a solución exacta si Qiskit no está disponible.
    - Reparación/poda determinista de soluciones QAOA.
    - Métricas operativas.
    - Generación de grafo PNG.

Punto de entrada oficial:
    run(input_data, solver_params, extra_arguments)
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# IMPORTACIONES OPCIONALES
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


# Adaptador IQM Resonance
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
# LOGGER
# ============================================================

logger = logging.getLogger("qcentroid-user-log")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================
# IQM URL
# ============================================================

class URLSanitizer:
    """Limpia la URL de IQM para evitar que contenga el nombre de la QPU."""

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
# MWIS OBJECTIVE
# ============================================================

class MWISObjective:
    """
    Define la función objetivo MWIS y su representación QUBO/Ising.

    IMPORTANTE:
        "weight" es el valor que se optimiza.

    passenger_km, empty_km y operating_cost NO modifican
    el objetivo por defecto; se utilizan como métricas
    operativas de la solución.
    """

    def __init__(
        self,
        graph: nx.Graph,
        penalty_override: Optional[float] = None,
    ):
        self.graph = graph
        self.nodes = sorted(list(graph.nodes()))
        self.num_nodes = len(self.nodes)

        # ----------------------------------------------------
        # Función objetivo:
        #     maximizar sum(weight_i * x_i)
        # ----------------------------------------------------

        self.weights = {
            n: float(graph.nodes[n].get("weight", 1.0))
            for n in self.nodes
        }

        self.max_weight = max(
            self.weights.values(),
            default=1.0,
        )

        # Penalización:
        #
        # lambda = 4 * max(weight)
        #
        # garantiza que una solución con conflicto sea
        # suficientemente penalizada frente a seleccionar
        # un nodo individual de mayor peso.

        self.lambda_penalty = (
            float(penalty_override)
            if penalty_override is not None
            else 4.0 * self.max_weight
        )

        # Índice nodo -> posición del qubit
        self.node_index = {
            node: i
            for i, node in enumerate(self.nodes)
        }

    # --------------------------------------------------------
    # Conversión bitstring -> bits
    # --------------------------------------------------------

    def bitstring_to_bits(self, bitstring: str) -> List[int]:
        """
        Convierte el bitstring de Qiskit a bits alineados
        con self.nodes.

        Qiskit devuelve el bitstring con el bit de mayor índice
        a la izquierda. Por eso se invierte mediante [-i-1].
        """

        clean = bitstring.replace(" ", "")

        if len(clean) != self.num_nodes:
            raise ValueError(
                f"Bitstring inválido: longitud {len(clean)}, "
                f"se esperaban {self.num_nodes} bits."
            )

        return [
            1 if clean[-(i + 1)] == "1" else 0
            for i in range(self.num_nodes)
        ]

    # --------------------------------------------------------
    # Energía QUBO
    # --------------------------------------------------------

    def qubo_energy(self, bitstring: str) -> float:
        """
        Calcula:

            H(x) =
                -sum(w_i*x_i)
                + lambda*sum(x_i*x_j)

        para las aristas del grafo.

        MENOR energía = MEJOR solución.
        """

        bits = self.bitstring_to_bits(bitstring)

        energy = 0.0

        # Término lineal
        for i, node in enumerate(self.nodes):
            if bits[i]:
                energy -= self.weights[node]

        # Penalización por conflictos
        for u, v in self.graph.edges():

            i = self.node_index[u]
            j = self.node_index[v]

            if bits[i] and bits[j]:
                energy += self.lambda_penalty

        return energy

    # --------------------------------------------------------
    # Peso de una solución
    # --------------------------------------------------------

    def solution_weight(self, selected_nodes: List[str]) -> float:
        return sum(
            self.weights[n]
            for n in selected_nodes
        )

    # --------------------------------------------------------
    # Factibilidad
    # --------------------------------------------------------

    def is_feasible(
        self,
        selected_nodes: List[str],
    ) -> bool:

        selected = set(selected_nodes)

        return all(
            not (u in selected and v in selected)
            for u, v in self.graph.edges()
        )

    # --------------------------------------------------------
    # Ground Truth exacto
    # --------------------------------------------------------

    def solve_exact_ground_truth(
        self,
    ) -> Tuple[List[str], float]:
        """
        Resuelve exactamente MWIS mediante:

            MWIS(G) = Maximum Weight Clique(complement(G))

        Se utiliza exclusivamente como referencia para evaluar
        el resultado QAOA en esta PoC.
        """

        complement_graph = nx.complement(self.graph)

        for node in complement_graph.nodes():
            complement_graph.nodes[node]["weight"] = (
                self.weights[node]
            )

        clique, weight = nx.max_weight_clique(
            complement_graph,
            weight="weight",
        )

        return sorted(clique), float(weight)


# ============================================================
# PODA / REPARACIÓN
# ============================================================

class MWISPruner:
    """
    Repara una solución que contenga conflictos.

    Elimina iterativamente el nodo con menor relación:

        weight / degree

    entre los nodos implicados en conflictos.
    """

    @staticmethod
    def prune(
        graph: nx.Graph,
        initial_selected: List[str],
    ) -> Tuple[List[str], List[str]]:

        selected = set(initial_selected)
        pruned_nodes: List[str] = []

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
            pruned_nodes.append(worst_node)

        return (
            sorted(selected),
            sorted(pruned_nodes),
        )


# ============================================================
# VISUALIZACIÓN
# ============================================================

class VisualizationAssetGenerator:
    """Genera el PNG del grafo de conflictos."""

    @staticmethod
    def generate_conflict_graph_visualization(
        graph: nx.Graph,
        selected: List[str],
        pruned: List[str],
        filename: str = "conflict_graph.png",
    ) -> Optional[str]:

        try:

            output_dir = "additional_output"
            os.makedirs(output_dir, exist_ok=True)

            path = os.path.join(
                output_dir,
                filename,
            )

            plt.figure(
                figsize=(9, 7),
                dpi=150,
            )

            pos = nx.spring_layout(
                graph,
                seed=42,
            )

            node_colors = []

            for node in graph.nodes():

                if node in selected:
                    node_colors.append("#2ecc71")

                elif node in pruned:
                    node_colors.append("#e74c3c")

                else:
                    node_colors.append("#95a5a6")

            nx.draw_networkx_nodes(
                graph,
                pos,
                node_color=node_colors,
                node_size=800,
                edgecolors="#2c3e50",
            )

            nx.draw_networkx_edges(
                graph,
                pos,
                edge_color="#e74c3c",
                width=1.5,
                alpha=0.7,
            )

            labels = {
                node: (
                    f"{node}\n"
                    f"(w={graph.nodes[node]['weight']:.0f})"
                )
                for node in graph.nodes()
            }

            nx.draw_networkx_labels(
                graph,
                pos,
                labels=labels,
                font_size=8,
                font_weight="bold",
                font_color="black",
            )

            plt.title(
                "Grafo de Conflictos de Material Rodante (MWIS)",
                fontsize=12,
                fontweight="bold",
            )

            plt.axis("off")
            plt.tight_layout()

            plt.savefig(
                path,
                bbox_inches="tight",
            )

            plt.close()

            logger.info(
                "Visualización guardada en %s",
                path,
            )

            return path

        except Exception as exc:

            logger.warning(
                "No se pudo generar la imagen del grafo: %s",
                exc,
            )

            return None


# ============================================================
# CONSTRUCCIÓN DEL GRAFO
# ============================================================

def build_conflict_graph(
    nodes_raw: List[Dict[str, Any]],
    edges_raw: List[List[str]],
) -> Tuple[nx.Graph, set]:
    """
    Construye el grafo de conflictos.

    Si se proporcionan edges explícitamente:
        se utilizan esas relaciones.

    Si no se proporcionan:
        se generan conflictos automáticamente cuando dos ciclos
        comparten algún trip.
    """

    graph = nx.Graph()
    all_trips = set()

    # --------------------------------------------------------
    # NODOS
    # --------------------------------------------------------

    for node_data in nodes_raw:

        node_id = node_data["id"]

        trips = node_data.get("trips", [])

        all_trips.update(trips)

        passenger_km = float(
            node_data.get("passenger_km", 0.0)
        )

        empty_km = float(
            node_data.get("empty_km", 0.0)
        )

        # ----------------------------------------------------
        # FUNCIÓN OBJETIVO
        # ----------------------------------------------------
        #
        # Si "weight" está presente, ES el valor objetivo.
        #
        # Si no está presente, se calcula:
        #
        #     weight = 2*passenger_km - empty_km
        #
        # Esto evita mezclar silenciosamente dos definiciones.
        # ----------------------------------------------------

        if "weight" in node_data:

            weight = float(node_data["weight"])

        else:

            weight = (
                2.0 * passenger_km
                - empty_km
            )

        operating_cost = float(
            node_data.get("operating_cost", 0.0)
        )

        graph.add_node(
            node_id,
            weight=weight,
            trips=trips,
            passenger_km=passenger_km,
            empty_km=empty_km,
            operating_cost=operating_cost,
        )

    # --------------------------------------------------------
    # ARISTAS EXPLÍCITAS
    # --------------------------------------------------------

    if edges_raw:

        for edge in edges_raw:

            if not isinstance(edge, (list, tuple)):
                logger.warning(
                    "Arista ignorada: formato inválido %r",
                    edge,
                )
                continue

            if len(edge) != 2:
                logger.warning(
                    "Arista ignorada: se esperaban 2 nodos: %r",
                    edge,
                )
                continue

            # CORRECCIÓN CLAVE:
            #
            # antes:
            #     u, v = e, e[6]
            #
            # correcto:
            #     u, v = e[0], e[1]

            u = edge[0]
            v = edge[1]

            if u not in graph:
                logger.warning(
                    "Arista ignorada: nodo inexistente %s",
                    u,
                )
                continue

            if v not in graph:
                logger.warning(
                    "Arista ignorada: nodo inexistente %s",
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

    # --------------------------------------------------------
    # GENERACIÓN AUTOMÁTICA DE CONFLICTOS
    # --------------------------------------------------------

    else:

        node_list = list(graph.nodes())

        for i in range(len(node_list)):

            for j in range(i + 1, len(node_list)):

                u = node_list[i]
                v = node_list[j]

                trips_u = set(
                    graph.nodes[u]["trips"]
                )

                trips_v = set(
                    graph.nodes[v]["trips"]
                )

                if trips_u & trips_v:
                    graph.add_edge(u, v)

    logger.info(
        "Grafo construido: %d nodos, %d conflictos",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    return graph, all_trips


# ============================================================
# CIRCUITO QAOA
# ============================================================

def build_qaoa_circuit(
    objective: MWISObjective,
    qaoa_depth: int,
) -> QuantumCircuit:
    """
    Construye el circuito QAOA correspondiente al Hamiltoniano
    Ising de MWIS.
    """

    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Qiskit no está disponible."
        )

    if qaoa_depth < 1:
        raise ValueError(
            "qaoa_depth debe ser >= 1."
        )

    n_qubits = objective.num_nodes

    qr = QuantumRegister(
        n_qubits,
        "q",
    )

    cr = ClassicalRegister(
        n_qubits,
        "c",
    )

    qc = QuantumCircuit(
        qr,
        cr,
    )

    # --------------------------------------------------------
    # Estado inicial |+>
    # --------------------------------------------------------

    for i in range(n_qubits):
        qc.h(qr[i])

    # --------------------------------------------------------
    # Parámetros heurísticos
    #
    # No se realiza optimización variacional en esta PoC.
    # --------------------------------------------------------

    gamma_values = [
        0.05 / (layer + 1)
        for layer in range(qaoa_depth)
    ]

    beta_values = [
        0.25 / (layer + 1)
        for layer in range(qaoa_depth)
    ]

    # --------------------------------------------------------
    # Capas QAOA
    # --------------------------------------------------------

    for layer in range(qaoa_depth):

        gamma = gamma_values[layer]
        beta = beta_values[layer]

        # --------------------------------------------
        # Términos lineales
        #
        # h_i = w_i/2 - lambda*degree(i)/4
        # --------------------------------------------

        for i, node in enumerate(objective.nodes):

            degree = objective.graph.degree(node)

            h_i = (
                objective.weights[node] / 2.0
                - (
                    objective.lambda_penalty
                    * degree
                    / 4.0
                )
            )

            qc.rz(
                2.0 * gamma * h_i,
                qr[i],
            )

        # --------------------------------------------
        # Términos ZZ
        #
        # J_ij = lambda / 4
        # --------------------------------------------

        for u, v in objective.graph.edges():

            i = objective.node_index[u]
            j = objective.node_index[v]

            coupling = (
                objective.lambda_penalty / 4.0
            )

            qc.rzz(
                2.0 * gamma * coupling,
                qr[i],
                qr[j],
            )

        # --------------------------------------------
        # Mixer
        # --------------------------------------------

        for i in range(n_qubits):

            qc.rx(
                2.0 * beta,
                qr[i],
            )

    qc.measure(
        qr,
        cr,
    )

    return qc


# ============================================================
# SELECCIÓN DEL MEJOR BITSTRING
# ============================================================

def select_best_qaoa_bitstring(
    counts: Dict[str, int],
    objective: MWISObjective,
) -> Tuple[str, float]:
    """
    Selecciona el estado de menor energía QUBO entre los
    estados realmente observados.

    En caso de empate energético:
        se prefiere el estado con mayor número de shots.

    CORRECCIÓN CLAVE:
        no se utiliza simplemente max(counts, key=counts.get).
    """

    if not counts:
        raise ValueError(
            "QAOA no produjo resultados."
        )

    best_bitstring = min(
        counts.keys(),
        key=lambda bitstring: (
            objective.qubo_energy(bitstring),
            -counts[bitstring],
        ),
    )

    best_energy = objective.qubo_energy(
        best_bitstring
    )

    return (
        best_bitstring,
        best_energy,
    )


# ============================================================
# BITSTRING -> SOLUCIÓN
# ============================================================

def bitstring_to_selected_nodes(
    bitstring: str,
    objective: MWISObjective,
) -> List[str]:

    bits = objective.bitstring_to_bits(
        bitstring
    )

    return [
        objective.nodes[i]
        for i, bit in enumerate(bits)
        if bit == 1
    ]


# ============================================================
# FUNCIÓN PRINCIPAL QCENTROID
# ============================================================

def run(
    input_data: Dict[str, Any],
    solver_params: Dict[str, Any],
    extra_arguments: Dict[str, Any],
) -> Dict[str, Any]:

    logger.info("=" * 70)
    logger.info(
        "Iniciando Solver QCentroid MWIS de Material Rodante"
    )
    logger.info("=" * 70)

    start_time = time.time()

    try:

        # ====================================================
        # PARÁMETROS
        # ====================================================

        iqm_token = (
            solver_params.get("iqm_token")
            or extra_arguments.get("iqm_token")
            or extra_arguments.get("api_token")
            or os.environ.get("IQM_TOKEN")
            or os.environ.get("QCENTROID_TOKEN")
        )

        quantum_computer = solver_params.get(
            "quantum_computer",
            "emerald",
        )

        raw_server_url = solver_params.get(
            "server_url",
            "https://resonance.iqm.tech/",
        )

        server_url = URLSanitizer.sanitize_iqm_url(
            raw_server_url
        )

        shots = int(
            solver_params.get(
                "shots",
                2048,
            )
        )

        qaoa_depth = int(
            solver_params.get(
                "qaoa_depth",
                2,
            )
        )

        penalty_param = solver_params.get(
            "penalty"
        )

        # ====================================================
        # VALIDACIÓN
        # ====================================================

        if shots <= 0:
            raise ValueError(
                "shots debe ser > 0."
            )

        if qaoa_depth <= 0:
            raise ValueError(
                "qaoa_depth debe ser >= 1."
            )

        nodes_raw = input_data.get(
            "nodes",
            [],
        )

        edges_raw = input_data.get(
            "edges",
            [],
        )

        if not nodes_raw:

            return {
                "selected_cycles": [],
                "total_weight": 0.0,
                "is_feasible": False,
                "error": "Dataset de entrada vacío.",
            }

        # ====================================================
        # CONSTRUIR GRAFO
        # ====================================================

        graph, all_trips = build_conflict_graph(
            nodes_raw,
            edges_raw,
        )

        # ====================================================
        # OBJETIVO MWIS
        # ====================================================

        objective = MWISObjective(
            graph,
            penalty_override=penalty_param,
        )

        logger.info(
            "Función objetivo: maximizar sum(weight_i * x_i)"
        )

        logger.info(
            "Penalización QUBO: %.4f",
            objective.lambda_penalty,
        )

        # ====================================================
        # GROUND TRUTH EXACTO
        # ====================================================

        exact_cycles, exact_weight = (
            objective.solve_exact_ground_truth()
        )

        logger.info(
            "Ground Truth exacto: %s | Peso óptimo: %.4f",
            exact_cycles,
            exact_weight,
        )

        # ====================================================
        # BACKEND
        # ====================================================

        backend = None

        backend_info = "Solución exacta clásica"

        job_id = "local"

        # ----------------------------------------------------
        # IQM
        # ----------------------------------------------------

        if iqm_token and IQM_AVAILABLE:

            try:

                logger.info(
                    "Conectando con IQM Resonance (%s)...",
                    quantum_computer,
                )

                provider = IQMProvider(
                    url=server_url,
                    quantum_computer=quantum_computer,
                    token=iqm_token,
                )

                backend = provider.get_backend()

                backend_info = (
                    f"IQM Resonance ({quantum_computer})"
                )

            except Exception as exc:

                logger.warning(
                    "No se pudo conectar con IQM: %s",
                    exc,
                )

        # ----------------------------------------------------
        # AER
        # ----------------------------------------------------

        if backend is None and AER_AVAILABLE:

            backend = AerSimulator()

            backend_info = (
                "Qiskit AerSimulator (CPU)"
            )

        # ====================================================
        # SOLUCIÓN
        # ====================================================

        best_bitstring = "N/A"
        best_qubo_energy: Optional[float] = None

        raw_selected: List[str]

        if backend is not None and QISKIT_AVAILABLE:

            # ------------------------------------------------
            # Construcción QAOA
            # ------------------------------------------------

            qc = build_qaoa_circuit(
                objective,
                qaoa_depth,
            )

            qc_transpiled = transpile(
                qc,
                backend=backend,
                optimization_level=2,
            )

            logger.info(
                "Ejecutando QAOA p=%d en %s (%d shots)...",
                qaoa_depth,
                backend_info,
                shots,
            )

            q_job = backend.run(
                qc_transpiled,
                shots=shots,
            )

            job_id_attr = getattr(
                q_job,
                "job_id",
                None,
            )

            if callable(job_id_attr):
                job_id = job_id_attr()

            elif job_id_attr is not None:
                job_id = str(job_id_attr)

            result = q_job.result()

            counts = result.get_counts()

            # ------------------------------------------------
            # CORRECCIÓN CLAVE:
            #
            # Seleccionar por MENOR ENERGÍA QUBO.
            # ------------------------------------------------

            best_bitstring, best_qubo_energy = (
                select_best_qaoa_bitstring(
                    counts,
                    objective,
                )
            )

            raw_selected = (
                bitstring_to_selected_nodes(
                    best_bitstring,
                    objective,
                )
            )

            logger.info(
                "Mejor bitstring observado: %s",
                best_bitstring,
            )

            logger.info(
                "Energía QUBO: %.4f",
                best_qubo_energy,
            )

            logger.info(
                "Solución QAOA inicial: %s",
                raw_selected,
            )

        else:

            # ------------------------------------------------
            # Fallback clásico
            #
            # Utilizamos el Ground Truth exacto para garantizar
            # que la PoC siga siendo funcional sin Qiskit.
            # ------------------------------------------------

            raw_selected = exact_cycles

            backend_info = (
                "Solución exacta clásica "
                "(fallback sin backend cuántico)"
            )

            logger.info(
                "Qiskit/backend no disponible. "
                "Usando Ground Truth exacto como fallback."
            )

        # ====================================================
        # REPARACIÓN / PODA
        # ====================================================

        final_selected, pruned_nodes = (
            MWISPruner.prune(
                graph,
                raw_selected,
            )
        )

        # ====================================================
        # MÉTRICAS
        # ====================================================

        final_weight = (
            objective.solution_weight(
                final_selected
            )
        )

        covered_trips = set()

        total_passenger_km = 0.0
        total_empty_km = 0.0
        total_operating_cost = 0.0

        for node in final_selected:

            data = graph.nodes[node]

            covered_trips.update(
                data["trips"]
            )

            total_passenger_km += (
                data["passenger_km"]
            )

            total_empty_km += (
                data["empty_km"]
            )

            total_operating_cost += (
                data["operating_cost"]
            )

        coverage_rate = (
            len(covered_trips)
            / len(all_trips)
            if all_trips
            else 1.0
        )

        feasible = objective.is_feasible(
            final_selected
        )

        exec_time = (
            time.time()
            - start_time
        )

        # ====================================================
        # GAP DE OPTIMALIDAD
        # ====================================================

        if exact_weight > 0:

            optimality_gap = (
                (exact_weight - final_weight)
                / exact_weight
                * 100.0
            )

        else:

            optimality_gap = 0.0

        # ====================================================
        # VISUALIZACIÓN
        # ====================================================

        asset_path = (
            VisualizationAssetGenerator
            .generate_conflict_graph_visualization(
                graph,
                final_selected,
                pruned_nodes,
            )
        )

        assets = (
            {"conflict_graph_png": asset_path}
            if asset_path
            else {}
        )

        # ====================================================
        # RESPUESTA QCENTROID
        # ====================================================

        response = {

            "selected_cycles": final_selected,

            "total_weight": final_weight,

            "coverage_rate": round(
                coverage_rate,
                4,
            ),

            "coverage_percent": round(
                coverage_rate * 100.0,
                2,
            ),

            "covered_trips": len(
                covered_trips
            ),

            "scheduled_trips": len(
                all_trips
            ),

            "total_empty_km": (
                total_empty_km
            ),

            "total_passenger_km": (
                total_passenger_km
            ),

            "total_operating_cost": (
                total_operating_cost
            ),

            "is_feasible": feasible,

            "nodes_pruned": len(
                pruned_nodes
            ),

            # ------------------------------------------------
            # Referencia exacta
            # ------------------------------------------------

            "ground_truth_exact": {

                "selected_cycles": (
                    exact_cycles
                ),

                "total_weight": (
                    exact_weight
                ),

                "optimality_gap_percent": round(
                    optimality_gap,
                    2,
                ),
            },

            # ------------------------------------------------
            # Ejecución
            # ------------------------------------------------

            "execution_metrics": {

                "execution_time_seconds": (
                    exec_time
                ),

                "initial_solution_size": (
                    len(raw_selected)
                ),

                "final_solution_size": (
                    len(final_selected)
                ),

                "initial_bitstring_min_qubo": (
                    best_bitstring
                ),

                "initial_qubo_energy": (
                    best_qubo_energy
                ),

                "qaoa_depth": (
                    qaoa_depth
                ),

                "shots": (
                    shots
                ),

                "penalty": (
                    objective.lambda_penalty
                ),

                "job_id": (
                    job_id
                ),
            },

            # ------------------------------------------------
            # Poda
            # ------------------------------------------------

            "pruning_stats": {

                "nodes_pruned_count": (
                    len(pruned_nodes)
                ),

                "pruned_nodes_list": (
                    pruned_nodes
                ),
            },

            # ------------------------------------------------
            # Backend
            # ------------------------------------------------

            "backend_used": (
                backend_info
            ),

            # ------------------------------------------------
            # Assets
            # ------------------------------------------------

            "assets": assets,
        }

        logger.info(
            "Ejecución completada."
        )

        logger.info(
            "Solución final: %s",
            final_selected,
        )

        logger.info(
            "Peso final: %.4f",
            final_weight,
        )

        logger.info(
            "Ground Truth: %s",
            exact_cycles,
        )

        logger.info(
            "Gap de optimalidad: %.2f%%",
            optimality_gap,
        )

        return response

    except Exception as exc:

        logger.exception(
            "Fallo durante la ejecución del solver."
        )

        return {
            "selected_cycles": [],
            "total_weight": 0.0,
            "is_feasible": False,
            "error": str(exc),
        }
