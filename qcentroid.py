import os
import time
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    from qiskit import QuantumCircuit, transpile
except ImportError:
    QuantumCircuit = None
    transpile = None

try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None

try:
    from iqm.qiskit_iqm import IQMProvider
except ImportError:
    IQMProvider = None


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# URL SANITIZER
# ============================================================

class URLSanitizer:
    """
    Keeps URL-related configuration out of logs/results when possible.
    """

    @staticmethod
    def sanitize(url):
        if not url:
            return None

        return str(url).rstrip("/")


# ============================================================
# MWIS OBJECTIVE
# ============================================================

class MWISObjective:
    """
    Maximum Weighted Independent Set objective.

    The explicit 'weight' field is the optimization objective.

    If weight is not supplied, a fallback objective is calculated as:

        2 * passenger_km - empty_km

    passenger_km, empty_km and operating_cost are operational
    metrics and are not included in the optimization objective
    when an explicit weight is supplied.
    """

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

        self.graph = nx.Graph()

        self.weights = {}
        self.node_data = {}

        for node in nodes:
            node_id = node["id"]

            self.graph.add_node(node_id)

            if "weight" in node:
                weight = float(node["weight"])
            else:
                weight = (
                    2.0 * float(node.get("passenger_km", 0.0))
                    - float(node.get("empty_km", 0.0))
                )

            self.weights[node_id] = weight
            self.node_data[node_id] = node

        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError(
                    f"Invalid edge format: {edge}. "
                    "Edges must contain exactly two node IDs."
                )

            u = edge[0]
            v = edge[1]

            if u not in self.graph.nodes:
                raise ValueError(
                    f"Edge references unknown node: {u}"
                )

            if v not in self.graph.nodes:
                raise ValueError(
                    f"Edge references unknown node: {v}"
                )

            self.graph.add_edge(u, v)

    def objective_value(self, selected_nodes):
        return float(
            sum(
                self.weights[node]
                for node in selected_nodes
            )
        )

    def qubo_energy(self, selected_nodes, penalty):
        """
        QUBO:

            H(x) =
                - sum_i w_i x_i
                + penalty * sum_(i,j) x_i x_j

        A selected conflict therefore incurs the penalty.
        """

        selected = set(selected_nodes)

        energy = -self.objective_value(selected)

        for u, v in self.graph.edges():
            if u in selected and v in selected:
                energy += penalty

        return float(energy)

    def exact_solution(self):
        """
        Exact MWIS through maximum-weight clique on the
        complement graph.

        NetworkX max_weight_clique requires integer node weights
        in the installed NetworkX version, therefore the weights
        are converted to integers for the exact reference.

        The original floating-point weights are then used to
        calculate the final objective value.
        """

        complement = nx.complement(self.graph)

        for node in complement.nodes:
            complement.nodes[node]["weight"] = int(
                round(self.weights[node])
            )

        clique, _ = nx.algorithms.clique.max_weight_clique(
            complement,
            weight="weight",
        )

        selected_nodes = list(clique)

        total_weight = self.objective_value(
            selected_nodes
        )

        return selected_nodes, float(total_weight)


# ============================================================
# MWIS PRUNER / REPAIR
# ============================================================

class MWISPruner:
    """
    Deterministic repair step.

    If a measured bitstring contains conflicts, the selected
    nodes are repaired by keeping nodes according to:

        weight / (degree + 1)

    This is only a feasibility repair heuristic.
    """

    def __init__(self, graph, weights):
        self.graph = graph
        self.weights = weights

    def prune_solution(self, selected_nodes):
        selected = set(selected_nodes)
        pruned_nodes = []

        while True:
            conflicts = []

            for u, v in self.graph.edges():
                if u in selected and v in selected:
                    conflicts.append((u, v))

            if not conflicts:
                break

            conflicting_nodes = set()

            for u, v in conflicts:
                conflicting_nodes.add(u)
                conflicting_nodes.add(v)

            ranked = sorted(
                conflicting_nodes,
                key=lambda node: (
                    self.weights[node]
                    / (self.graph.degree(node) + 1)
                ),
                reverse=True,
            )

            keep = ranked[0]

            for node in conflicting_nodes:
                if node != keep:
                    selected.discard(node)
                    pruned_nodes.append(node)

        return list(selected), pruned_nodes


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

class VisualizationAssetGenerator:

    @staticmethod
    def generate(graph, weights, selected_nodes):
        output_dir = Path("additional_output")
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir / "conflict_graph.png"
        )

        plt.figure(
            figsize=(9, 7),
            dpi=150,
        )

        positions = nx.spring_layout(
            graph,
            seed=42,
        )

        node_colors = []

        selected_set = set(selected_nodes)

        for node in graph.nodes:
            if node in selected_set:
                node_colors.append("green")
            else:
                node_colors.append("lightgray")

        nx.draw_networkx_edges(
            graph,
            positions,
            alpha=0.5,
        )

        nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=node_colors,
            node_size=1600,
            edgecolors="black",
        )

        labels = {
            node: f"{node}\nW={weights[node]:.0f}"
            for node in graph.nodes
        }

        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=9,
        )

        plt.title(
            "Railway Rolling Stock Cycle Conflict Graph"
        )

        plt.axis("off")

        plt.savefig(
            output_path,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            "Visualización guardada en %s",
            output_path,
        )

        return {
            "conflict_graph_png": str(output_path)
        }


# ============================================================
# QAOA CIRCUIT
# ============================================================

def build_qaoa_circuit(
    objective,
    penalty,
    qaoa_depth=2,
    gamma=0.2,
    beta=0.5,
):
    """
    Build a QAOA circuit from the MWIS QUBO.

    QUBO:

        H = -sum(w_i x_i)
            + penalty * sum(x_i x_j)

    Using:

        x_i = (1 - Z_i) / 2

    the QUBO is transformed into:

        H = constant
            + sum(h_i Z_i)
            + sum(J_ij Z_i Z_j)

    The constant can be ignored during QAOA because it does
    not affect the argmin.

    The circuit uses:

        RZ(2 * gamma * h_i)
        RZZ(2 * gamma * J_ij)
        RX(2 * beta)

    for each QAOA layer.
    """

    if QuantumCircuit is None:
        raise RuntimeError(
            "Qiskit is not installed."
        )

    nodes = list(objective.graph.nodes)
    n_qubits = len(nodes)

    index = {
        node: i
        for i, node in enumerate(nodes)
    }

    circuit = QuantumCircuit(
        n_qubits,
        n_qubits,
    )

    # Initial |+> state.
    for qubit in range(n_qubits):
        circuit.h(qubit)

    # QUBO -> Ising coefficients.
    h = {
        node: objective.weights[node] / 2.0
        for node in nodes
    }

    j = {}

    for u, v in objective.graph.edges():

        # x_u x_v =
        # 1/4 (1 - Z_u - Z_v + Z_u Z_v)

        h[u] -= penalty / 4.0
        h[v] -= penalty / 4.0

        j[(u, v)] = penalty / 4.0

    for _ in range(qaoa_depth):

        # Cost Hamiltonian: single-qubit Z terms.
        for node in nodes:
            qubit = index[node]

            circuit.rz(
                2.0 * gamma * h[node],
                qubit,
            )

        # Cost Hamiltonian: ZZ terms.
        for (u, v), coefficient in j.items():

            circuit.rzz(
                2.0 * gamma * coefficient,
                index[u],
                index[v],
            )

        # Mixer Hamiltonian.
        for qubit in range(n_qubits):
            circuit.rx(
                2.0 * beta,
                qubit,
            )

    circuit.measure(
        range(n_qubits),
        range(n_qubits),
    )

    return circuit


# ============================================================
# BITSTRING UTILITIES
# ============================================================

def normalize_bitstring(bitstring, n_qubits):
    """
    Normalize Qiskit/IQM bitstring representation.
    """

    bitstring = str(bitstring).replace(
        " ",
        "",
    )

    if len(bitstring) < n_qubits:
        bitstring = bitstring.zfill(n_qubits)

    if len(bitstring) > n_qubits:
        bitstring = bitstring[-n_qubits:]

    return bitstring


def bitstring_to_selected_nodes(
    bitstring,
    nodes,
):
    """
    Convert measured bitstring to selected MWIS nodes.

    Qiskit classical-bit strings are conventionally displayed
    with the highest-index classical bit on the left.

    Therefore bitstring position [::-1] is mapped to the
    logical node order.
    """

    bitstring = normalize_bitstring(
        bitstring,
        len(nodes),
    )

    bits = bitstring[::-1]

    selected = []

    for i, bit in enumerate(bits):

        if bit == "1":
            selected.append(nodes[i])

    return selected


# ============================================================
# SELECT BEST QAOA BITSTRING
# ============================================================

def select_best_qaoa_bitstring(
    counts,
    objective,
    penalty,
):
    """
    Select the measured bitstring with the lowest QUBO energy.

    IMPORTANT:

    We do NOT select the most frequent bitstring.

    QAOA is an optimization algorithm, so the measured samples
    are evaluated against the actual QUBO objective.

    Tie-breaker:
        if several bitstrings have the same energy, choose the
        one with the highest number of shots.
    """

    nodes = list(objective.graph.nodes)

    candidates = []

    for bitstring, shots in counts.items():

        selected = bitstring_to_selected_nodes(
            bitstring,
            nodes,
        )

        energy = objective.qubo_energy(
            selected,
            penalty,
        )

        candidates.append(
            (
                float(energy),
                -int(shots),
                bitstring,
                selected,
                int(shots),
            )
        )

    if not candidates:
        raise RuntimeError(
            "QAOA returned no measurement counts."
        )

    best = min(candidates)

    return {
        "bitstring": best[2],
        "selected_nodes": best[3],
        "qubo_energy": best[0],
        "shots": best[4],
    }


# ============================================================
# QAOA EXECUTION
# ============================================================

def execute_qaoa(
    circuit,
    shots,
    extra_arguments,
):
    """
    Execute the QAOA circuit.

    Preferred backend:
        IQM Resonance

    Fallback:
        local Aer simulator.

    IQM configuration is taken from environment variables:

        IQM_TOKEN
        IQM_SERVER_URL
        IQM_QUANTUM_COMPUTER

    Example:

        IQM_SERVER_URL=https://resonance.iqm.tech/
        IQM_QUANTUM_COMPUTER=emerald

    The token must NOT be embedded in source code.
    """

    use_iqm = bool(
        extra_arguments.get(
            "use_iqm",
            True,
        )
    )

    iqm_url = (
        extra_arguments.get("iqm_server_url")
        or os.getenv("IQM_SERVER_URL")
    )

    quantum_computer = (
        extra_arguments.get("quantum_computer")
        or os.getenv(
            "IQM_QUANTUM_COMPUTER",
            "emerald",
        )
    )

    iqm_token = os.getenv("IQM_TOKEN")

    # --------------------------------------------------------
    # IQM Resonance
    # --------------------------------------------------------

    if (
        use_iqm
        and IQMProvider is not None
        and iqm_url
        and iqm_token
    ):

        logger.info(
            "Ejecutando QAOA en IQM Resonance (%s)",
            quantum_computer,
        )

        provider = IQMProvider(
            iqm_url,
            quantum_computer=quantum_computer,
            token=iqm_token,
        )

        backend = provider.get_backend()

        if transpile is None:
            raise RuntimeError(
                "Qiskit transpile is not available."
            )

        transpiled_circuit = transpile(
            circuit,
            backend=backend,
        )

        job = backend.run(
            transpiled_circuit,
            shots=shots,
        )

        result = job.result()

        counts = result.get_counts()

        job_id = None

        try:
            job_id = job.job_id()
        except Exception:
            pass

        return (
            counts,
            f"IQM Resonance ({quantum_computer})",
            job_id,
        )

    # --------------------------------------------------------
    # Local Aer fallback
    # --------------------------------------------------------

    if AerSimulator is not None:

        logger.info(
            "IQM no disponible; usando AerSimulator local."
        )

        simulator = AerSimulator()

        job = simulator.run(
            circuit,
            shots=shots,
        )

        result = job.result()

        counts = result.get_counts()

        return (
            counts,
            "Qiskit AerSimulator",
            None,
        )

    raise RuntimeError(
        "No quantum backend available. "
        "Install qiskit-aer or configure IQM Resonance."
    )


# ============================================================
# OPERATIONAL METRICS
# ============================================================

def calculate_operational_metrics(
    selected_nodes,
    node_data,
):
    total_passenger_km = 0.0
    total_empty_km = 0.0
    total_operating_cost = 0.0

    scheduled_trips = set()
    covered_trips = set()

    for node in node_data.values():

        for trip in node.get("trips", []):
            scheduled_trips.add(trip)

    for node_id in selected_nodes:

        data = node_data[node_id]

        total_passenger_km += float(
            data.get(
                "passenger_km",
                0.0,
            )
        )

        total_empty_km += float(
            data.get(
                "empty_km",
                0.0,
            )
        )

        total_operating_cost += float(
            data.get(
                "operating_cost",
                0.0,
            )
        )

        for trip in data.get("trips", []):
            covered_trips.add(trip)

    scheduled_count = len(scheduled_trips)
    covered_count = len(
        covered_trips.intersection(
            scheduled_trips
        )
    )

    if scheduled_count > 0:
        coverage_percent = (
            covered_count
            / scheduled_count
            * 100.0
        )
    else:
        coverage_percent = 0.0

    return {
        "total_passenger_km": float(
            total_passenger_km
        ),
        "total_empty_km": float(
            total_empty_km
        ),
        "total_operating_cost": float(
            total_operating_cost
        ),
        "scheduled_trips": int(
            scheduled_count
        ),
        "covered_trips": int(
            covered_count
        ),
        "coverage_percent": float(
            coverage_percent
        ),
    }


# ============================================================
# MAIN SOLVER
# ============================================================

def run(
    input_data,
    solver_params=None,
    extra_arguments=None,
):
    """
    QCentroid solver entry point.
    """

    start_time = time.perf_counter()

    solver_params = solver_params or {}
    extra_arguments = extra_arguments or {}

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    nodes = input_data.get(
        "nodes",
        [],
    )

    edges = input_data.get(
        "edges",
        [],
    )

    if not nodes:
        raise ValueError(
            "Input data contains no nodes."
        )

    logger.info(
        "Construyendo grafo de conflictos: "
        "%d nodos, %d conflictos.",
        len(nodes),
        len(edges),
    )

    objective = MWISObjective(
        nodes,
        edges,
    )

    logger.info(
        "Objetivo: maximizar suma(weight_i * x_i)."
    )

    # --------------------------------------------------------
    # QUBO penalty
    # --------------------------------------------------------

    max_weight = max(
        objective.weights.values()
    )

    penalty = float(
        solver_params.get(
            "penalty",
            4.0 * max_weight,
        )
    )

    logger.info(
        "Penalty QUBO: %.4f",
        penalty,
    )

    # --------------------------------------------------------
    # QAOA parameters
    # --------------------------------------------------------

    qaoa_depth = int(
        solver_params.get(
            "qaoa_depth",
            2,
        )
    )

    shots = int(
        solver_params.get(
            "shots",
            2048,
        )
    )

    gamma = float(
        solver_params.get(
            "gamma",
            0.2,
        )
    )

    beta = float(
        solver_params.get(
            "beta",
            0.5,
        )
    )

    logger.info(
        "QAOA: depth=%d, shots=%d, gamma=%.4f, beta=%.4f",
        qaoa_depth,
        shots,
        gamma,
        beta,
    )

    # --------------------------------------------------------
    # Build QAOA circuit
    # --------------------------------------------------------

    circuit = build_qaoa_circuit(
        objective=objective,
        penalty=penalty,
        qaoa_depth=qaoa_depth,
        gamma=gamma,
        beta=beta,
    )

    # --------------------------------------------------------
    # Execute QAOA
    # --------------------------------------------------------

    counts, backend_used, job_id = execute_qaoa(
        circuit=circuit,
        shots=shots,
        extra_arguments=extra_arguments,
    )

    logger.info(
        "Backend utilizado: %s",
        backend_used,
    )

    logger.info(
        "Measurement counts: %s",
        counts,
    )

    # --------------------------------------------------------
    # Evaluate all observed bitstrings
    # --------------------------------------------------------

    best_qaoa = select_best_qaoa_bitstring(
        counts=counts,
        objective=objective,
        penalty=penalty,
    )

    initial_bitstring = best_qaoa[
        "bitstring"
    ]

    initial_selected_nodes = best_qaoa[
        "selected_nodes"
    ]

    initial_qubo_energy = best_qaoa[
        "qubo_energy"
    ]

    initial_solution_size = len(
        initial_selected_nodes
    )

    logger.info(
        "Mejor bitstring QAOA por energía QUBO: %s",
        initial_bitstring,
    )

    logger.info(
        "Energía QUBO: %.4f",
        initial_qubo_energy,
    )

    # --------------------------------------------------------
    # Feasibility repair
    # --------------------------------------------------------

    pruner = MWISPruner(
        objective.graph,
        objective.weights,
    )

    selected_nodes, pruned_nodes = (
        pruner.prune_solution(
            initial_selected_nodes
        )
    )

    nodes_pruned = len(
        pruned_nodes
    )

    if nodes_pruned > 0:
        logger.info(
            "Se eliminaron %d nodos para reparar conflictos.",
            nodes_pruned,
        )

    # --------------------------------------------------------
    # Final objective
    # --------------------------------------------------------

    total_weight = objective.objective_value(
        selected_nodes
    )

    final_qubo_energy = objective.qubo_energy(
        selected_nodes,
        penalty,
    )

    # --------------------------------------------------------
    # Feasibility
    # --------------------------------------------------------

    is_feasible = True

    selected_set = set(
        selected_nodes
    )

    for u, v in objective.graph.edges():

        if (
            u in selected_set
            and v in selected_set
        ):
            is_feasible = False
            break

    # --------------------------------------------------------
    # Exact classical ground truth
    # --------------------------------------------------------

    ground_truth_exact = None
    optimality_gap_percent = None

    try:

        exact_selected_nodes, exact_total_weight = (
            objective.exact_solution()
        )

        ground_truth_exact = {
            "selected_cycles": exact_selected_nodes,
            "total_weight": float(
                exact_total_weight
            ),
        }

        # ----------------------------------------------------
        # IMPORTANT BENCHMARK CALCULATION
        #
        # gap (%) =
        #
        #     (OPT - SOL) / OPT * 100
        #
        # Therefore:
        #
        #     SOL == OPT  -> 0.0 %
        # ----------------------------------------------------

        if (
            exact_total_weight is not None
            and float(exact_total_weight) > 0.0
        ):

            optimality_gap_percent = (
                (
                    float(exact_total_weight)
                    - float(total_weight)
                )
                / float(exact_total_weight)
            ) * 100.0

            # Remove floating point noise.
            if abs(
                optimality_gap_percent
            ) < 1e-10:

                optimality_gap_percent = 0.0

            # A solution cannot have a negative
            # optimality gap relative to the exact optimum.
            optimality_gap_percent = max(
                0.0,
                float(
                    optimality_gap_percent
                ),
            )

            optimality_gap_percent = float(
                optimality_gap_percent
            )

        else:
            optimality_gap_percent = None

        ground_truth_exact[
            "optimality_gap_percent"
        ] = (
            float(optimality_gap_percent)
            if optimality_gap_percent is not None
            else None
        )

        logger.info(
            "Ground truth exacto: %s | weight=%.4f",
            exact_selected_nodes,
            exact_total_weight,
        )

        logger.info(
            "Optimality gap: %s%%",
            optimality_gap_percent,
        )

    except Exception as exc:

        logger.warning(
            "No se pudo calcular el ground truth exacto: %s",
            exc,
        )

        ground_truth_exact = {
            "selected_cycles": [],
            "total_weight": None,
            "optimality_gap_percent": None,
        }

    # --------------------------------------------------------
    # Operational metrics
    # --------------------------------------------------------

    operational = calculate_operational_metrics(
        selected_nodes=selected_nodes,
        node_data=objective.node_data,
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    assets = {}

    try:

        assets = (
            VisualizationAssetGenerator.generate(
                graph=objective.graph,
                weights=objective.weights,
                selected_nodes=selected_nodes,
            )
        )

    except Exception as exc:

        logger.warning(
            "No se pudo generar la visualización: %s",
            exc,
        )

    # --------------------------------------------------------
    # Execution time
    # --------------------------------------------------------

    execution_time_seconds = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Benchmark logging
    # --------------------------------------------------------

    logger.info(
        "BENCHMARK METRICS | "
        "total_weight=%.4f | "
        "optimality_gap_percent=%s | "
        "coverage_percent=%.4f | "
        "execution_time_seconds=%.6f | "
        "total_operating_cost=%.4f",
        float(total_weight),
        (
            f"{optimality_gap_percent:.4f}"
            if optimality_gap_percent is not None
            else "None"
        ),
        float(
            operational[
                "coverage_percent"
            ]
        ),
        float(
            execution_time_seconds
        ),
        float(
            operational[
                "total_operating_cost"
            ]
        ),
    )

    # ========================================================
    # FINAL RESULT
    #
    # IMPORTANT:
    #
    # These five fields MUST be at top level because they are
    # the QCentroid output benchmark parameters.
    # ========================================================

    result = {

        # ----------------------------------------------------
        # QCentroid benchmark metrics
        # ----------------------------------------------------

        "total_weight": float(
            total_weight
        ),

        "optimality_gap_percent": (
            float(
                optimality_gap_percent
            )
            if optimality_gap_percent is not None
            else None
        ),

        "coverage_percent": float(
            operational[
                "coverage_percent"
            ]
        ),

        "execution_time_seconds": float(
            execution_time_seconds
        ),

        "total_operating_cost": float(
            operational[
                "total_operating_cost"
            ]
        ),

        # ----------------------------------------------------
        # Solution
        # ----------------------------------------------------

        "selected_cycles": list(
            selected_nodes
        ),

        "scheduled_trips": int(
            operational[
                "scheduled_trips"
            ]
        ),

        "covered_trips": int(
            operational[
                "covered_trips"
            ]
        ),

        "is_feasible": bool(
            is_feasible
        ),

        "backend_used": backend_used,

        # ----------------------------------------------------
        # Exact classical reference
        # ----------------------------------------------------

        "ground_truth_exact": (
            ground_truth_exact
        ),

        # ----------------------------------------------------
        # Operational metrics
        # ----------------------------------------------------

        "total_passenger_km": float(
            operational[
                "total_passenger_km"
            ]
        ),

        "total_empty_km": float(
            operational[
                "total_empty_km"
            ]
        ),

        # ----------------------------------------------------
        # Execution metrics
        # ----------------------------------------------------

        "execution_metrics": {

            "execution_time_seconds": float(
                execution_time_seconds
            ),

            "final_solution_size": int(
                len(selected_nodes)
            ),

            "initial_bitstring_min_qubo": (
                initial_bitstring
            ),

            "initial_qubo_energy": float(
                initial_qubo_energy
            ),

            "final_qubo_energy": float(
                final_qubo_energy
            ),

            "initial_solution_size": int(
                initial_solution_size
            ),

            "penalty": float(
                penalty
            ),

            "qaoa_depth": int(
                qaoa_depth
            ),

            "shots": int(
                shots
            ),

            "job_id": job_id,
        },

        # ----------------------------------------------------
        # Pruning
        # ----------------------------------------------------

        "nodes_pruned": int(
            nodes_pruned
        ),

        "pruning_stats": {
            "nodes_pruned_count": int(
                nodes_pruned
            ),
            "pruned_nodes_list": list(
                pruned_nodes
            ),
        },

        # ----------------------------------------------------
        # Assets
        # ----------------------------------------------------

        "assets": assets,
    }

    # --------------------------------------------------------
    # Final diagnostic
    # --------------------------------------------------------

    logger.info(
        "Resultado final: selected_cycles=%s | "
        "total_weight=%.4f | "
        "optimality_gap_percent=%s | "
        "coverage_percent=%.2f | "
        "feasible=%s",
        selected_nodes,
        total_weight,
        optimality_gap_percent,
        operational[
            "coverage_percent"
        ],
        is_feasible,
    )

    return result

 
