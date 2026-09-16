import os
import time
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
except ImportError:
    QuantumCircuit = None
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
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# URL SANITIZER
# ============================================================

class URLSanitizer:
    """
    Sanitizes backend URLs and prevents accidental exposure of
    credentials in logs.
    """

    @staticmethod
    def sanitize(url):
        if not url:
            return url

        for token_name in [
            "IQM_TOKEN",
            "IQM_API_TOKEN",
            "API_TOKEN",
            "TOKEN"
        ]:
            token = os.getenv(token_name)
            if token and token in url:
                url = url.replace(token, "***")

        return url


# ============================================================
# MWIS OBJECTIVE
# ============================================================

class MWISObjective:
    """
    Defines the MWIS objective.

    If an explicit 'weight' is supplied for every node, it is used
    as the optimization objective.

    Otherwise, a fallback objective is calculated as:

        2 * passenger_km - empty_km
    """

    def __init__(self, nodes):
        self.nodes = nodes

    def calculate_weights(self):
        weights = {}

        for node in self.nodes:
            node_id = node["id"]

            if "weight" in node:
                weights[node_id] = float(node["weight"])
            else:
                passenger_km = float(node.get("passenger_km", 0))
                empty_km = float(node.get("empty_km", 0))

                weights[node_id] = (
                    2.0 * passenger_km - empty_km
                )

        return weights


# ============================================================
# PRUNER / REPAIR
# ============================================================

class MWISPruner:
    """
    Deterministic repair of an infeasible solution.

    Nodes are considered according to weight / degree ratio.
    """

    @staticmethod
    def prune_solution(graph, selected_nodes, weights):
        selected = set(selected_nodes)

        while True:
            conflicts = []

            for u in selected:
                for v in selected:
                    if u != v and graph.has_edge(u, v):
                        conflicts.append((u, v))

            if not conflicts:
                break

            # Determine the node to remove from the first conflict.
            u, v = conflicts[0]

            degree_u = max(graph.degree(u), 1)
            degree_v = max(graph.degree(v), 1)

            score_u = weights[u] / degree_u
            score_v = weights[v] / degree_v

            if score_u < score_v:
                selected.remove(u)
            else:
                selected.remove(v)

        return sorted(selected)


# ============================================================
# VISUALIZATION
# ============================================================

class VisualizationAssetGenerator:

    @staticmethod
    def generate_conflict_graph(graph, selected_nodes):
        output_dir = Path("additional_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "conflict_graph.png"

        plt.figure(figsize=(9, 7), dpi=150)

        pos = nx.spring_layout(
            graph,
            seed=42
        )

        selected_nodes = set(selected_nodes)

        node_sizes = []
        for node in graph.nodes:
            if node in selected_nodes:
                node_sizes.append(900)
            else:
                node_sizes.append(600)

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=node_sizes
        )

        nx.draw_networkx_edges(
            graph,
            pos,
            width=1.5
        )

        nx.draw_networkx_labels(
            graph,
            pos,
            font_size=10
        )

        plt.title(
            "Railway Rolling Stock Cycle Conflict Graph"
        )

        plt.axis("off")
        plt.savefig(
            path,
            bbox_inches="tight"
        )
        plt.close()

        logger.info(
            "Visualización guardada en %s",
            path
        )

        return str(path)


# ============================================================
# GRAPH
# ============================================================

def build_conflict_graph(input_data):
    nodes = input_data.get("nodes", [])
    edges = input_data.get("edges", [])

    graph = nx.Graph()

    for node in nodes:
        node_id = node["id"]
        graph.add_node(node_id)

    for edge in edges:
        if len(edge) != 2:
            raise ValueError(
                f"Invalid edge format: {edge}"
            )

        u = edge[0]
        v = edge[1]

        if u not in graph:
            raise ValueError(
                f"Unknown node in edge: {u}"
            )

        if v not in graph:
            raise ValueError(
                f"Unknown node in edge: {v}"
            )

        graph.add_edge(u, v)

    return graph


# ============================================================
# QUBO
# ============================================================

def calculate_penalty(weights):
    """
    Penalty chosen as:

        lambda = 4 * max(weight)

    This is sufficiently large for the current MWIS formulation.
    """

    if not weights:
        return 1.0

    return 4.0 * max(weights.values())


def calculate_qubo_energy(
    bitstring,
    graph,
    weights,
    penalty
):
    """
    H(x) =
        - sum(w_i x_i)
        + lambda sum(x_i x_j)
    """

    nodes = list(graph.nodes)

    if len(bitstring) != len(nodes):
        raise ValueError(
            "Bitstring length does not match number of nodes"
        )

    energy = 0.0

    selected = {}

    for index, node in enumerate(nodes):
        value = int(bitstring[index])
        selected[node] = value

        energy -= weights[node] * value

    for u, v in graph.edges:
        energy += (
            penalty
            * selected[u]
            * selected[v]
        )

    return float(energy)


# ============================================================
# QAOA CIRCUIT
# ============================================================

def build_qaoa_circuit(
    graph,
    weights,
    penalty,
    depth=2
):
    """
    Builds a simple QAOA circuit for the MWIS QUBO.

    The parameters are intentionally fixed/heuristic for this PoC.
    """

    if QuantumCircuit is None:
        raise RuntimeError(
            "Qiskit is not available"
        )

    nodes = list(graph.nodes)
    n = len(nodes)

    node_index = {
        node: index
        for index, node in enumerate(nodes)
    }

    qc = QuantumCircuit(n, n)

    # Initial |+> state
    for q in range(n):
        qc.h(q)

    # Heuristic QAOA parameters
    beta = 0.35
    gamma = 0.20

    for _ in range(depth):

        # ----------------------------------------------------
        # Cost Hamiltonian
        # ----------------------------------------------------

        for node in nodes:
            q = node_index[node]

            # QUBO linear term:
            #
            # -weight * x
            #
            # x = (1-Z)/2
            #
            # The exact constant is irrelevant for optimization.
            angle = gamma * weights[node]

            qc.rz(
                2.0 * angle,
                q
            )

        for u, v in graph.edges:
            qu = node_index[u]
            qv = node_index[v]

            qc.cx(qu, qv)
            qc.rz(
                2.0 * gamma * penalty,
                qv
            )
            qc.cx(qu, qv)

        # ----------------------------------------------------
        # Mixer
        # ----------------------------------------------------

        for q in range(n):
            qc.rx(
                2.0 * beta,
                q
            )

    qc.measure(
        range(n),
        range(n)
    )

    return qc


# ============================================================
# QAOA BITSTRING SELECTION
# ============================================================

def select_best_qaoa_bitstring(
    counts,
    graph,
    weights,
    penalty
):
    """
    Selects the observed bitstring with the minimum QUBO energy.

    If several bitstrings have the same energy, the one with the
    largest number of shots is selected.
    """

    candidates = []

    for bitstring, shots in counts.items():

        clean_bitstring = bitstring.replace(" ", "")

        energy = calculate_qubo_energy(
            clean_bitstring,
            graph,
            weights,
            penalty
        )

        candidates.append(
            (
                energy,
                -shots,
                clean_bitstring
            )
        )

    if not candidates:
        raise RuntimeError(
            "No QAOA measurement results were returned"
        )

    candidates.sort()

    best_energy = candidates[0][0]
    best_bitstring = candidates[0][2]

    logger.info(
        "Best observed QAOA bitstring: %s",
        best_bitstring
    )

    logger.info(
        "Best QUBO energy: %.6f",
        best_energy
    )

    return best_bitstring, best_energy


# ============================================================
# BITSTRING → NODES
# ============================================================

def bitstring_to_selected_nodes(
    bitstring,
    graph
):
    nodes = list(graph.nodes)

    if len(bitstring) != len(nodes):
        raise ValueError(
            "Bitstring length does not match graph size"
        )

    selected = []

    for index, bit in enumerate(bitstring):
        if bit == "1":
            selected.append(nodes[index])

    return selected


# ============================================================
# FEASIBILITY
# ============================================================

def check_feasibility(
    graph,
    selected_nodes
):
    selected_nodes = list(selected_nodes)

    for i in range(len(selected_nodes)):
        for j in range(i + 1, len(selected_nodes)):

            u = selected_nodes[i]
            v = selected_nodes[j]

            if graph.has_edge(u, v):
                return False

    return True


# ============================================================
# EXACT GROUND TRUTH
# ============================================================

def calculate_exact_mwis(
    graph,
    weights
):
    """
    Exact MWIS using maximum-weight clique on the complement graph.

    NetworkX requires integer node weights for max_weight_clique().
    """

    complement = nx.complement(graph)

    integer_weights = {
        node: int(round(weights[node]))
        for node in complement.nodes
    }

    nx.set_node_attributes(
        complement,
        integer_weights,
        "weight"
    )

    clique, integer_weight = nx.max_weight_clique(
        complement,
        weight="weight"
    )

    selected_nodes = sorted(clique)

    total_weight = sum(
        weights[node]
        for node in selected_nodes
    )

    return (
        selected_nodes,
        float(total_weight)
    )


# ============================================================
# OPERATIONAL METRICS
# ============================================================

def calculate_operational_metrics(
    nodes,
    selected_nodes
):
    selected_set = set(selected_nodes)

    selected_data = [
        node
        for node in nodes
        if node["id"] in selected_set
    ]

    total_passenger_km = sum(
        float(node.get("passenger_km", 0))
        for node in selected_data
    )

    total_empty_km = sum(
        float(node.get("empty_km", 0))
        for node in selected_data
    )

    total_operating_cost = sum(
        float(node.get("operating_cost", 0))
        for node in selected_data
    )

    return {
        "total_passenger_km": total_passenger_km,
        "total_empty_km": total_empty_km,
        "total_operating_cost": total_operating_cost
    }


# ============================================================
# COVERAGE
# ============================================================

def calculate_coverage(
    nodes,
    selected_nodes
):
    scheduled_trips = set()

    covered_trips = set()

    selected_set = set(selected_nodes)

    for node in nodes:

        trips = node.get("trips", [])

        scheduled_trips.update(trips)

        if node["id"] in selected_set:
            covered_trips.update(trips)

    if not scheduled_trips:
        coverage_percent = 0.0
    else:
        coverage_percent = (
            len(covered_trips)
            / len(scheduled_trips)
            * 100.0
        )

    return (
        len(scheduled_trips),
        len(covered_trips),
        float(coverage_percent)
    )


# ============================================================
# BACKEND EXECUTION
# ============================================================

def execute_qaoa(
    circuit,
    shots,
    backend_name=None
):
    """
    Executes QAOA.

    If IQM Resonance is configured, it is used.

    Otherwise AerSimulator is used as a fallback.
    """

    # --------------------------------------------------------
    # IQM
    # --------------------------------------------------------

    iqm_token = (
        os.getenv("IQM_TOKEN")
        or os.getenv("IQM_API_TOKEN")
    )

    iqm_url = os.getenv(
        "IQM_URL",
        "https://resonance.cloud"
    )

    if (
        iqm_token
        and IQMProvider is not None
    ):

        try:
            logger.info(
                "Attempting IQM Resonance execution"
            )

            provider = IQMProvider(
                iqm_url,
                token=iqm_token
            )

            if backend_name:
                backend = provider.get_backend(
                    backend_name
                )
            else:
                backend = provider.get_backend(
                    "IQM Resonance (emerald)"
                )

            job = backend.run(
                circuit,
                shots=shots
            )

            result = job.result()

            counts = result.get_counts()

            return (
                counts,
                "IQM Resonance (emerald)",
                getattr(job, "job_id", lambda: None)()
            )

        except Exception as exc:

            logger.warning(
                "IQM execution failed: %s",
                exc
            )

            logger.warning(
                "Falling back to AerSimulator"
            )

    # --------------------------------------------------------
    # AER
    # --------------------------------------------------------

    if AerSimulator is None:
        raise RuntimeError(
            "Neither IQM nor Qiskit Aer is available"
        )

    simulator = AerSimulator()

    job = simulator.run(
        circuit,
        shots=shots
    )

    result = job.result()

    counts = result.get_counts()

    return (
        counts,
        "AerSimulator",
        None
    )


# ============================================================
# MAIN SOLVER
# ============================================================

def run(
    input_data,
    solver_params=None,
    extra_arguments=None
):

    solver_params = solver_params or {}
    extra_arguments = extra_arguments or {}

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    qaoa_depth = int(
        solver_params.get(
            "qaoa_depth",
            2
        )
    )

    shots = int(
        solver_params.get(
            "shots",
            2048
        )
    )

    backend_name = solver_params.get(
        "backend"
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    nodes = input_data.get("nodes", [])

    if not nodes:
        raise ValueError(
            "Input must contain at least one node"
        )

    graph = build_conflict_graph(
        input_data
    )

    logger.info(
        "Graph: %d nodes, %d conflicts",
        graph.number_of_nodes(),
        graph.number_of_edges()
    )

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    objective = MWISObjective(nodes)

    weights = objective.calculate_weights()

    logger.info(
        "Objective: maximize sum(weight_i*x_i)"
    )

    # --------------------------------------------------------
    # PENALTY
    # --------------------------------------------------------

    penalty = calculate_penalty(weights)

    logger.info(
        "Penalty: %.6f",
        penalty
    )

    # --------------------------------------------------------
    # EXACT GROUND TRUTH
    # --------------------------------------------------------

    exact_selected, exact_weight = (
        calculate_exact_mwis(
            graph,
            weights
        )
    )

    logger.info(
        "Exact MWIS: %s",
        exact_selected
    )

    logger.info(
        "Exact total weight: %.6f",
        exact_weight
    )

    # --------------------------------------------------------
    # QAOA
    # --------------------------------------------------------

    circuit = build_qaoa_circuit(
        graph,
        weights,
        penalty,
        depth=qaoa_depth
    )

    counts, backend_used, job_id = execute_qaoa(
        circuit,
        shots=shots,
        backend_name=backend_name
    )

    (
        initial_bitstring,
        initial_qubo_energy
    ) = select_best_qaoa_bitstring(
        counts,
        graph,
        weights,
        penalty
    )

    initial_selected = (
        bitstring_to_selected_nodes(
            initial_bitstring,
            graph
        )
    )

    # --------------------------------------------------------
    # REPAIR / PRUNING
    # --------------------------------------------------------

    selected_cycles = MWISPruner.prune_solution(
        graph,
        initial_selected,
        weights
    )

    selected_cycles = sorted(selected_cycles)

    # --------------------------------------------------------
    # FEASIBILITY
    # --------------------------------------------------------

    is_feasible = check_feasibility(
        graph,
        selected_cycles
    )

    # --------------------------------------------------------
    # SOLUTION WEIGHT
    # --------------------------------------------------------

    total_weight = sum(
        weights[node]
        for node in selected_cycles
    )

    # --------------------------------------------------------
    # OPTIMALITY GAP
    # --------------------------------------------------------

    if exact_weight != 0:
        optimality_gap_percent = (
            (exact_weight - total_weight)
            / abs(exact_weight)
            * 100.0
        )
    else:
        optimality_gap_percent = 0.0

    # Numerical protection
    if abs(optimality_gap_percent) < 1e-12:
        optimality_gap_percent = 0.0

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    (
        scheduled_trips,
        covered_trips,
        coverage_percent
    ) = calculate_coverage(
        nodes,
        selected_cycles
    )

    # --------------------------------------------------------
    # OPERATIONAL METRICS
    # --------------------------------------------------------

    operational = calculate_operational_metrics(
        nodes,
        selected_cycles
    )

    # --------------------------------------------------------
    # EXECUTION TIME
    # --------------------------------------------------------

    execution_time_seconds = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    asset_path = (
        VisualizationAssetGenerator
        .generate_conflict_graph(
            graph,
            selected_cycles
        )
    )

    # --------------------------------------------------------
    # PRUNING INFORMATION
    # --------------------------------------------------------

    nodes_pruned = max(
        len(initial_selected)
        - len(selected_cycles),
        0
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # The five QCentroid benchmark metrics are exposed
    # directly at the top level of the result.
    #

    result = {

        # ====================================================
        # BENCHMARK METRICS
        # ====================================================

        "total_weight": float(
            total_weight
        ),

        "optimality_gap_percent": float(
            optimality_gap_percent
        ),

        "coverage_percent": float(
            coverage_percent
        ),

        "execution_time_seconds": float(
            execution_time_seconds
        ),

        "total_operating_cost": float(
            operational["total_operating_cost"]
        ),

        # ====================================================
        # SOLUTION
        # ====================================================

        "selected_cycles": selected_cycles,

        "is_feasible": bool(
            is_feasible
        ),

        # ====================================================
        # OPERATIONAL RESULTS
        # ====================================================

        "total_passenger_km": float(
            operational["total_passenger_km"]
        ),

        "total_empty_km": float(
            operational["total_empty_km"]
        ),

        "scheduled_trips": int(
            scheduled_trips
        ),

        "covered_trips": int(
            covered_trips
        ),

        # ====================================================
        # EXECUTION INFORMATION
        # ====================================================

        "backend_used": backend_used,

        "execution_metrics": {

            "execution_time_seconds": float(
                execution_time_seconds
            ),

            "final_solution_size": int(
                len(selected_cycles)
            ),

            "initial_solution_size": int(
                len(initial_selected)
            ),

            "initial_bitstring_min_qubo": (
                initial_bitstring
            ),

            "initial_qubo_energy": float(
                initial_qubo_energy
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

            "job_id": job_id
        },

        # ====================================================
        # GROUND TRUTH
        # ====================================================

        "ground_truth_exact": {

            "selected_cycles": exact_selected,

            "total_weight": float(
                exact_weight
            ),

            "optimality_gap_percent": float(
                optimality_gap_percent
            )
        },

        # ====================================================
        # PRUNING
        # ====================================================

        "nodes_pruned": int(
            nodes_pruned
        ),

        "pruning_stats": {

            "nodes_pruned_count": int(
                nodes_pruned
            ),

            "pruned_nodes_list": []
        },

        # ====================================================
        # ASSETS
        # ====================================================

        "assets": {

            "conflict_graph_png": asset_path
        }
    }

    # --------------------------------------------------------
    # LOG BENCHMARK VALUES
    # --------------------------------------------------------

    logger.info(
        "Benchmark total_weight = %.4f",
        result["total_weight"]
    )

    logger.info(
        "Benchmark optimality_gap_percent = %.4f",
        result["optimality_gap_percent"]
    )

    logger.info(
        "Benchmark coverage_percent = %.4f",
        result["coverage_percent"]
    )

    logger.info(
        "Benchmark execution_time_seconds = %.4f",
        result["execution_time_seconds"]
    )

    logger.info(
        "Benchmark total_operating_cost = %.4f",
        result["total_operating_cost"]
    )

    logger.info(
        "Selected cycles: %s",
        selected_cycles
    )

    logger.info(
        "Execution finished"
    )

    return result


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    example_input = {

        "nodes": [

            {
                "id": "C01",
                "weight": 86,
                "trips": ["T01", "T02"],
                "passenger_km": 8200,
                "empty_km": 80,
                "operating_cost": 1180
            },

            {
                "id": "C02",
                "weight": 78,
                "trips": ["T02", "T03"],
                "passenger_km": 7600,
                "empty_km": 120,
                "operating_cost": 1210
            },

            {
                "id": "C03",
                "weight": 91,
                "trips": ["T03", "T04"],
                "passenger_km": 9000,
                "empty_km": 60,
                "operating_cost": 1160
            },

            {
                "id": "C04",
                "weight": 73,
                "trips": ["T04", "T05"],
                "passenger_km": 7100,
                "empty_km": 150,
                "operating_cost": 1240
            },

            {
                "id": "C05",
                "weight": 88,
                "trips": ["T05", "T01"],
                "passenger_km": 8500,
                "empty_km": 70,
                "operating_cost": 1190
            }
        ],

        "edges": [

            ["C01", "C02"],
            ["C02", "C03"],
            ["C03", "C04"],
            ["C04", "C05"],
            ["C05", "C01"]
        ]
    }

    result = run(
        example_input,
        solver_params={
            "qaoa_depth": 2,
            "shots": 2048,
            "backend": "IQM Resonance (emerald)"
        }
    )

    print("\nRESULT:")
    print(result)
