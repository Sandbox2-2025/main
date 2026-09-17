"""
QCentroid Solver
================

Railway Rolling Stock Cycle Selection
Maximum Weighted Independent Set (MWIS)
QUBO + QAOA
IQM Resonance

Objetivo
--------
Seleccionar ciclos de material rodante compatibles entre sí y maximizar
el valor total de los ciclos seleccionados.

Formulación:
    MWIS = Maximum Weighted Independent Set

QUBO:
    H(x) = - sum(w_i * x_i)
           + lambda * sum(x_i * x_j)

donde:
    x_i = 1 si se selecciona el ciclo i
    x_i = 0 en otro caso

    w_i = peso/valor del ciclo
    lambda = penalización por conflicto

El solver puede ejecutarse:

1. En IQM Resonance cuando se proporciona iqm_token.
2. En AerSimulator únicamente cuando se solicita explícitamente
   use_iqm=False.

La solución cuántica se valida contra una solución exacta clásica
para instancias pequeñas.

QCentroid invoca la función:

    run(input_data, solver_params, extra_arguments)

No es necesario ejecutar este archivo localmente.
"""

import os
import time
import logging

import matplotlib

# QCentroid se ejecuta normalmente en un entorno sin interfaz gráfica.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ---------------------------------------------------------------------
# CONFIGURACIÓN IQM
# ---------------------------------------------------------------------

DEFAULT_IQM_SERVER_URL = "https://resonance.iqm.tech/"

# Hosts utilizados por versiones antiguas de IQM/CoCoS.
OBSOLETE_IQM_HOSTS = (
    "cocos.resonance.meetiqm.com",
    "resonance.meetiqm.com",
)


def resolve_iqm_server_url():
    """
    Devuelve el endpoint IQM actualmente soportado.

    Para evitar que una configuración antigua del entorno de ejecución
    fuerce accidentalmente el endpoint obsoleto, cualquier URL de los
    hosts antiguos se sustituye por el endpoint actual de Resonance.
    """

    candidate = (
        os.getenv("IQM_SERVER_URL")
        or DEFAULT_IQM_SERVER_URL
    )

    candidate = str(candidate).strip()

    if any(host in candidate for host in OBSOLETE_IQM_HOSTS):
        logger.warning(
            "Se ha detectado un endpoint IQM obsoleto. "
            "Se utilizará el endpoint actual de IQM Resonance."
        )
        candidate = DEFAULT_IQM_SERVER_URL

    candidate = candidate.rstrip("/") + "/"

    return candidate


# ---------------------------------------------------------------------
# SANITIZACIÓN DE URL
# ---------------------------------------------------------------------

class URLSanitizer:
    """
    Utilidad para normalizar URLs sin mostrar credenciales.
    """

    @staticmethod
    def sanitize(url):
        if not url:
            return ""

        url = str(url).strip()

        # Nunca registrar tokens u otros parámetros de autenticación.
        for separator in ("?token=", "&token=", "?access_token=", "&access_token="):
            if separator in url:
                url = url.split(separator)[0]

        return url


# ---------------------------------------------------------------------
# OBJETIVO MWIS
# ---------------------------------------------------------------------

class MWISObjective:
    """
    Representa el objetivo de optimización.

    Si los nodos contienen 'weight', éste constituye explícitamente
    el objetivo.

    En ausencia de weight se utiliza una métrica de respaldo:

        2 * passenger_km - empty_km

    passenger_km, empty_km y operating_cost NO forman parte del QUBO
    cuando existe weight. Se mantienen como métricas operativas.
    """

    def __init__(self, nodes):
        self.nodes = nodes

        self.weights = {}

        for node in nodes:
            node_id = node["id"]

            if "weight" in node:
                weight = float(node["weight"])
            else:
                passenger_km = float(node.get("passenger_km", 0))
                empty_km = float(node.get("empty_km", 0))

                weight = (
                    2.0 * passenger_km
                    - empty_km
                )

            self.weights[node_id] = weight

    def total_weight(self, selected_nodes):
        return sum(
            self.weights[node]
            for node in selected_nodes
        )


# ---------------------------------------------------------------------
# PRUNING / REPARACIÓN
# ---------------------------------------------------------------------

class MWISPruner:
    """
    Reparación determinista de una solución que contiene conflictos.

    Se conserva preferentemente el nodo con mayor relación:

        weight / (degree + 1)
    """

    def __init__(self, graph, weights):
        self.graph = graph
        self.weights = weights

    def prune_solution(self, selected_nodes):

        selected = set(selected_nodes)

        while True:

            conflicts = []

            for u, v in self.graph.edges():

                if u in selected and v in selected:
                    conflicts.append((u, v))

            if not conflicts:
                break

            conflict_nodes = set()

            for u, v in conflicts:
                conflict_nodes.add(u)
                conflict_nodes.add(v)

            def score(node):

                degree = self.graph.degree(node)

                return (
                    self.weights[node]
                    / max(1, degree + 1)
                )

            node_to_remove = min(
                conflict_nodes,
                key=score
            )

            selected.remove(node_to_remove)

        return list(selected)


# ---------------------------------------------------------------------
# GRAFO DE CONFLICTOS
# ---------------------------------------------------------------------

def build_conflict_graph(input_data):

    nodes = input_data.get("nodes", [])
    edges = input_data.get("edges", [])

    graph = nx.Graph()

    for node in nodes:
        node_id = node["id"]
        graph.add_node(node_id)

    for edge in edges:

        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(
                f"Arista inválida: {edge}. "
                "Cada arista debe contener exactamente dos nodos."
            )

        u = edge[0]
        v = edge[1]

        if u not in graph:
            raise ValueError(
                f"El nodo '{u}' de la arista {edge} no existe."
            )

        if v not in graph:
            raise ValueError(
                f"El nodo '{v}' de la arista {edge} no existe."
            )

        if u == v:
            raise ValueError(
                f"No se permiten self-loops: {edge}"
            )

        graph.add_edge(u, v)

    return graph


# ---------------------------------------------------------------------
# QUBO
# ---------------------------------------------------------------------

def calculate_penalty(weights):
    """
    Penalización suficientemente grande para que una solución con
    conflictos resulte menos atractiva que una solución factible.

    Se utiliza:

        lambda = 4 * max(weight)
    """

    if not weights:
        return 1.0

    max_weight = max(
        abs(float(weight))
        for weight in weights.values()
    )

    return max(
        1.0,
        4.0 * max_weight
    )


def qubo_energy(graph, weights, selected_nodes, penalty):
    """
    Calcula directamente la energía QUBO:

        H(x) =
            -sum(w_i*x_i)
            +lambda*sum(x_i*x_j)
    """

    selected = set(selected_nodes)

    objective = sum(
        weights[node]
        for node in selected
    )

    conflicts = 0

    for u, v in graph.edges():

        if u in selected and v in selected:
            conflicts += 1

    return (
        -objective
        + penalty * conflicts
    )


# ---------------------------------------------------------------------
# CIRCUITO QAOA
# ---------------------------------------------------------------------

def build_qaoa_circuit(
    graph,
    weights,
    penalty,
    qaoa_depth=2,
    gamma=0.8,
    beta=0.6,
):
    """
    Construye un circuito QAOA para el QUBO de MWIS.

    Se utiliza la transformación:

        x = (1 - Z) / 2

    El término constante del Hamiltoniano no necesita implementarse
    porque no afecta a la selección del mínimo.

    Para:

        H = sum(a_i Z_i) + sum(b_ij Z_i Z_j)

    se aplican:

        RZ(2 * gamma * a_i)
        RZZ(2 * gamma * b_ij)

    El mixer estándar utiliza RX.
    """

    from qiskit import QuantumCircuit

    nodes = list(graph.nodes())
    n = len(nodes)

    node_to_qubit = {
        node: index
        for index, node in enumerate(nodes)
    }

    circuit = QuantumCircuit(n, n)

    # Superposición inicial.
    for qubit in range(n):
        circuit.h(qubit)

    # Precalcular grados.
    degrees = dict(graph.degree())

    # Coeficientes Z del Hamiltoniano QUBO.
    #
    # -w*x = -w*(1-Z)/2
    #       = -w/2 + w/2 Z
    #
    # lambda*x_i*x_j
    # = lambda/4 * (1-Z_i-Z_j+Z_iZ_j)
    #
    # Por tanto:
    #
    # a_i = w_i/2 - lambda/4 * degree(i)
    # b_ij = lambda/4

    z_coefficients = {}

    for node in nodes:

        w = float(weights[node])

        z_coefficients[node] = (
            w / 2.0
            - penalty * degrees[node] / 4.0
        )

    zz_coefficient = penalty / 4.0

    for layer in range(qaoa_depth):

        # -------------------------------------------------------------
        # COST UNITARY
        # -------------------------------------------------------------

        for node in nodes:

            qubit = node_to_qubit[node]

            angle = (
                2.0
                * gamma
                * z_coefficients[node]
            )

            circuit.rz(
                angle,
                qubit
            )

        for u, v in graph.edges():

            q_u = node_to_qubit[u]
            q_v = node_to_qubit[v]

            angle = (
                2.0
                * gamma
                * zz_coefficient
            )

            circuit.rzz(
                angle,
                q_u,
                q_v
            )

        # -------------------------------------------------------------
        # MIXER
        # -------------------------------------------------------------

        for qubit in range(n):

            circuit.rx(
                2.0 * beta,
                qubit
            )

    # Medición.
    circuit.measure(
        range(n),
        range(n)
    )

    return circuit


# ---------------------------------------------------------------------
# SELECCIÓN DEL MEJOR BITSTRING
# ---------------------------------------------------------------------

def bitstring_to_selected_nodes(bitstring, nodes):
    """
    Convierte un bitstring de Qiskit en nodos seleccionados.

    Qiskit devuelve los bits de los registros clásicos en orden
    inverso respecto al orden natural de los qubits.

    Ejemplo:

        nodes = [C01,C02,C03,C04,C05]

        bitstring = 10100

        reversed(bitstring) = 00101

        selección:
            C03
            C05
    """

    clean = bitstring.replace(" ", "")

    # Qiskit presenta el bit del qubit 0 a la derecha.
    clean = clean[::-1]

    selected = []

    for index, bit in enumerate(clean):

        if index >= len(nodes):
            break

        if bit == "1":
            selected.append(nodes[index])

    return selected


def select_best_qaoa_bitstring(
    counts,
    graph,
    weights,
    penalty,
    nodes,
):
    """
    Evalúa todos los bitstrings observados y selecciona el que tenga
    menor energía QUBO.

    No se selecciona simplemente el bitstring más frecuente.

    Criterio:
        1. menor energía QUBO
        2. mayor número de shots en caso de empate
    """

    candidates = []

    for bitstring, shots in counts.items():

        selected = bitstring_to_selected_nodes(
            bitstring,
            nodes
        )

        energy = qubo_energy(
            graph,
            weights,
            selected,
            penalty
        )

        candidates.append(
            (
                float(energy),
                -int(shots),
                bitstring,
                selected
            )
        )

    if not candidates:
        raise RuntimeError(
            "El backend QAOA no ha devuelto ningún bitstring."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    best_energy, _, best_bitstring, selected = candidates[0]

    return (
        best_bitstring,
        selected,
        float(best_energy)
    )


# ---------------------------------------------------------------------
# EJECUCIÓN QAOA
# ---------------------------------------------------------------------

def execute_qaoa(
    circuit,
    extra_arguments,
):
    """
    Ejecuta QAOA.

    IQM es el backend por defecto cuando existe iqm_token.

    Para utilizar Aer explícitamente:

        use_iqm = false

    en los parámetros adicionales de QCentroid.

    IMPORTANTE:
    El token procede de QCentroid y nunca se imprime en logs.
    """

    from qiskit import transpile

    use_iqm = extra_arguments.get(
        "use_iqm",
        True
    )

    shots = int(
        extra_arguments.get(
            "shots",
            2048
        )
    )

    quantum_computer = str(
        extra_arguments.get(
            "quantum_computer",
            "emerald"
        )
    )

    # -------------------------------------------------------------
    # IQM
    # -------------------------------------------------------------

    if use_iqm:

        iqm_token = extra_arguments.get(
            "iqm_token"
        )

        if not iqm_token:

            raise RuntimeError(
                "Se ha solicitado ejecución IQM, pero "
                "QCentroid no ha proporcionado 'iqm_token'. "
                "No se utilizará Aer automáticamente."
            )

        try:

            from iqm.qiskit_iqm import IQMProvider

            iqm_server_url = resolve_iqm_server_url()

            logger.info(
                "Conectando con IQM Resonance: %s",
                URLSanitizer.sanitize(iqm_server_url)
            )

            logger.info(
                "Quantum computer solicitado: %s",
                quantum_computer
            )

            provider = IQMProvider(
                iqm_server_url,
                quantum_computer=quantum_computer,
                token=iqm_token,
            )

            backend = provider.get_backend()

            logger.info(
                "Backend IQM seleccionado: %s",
                backend.name
            )

            transpiled_circuit = transpile(
                circuit,
                backend=backend
            )

            job = backend.run(
                transpiled_circuit,
                shots=shots
            )

            job_id = None

            try:
                job_id = job.job_id()
            except Exception:
                pass

            logger.info(
                "Job IQM enviado. Job ID: %s",
                job_id
            )

            result = job.result()

            counts = result.get_counts()

            return (
                counts,
                "IQM Resonance",
                job_id,
                backend.name,
            )

        except Exception as exc:

            raise RuntimeError(
                "No se pudo ejecutar el circuito en IQM Resonance. "
                f"Quantum computer solicitado: {quantum_computer}. "
                f"Detalle: {type(exc).__name__}: {exc}"
            ) from exc

    # -------------------------------------------------------------
    # AER — SOLO SI SE SOLICITA EXPRESAMENTE
    # -------------------------------------------------------------

    logger.info(
        "IQM desactivado explícitamente. "
        "Utilizando Qiskit AerSimulator."
    )

    try:

        from qiskit_aer import AerSimulator

        backend = AerSimulator()

        transpiled_circuit = transpile(
            circuit,
            backend=backend
        )

        job = backend.run(
            transpiled_circuit,
            shots=shots
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
            "Qiskit AerSimulator",
            job_id,
            backend.name,
        )

    except Exception as exc:

        raise RuntimeError(
            "No se pudo ejecutar QAOA con AerSimulator. "
            f"Detalle: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------
# SOLUCIÓN EXACTA CLÁSICA
# ---------------------------------------------------------------------

def exact_mwis(graph, weights):
    """
    Obtiene la solución MWIS exacta mediante:

        MWIS(G) = Maximum Weight Clique(complement(G))

    Esta función está destinada a instancias pequeñas/medianas y
    sirve como ground truth para validar QAOA.
    """

    complement = nx.complement(graph)

    # NetworkX 3.6.x requiere pesos enteros para max_weight_clique.
    for node in complement.nodes():

        complement.nodes[node]["weight"] = int(
            round(
                float(weights[node])
            )
        )

    clique, integer_weight = nx.algorithms.clique.max_weight_clique(
        complement,
        weight="weight"
    )

    selected = list(clique)

    total_weight = sum(
        float(weights[node])
        for node in selected
    )

    return (
        selected,
        total_weight
    )


# ---------------------------------------------------------------------
# MÉTRICAS OPERATIVAS
# ---------------------------------------------------------------------

def calculate_operational_metrics(
    nodes,
    selected_nodes,
):

    selected_set = set(selected_nodes)

    selected_data = [
        node
        for node in nodes
        if node["id"] in selected_set
    ]

    passenger_km = sum(
        float(node.get("passenger_km", 0))
        for node in selected_data
    )

    empty_km = sum(
        float(node.get("empty_km", 0))
        for node in selected_data
    )

    operating_cost = sum(
        float(node.get("operating_cost", 0))
        for node in selected_data
    )

    trips = set()

    for node in selected_data:

        for trip in node.get("trips", []):
            trips.add(trip)

    return {
        "total_passenger_km": passenger_km,
        "total_empty_km": empty_km,
        "total_operating_cost": operating_cost,
        "covered_trips": len(trips),
        "covered_trip_ids": sorted(trips),
    }


# ---------------------------------------------------------------------
# VISUALIZACIÓN
# ---------------------------------------------------------------------

class VisualizationAssetGenerator:

    @staticmethod
    def generate_conflict_graph(
        graph,
        selected_nodes,
    ):

        output_dir = "additional_output"

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        path = os.path.join(
            output_dir,
            "conflict_graph.png"
        )

        plt.figure(
            figsize=(9, 7),
            dpi=150
        )

        selected_set = set(
            selected_nodes
        )

        positions = nx.spring_layout(
            graph,
            seed=42
        )

        node_sizes = []

        for node in graph.nodes():

            if node in selected_set:
                node_sizes.append(900)
            else:
                node_sizes.append(650)

        nx.draw_networkx_nodes(
            graph,
            positions,
            node_size=node_sizes,
            node_color=[
                "tab:orange"
                if node in selected_set
                else "lightgray"
                for node in graph.nodes()
            ],
            edgecolors="black"
        )

        nx.draw_networkx_edges(
            graph,
            positions,
            width=2
        )

        nx.draw_networkx_labels(
            graph,
            positions,
            font_size=10,
            font_weight="bold"
        )

        plt.title(
            "Railway Rolling Stock — Conflict Graph"
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

        return path


# ---------------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE QCENTROID
# ---------------------------------------------------------------------

def run(
    input_data,
    solver_params,
    extra_arguments,
):
    """
    Entry point utilizado por QCentroid.
    """

    start_time = time.perf_counter()

    logger.info(
        "=================================================="
    )

    logger.info(
        "Inicio solver Railway Rolling Stock MWIS + QAOA"
    )

    logger.info(
        "=================================================="
    )

    # -------------------------------------------------------------
    # VALIDACIÓN DE INPUT
    # -------------------------------------------------------------

    if not isinstance(input_data, dict):

        raise ValueError(
            "input_data debe ser un objeto JSON."
        )

    nodes = input_data.get(
        "nodes",
        []
    )

    edges = input_data.get(
        "edges",
        []
    )

    if not nodes:

        raise ValueError(
            "El dataset no contiene nodos."
        )

    # -------------------------------------------------------------
    # GRAFO
    # -------------------------------------------------------------

    graph = build_conflict_graph(
        input_data
    )

    logger.info(
        "Grafo construido: %d nodos, %d conflictos.",
        graph.number_of_nodes(),
        graph.number_of_edges()
    )

    # -------------------------------------------------------------
    # OBJETIVO
    # -------------------------------------------------------------

    objective = MWISObjective(
        nodes
    )

    weights = objective.weights

    logger.info(
        "Objetivo: maximizar sum(weight_i * x_i)."
    )

    logger.info(
        "Pesos: %s",
        weights
    )

    # -------------------------------------------------------------
    # PENALIZACIÓN
    # -------------------------------------------------------------

    penalty = calculate_penalty(
        weights
    )

    logger.info(
        "Penalización QUBO: %.4f",
        penalty
    )

    # -------------------------------------------------------------
    # PARÁMETROS QAOA
    # -------------------------------------------------------------

    qaoa_depth = int(
        extra_arguments.get(
            "qaoa_depth",
            2
        )
    )

    shots = int(
        extra_arguments.get(
            "shots",
            2048
        )
    )

    gamma = float(
        extra_arguments.get(
            "gamma",
            0.8
        )
    )

    beta = float(
        extra_arguments.get(
            "beta",
            0.6
        )
    )

    logger.info(
        "QAOA: depth=%d, shots=%d, gamma=%.4f, beta=%.4f",
        qaoa_depth,
        shots,
        gamma,
        beta
    )

    # -------------------------------------------------------------
    # CIRCUITO
    # -------------------------------------------------------------

    circuit = build_qaoa_circuit(
        graph=graph,
        weights=weights,
        penalty=penalty,
        qaoa_depth=qaoa_depth,
        gamma=gamma,
        beta=beta,
    )

    logger.info(
        "Circuito QAOA construido: %d qubits.",
        graph.number_of_nodes()
    )

    # -------------------------------------------------------------
    # EJECUCIÓN CUÁNTICA
    # -------------------------------------------------------------

    counts, backend_used, job_id, backend_name = execute_qaoa(
        circuit,
        extra_arguments,
    )

    logger.info(
        "Backend utilizado: %s",
        backend_used
    )

    logger.info(
        "Resultados recibidos: %d bitstrings únicos.",
        len(counts)
    )

    # -------------------------------------------------------------
    # SELECCIÓN DEL MEJOR RESULTADO
    # -------------------------------------------------------------

    graph_nodes = list(
        graph.nodes()
    )

    (
        best_bitstring,
        selected_nodes,
        initial_qubo_energy,
    ) = select_best_qaoa_bitstring(
        counts=counts,
        graph=graph,
        weights=weights,
        penalty=penalty,
        nodes=graph_nodes,
    )

    logger.info(
        "Mejor bitstring: %s",
        best_bitstring
    )

    logger.info(
        "Energía QUBO inicial: %.4f",
        initial_qubo_energy
    )

    logger.info(
        "Solución QAOA inicial: %s",
        selected_nodes
    )

    # -------------------------------------------------------------
    # REPARACIÓN / PRUNING
    # -------------------------------------------------------------

    pruner = MWISPruner(
        graph,
        weights
    )

    repaired_nodes = pruner.prune_solution(
        selected_nodes
    )

    repaired_set = set(
        repaired_nodes
    )

    original_set = set(
        selected_nodes
    )

    pruned_nodes = sorted(
        original_set - repaired_set
    )

    if pruned_nodes:

        logger.info(
            "Nodos eliminados durante pruning: %s",
            pruned_nodes
        )

    else:

        logger.info(
            "No ha sido necesario realizar pruning."
        )

    selected_nodes = sorted(
        repaired_nodes
    )

    # -------------------------------------------------------------
    # VALIDACIÓN DE FACTIBILIDAD
    # -------------------------------------------------------------

    is_feasible = True

    for u, v in graph.edges():

        if (
            u in selected_nodes
            and v in selected_nodes
        ):

            is_feasible = False

            break

    if not is_feasible:

        raise RuntimeError(
            "La solución final contiene conflictos."
        )

    # -------------------------------------------------------------
    # PESO FINAL
    # -------------------------------------------------------------

    total_weight = objective.total_weight(
        selected_nodes
    )

    final_qubo_energy = qubo_energy(
        graph,
        weights,
        selected_nodes,
        penalty
    )

    # -------------------------------------------------------------
    # GROUND TRUTH EXACTO
    # -------------------------------------------------------------

    exact_selected_nodes, exact_total_weight = exact_mwis(
        graph,
        weights
    )

    exact_selected_nodes = sorted(
        exact_selected_nodes
    )

    if exact_total_weight > 0:

        optimality_gap_percent = (
            (
                exact_total_weight
                - total_weight
            )
            / exact_total_weight
            * 100.0
        )

    else:

        optimality_gap_percent = 0.0

    # Evitar errores numéricos mínimos.
    if abs(optimality_gap_percent) < 1e-12:
        optimality_gap_percent = 0.0

    logger.info(
        "Ground truth exacto: %s",
        exact_selected_nodes
    )

    logger.info(
        "Peso óptimo exacto: %.4f",
        exact_total_weight
    )

    logger.info(
        "Peso solución QAOA: %.4f",
        total_weight
    )

    logger.info(
        "Optimality gap: %.4f%%",
        optimality_gap_percent
    )

    # -------------------------------------------------------------
    # MÉTRICAS OPERATIVAS
    # -------------------------------------------------------------

    operational = calculate_operational_metrics(
        nodes,
        selected_nodes
    )

    all_trips = set()

    for node in nodes:

        for trip in node.get(
            "trips",
            []
        ):
            all_trips.add(trip)

    scheduled_trips = len(
        all_trips
    )

    covered_trips = operational[
        "covered_trips"
    ]

    if scheduled_trips > 0:

        coverage_rate = (
            covered_trips
            / scheduled_trips
        )

        coverage_percent = (
            coverage_rate
            * 100.0
        )

    else:

        coverage_rate = 0.0
        coverage_percent = 0.0

    # -------------------------------------------------------------
    # VISUALIZACIÓN
    # -------------------------------------------------------------

    asset_path = (
        VisualizationAssetGenerator
        .generate_conflict_graph(
            graph,
            selected_nodes
        )
    )

    # -------------------------------------------------------------
    # TIEMPO DE EJECUCIÓN
    # -------------------------------------------------------------

    execution_time_seconds = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "Tiempo de ejecución del solver: %.4f s",
        execution_time_seconds
    )

    # -------------------------------------------------------------
    # PRUNING STATS
    # -------------------------------------------------------------

    pruning_stats = {
        "nodes_pruned_count": len(
            pruned_nodes
        ),
        "pruned_nodes_list": pruned_nodes,
    }

    # -------------------------------------------------------------
    # RESULTADO QENTROID
    # -------------------------------------------------------------
    #
    # Los cinco parámetros principales están en el nivel superior
    # porque son los campos configurados como Output Benchmark
    # Metrics en QCentroid.
    #

    result = {

        # =========================================================
        # BENCHMARK OUTPUT PARAMETERS
        # =========================================================

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
            operational[
                "total_operating_cost"
            ]
        ),

        # =========================================================
        # SOLUCIÓN
        # =========================================================

        "selected_cycles": selected_nodes,

        "is_feasible": bool(
            is_feasible
        ),

        "backend_used": backend_used,

        # =========================================================
        # OPERACIÓN
        # =========================================================

        "scheduled_trips": scheduled_trips,

        "covered_trips": covered_trips,

        "coverage_rate": float(
            coverage_rate
        ),

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

        # =========================================================
        # GROUND TRUTH
        # =========================================================

        "ground_truth_exact": {

            "selected_cycles":
                exact_selected_nodes,

            "total_weight":
                float(
                    exact_total_weight
                ),

            "optimality_gap_percent":
                float(
                    optimality_gap_percent
                ),
        },

        # =========================================================
        # EJECUCIÓN
        # =========================================================

        "execution_metrics": {

            "execution_time_seconds":
                float(
                    execution_time_seconds
                ),

            "final_solution_size":
                len(
                    selected_nodes
                ),

            "initial_bitstring_min_qubo":
                best_bitstring,

            "initial_qubo_energy":
                float(
                    initial_qubo_energy
                ),

            "final_qubo_energy":
                float(
                    final_qubo_energy
                ),

            "initial_solution_size":
                len(
                    selected_nodes
                ),

            "job_id":
                job_id,

            "backend_name":
                backend_name,

            "penalty":
                float(
                    penalty
                ),

            "qaoa_depth":
                qaoa_depth,

            "shots":
                shots,
        },

        # =========================================================
        # PRUNING
        # =========================================================

        "nodes_pruned": len(
            pruned_nodes
        ),

        "pruning_stats":
            pruning_stats,

        # =========================================================
        # ASSETS
        # =========================================================

        "assets": {

            "conflict_graph_png":
                asset_path,
        },
    }

    logger.info(
        "=================================================="
    )

    logger.info(
        "Solver finalizado correctamente."
    )

    logger.info(
        "Selección final: %s",
        selected_nodes
    )

    logger.info(
        "Peso total: %.4f",
        total_weight
    )

    logger.info(
        "Cobertura: %.2f%%",
        coverage_percent
    )

    logger.info(
        "Optimality gap: %.2f%%",
        optimality_gap_percent
    )

    logger.info(
        "=================================================="
    )

    return result
