"""
qcentroid.py

Benchmark ferroviario:
    Problema ferroviario
        ↓
    Grafo de conflictos
        ↓
    MWIS (Maximum Weight Independent Set)
        ↓
    QUBO
        ↓
    QAOA
        ↓
    Hardware cuántico / AerSimulator
        ↓
    Solución candidata
        ↓
    Validación clásica
        ↓
    Métricas operativas

IMPORTANTE:
- QCentroid suministra el dataset mediante `input_data`.
- El dataset NO está hardcodeado en este fichero.
- La función de entrada para QCentroid es:

      run(input_data, solver_params, extra_arguments)
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


# ============================================================================
# OPTIONAL QUANTUM DEPENDENCIES
# ============================================================================

try:
    from qiskit import QuantumCircuit, transpile

    QISKIT_AVAILABLE = True
except Exception:
    QuantumCircuit = None
    transpile = None
    QISKIT_AVAILABLE = False


try:
    from qiskit_aer import AerSimulator

    AER_AVAILABLE = True
except Exception:
    AerSimulator = None
    AER_AVAILABLE = False


try:
    from iqm.qiskit_iqm import IQMProvider

    IQM_AVAILABLE = True
except Exception:
    IQMProvider = None
    IQM_AVAILABLE = False


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("qcentroid-user-log")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# ============================================================================
# URL SANITIZER
# ============================================================================

class URLSanitizer:
    """
    Pequeño helper para evitar errores cuando se reciben URLs vacías
    o parámetros de configuración incompletos.
    """

    @staticmethod
    def normalize(url: Optional[str]) -> Optional[str]:
        if url is None:
            return None

        url = str(url).strip()

        if not url:
            return None

        return url.rstrip("/") + "/"


# ============================================================================
# MWIS OBJECTIVE
# ============================================================================

class MWISObjective:
    """
    Representa el problema Maximum Weight Independent Set.

    Para cada nodo i:

        x_i = 1  -> nodo seleccionado
        x_i = 0  -> nodo no seleccionado

    Función objetivo QUBO:

        E(x) = - Σ w_i x_i
               + λ Σ x_i x_j

    donde (i,j) son aristas del grafo de conflicto.

    Minimizar E equivale a maximizar el peso total evitando conflictos.
    """

    def __init__(
        self,
        graph: nx.Graph,
        penalty: Optional[float] = None,
    ):
        self.graph = graph

        # Orden determinista de los nodos.
        self.nodes = sorted(
            list(graph.nodes()),
            key=lambda x: str(x),
        )

        self.node_to_index = {
            node: index
            for index, node in enumerate(self.nodes)
        }

        self.weights = {}

        for node in self.nodes:
            weight = graph.nodes[node].get("weight", 1.0)

            try:
                weight = float(weight)
            except Exception:
                weight = 1.0

            self.weights[node] = weight

        max_weight = max(
            self.weights.values(),
            default=1.0,
        )

        # Penalización suficientemente grande para evitar conflictos.
        self.penalty = (
            float(penalty)
            if penalty is not None
            else 4.0 * max_weight
        )

    # ---------------------------------------------------------------------

    def bitstring_to_bits(self, bitstring: str) -> List[int]:
        """
        Qiskit utiliza normalmente orden little-endian.

        El bitstring observado se invierte para volver al orden:

            q0, q1, q2, ...

        correspondiente a self.nodes.
        """

        clean = str(bitstring).replace(" ", "")

        bits = [
            int(bit)
            for bit in clean
        ]

        return list(reversed(bits))

    # ---------------------------------------------------------------------

    def qubo_energy(self, bits: List[int]) -> float:
        """
        Calcula la energía QUBO real de una solución.
        """

        if len(bits) != len(self.nodes):
            raise ValueError(
                "Número de bits incompatible con el número de nodos."
            )

        energy = 0.0

        # Término lineal.
        for index, node in enumerate(self.nodes):
            x = bits[index]

            energy -= self.weights[node] * x

        # Penalización de conflictos.
        for u, v in self.graph.edges():

            i = self.node_to_index[u]
            j = self.node_to_index[v]

            energy += (
                self.penalty
                * bits[i]
                * bits[j]
            )

        return float(energy)

    # ---------------------------------------------------------------------

    def solution_weight(self, bits: List[int]) -> float:
        """
        Peso total de los nodos seleccionados.
        """

        total = 0.0

        for index, node in enumerate(self.nodes):
            if bits[index]:
                total += self.weights[node]

        return float(total)

    # ---------------------------------------------------------------------

    def is_feasible(self, bits: List[int]) -> bool:
        """
        Comprueba si la solución es un conjunto independiente.
        """

        if len(bits) != len(self.nodes):
            return False

        for u, v in self.graph.edges():

            i = self.node_to_index[u]
            j = self.node_to_index[v]

            if bits[i] == 1 and bits[j] == 1:
                return False

        return True

    # ---------------------------------------------------------------------

    def exact_solution(self) -> Tuple[List[str], float]:
        """
        Calcula el MWIS exacto mediante transformación a Maximum Weight
        Clique sobre el grafo complemento.

        Adecuado para obtener ground truth en datasets pequeños.

        IMPORTANTE:
        Esta función es clásica y puede crecer exponencialmente.
        """

        if self.graph.number_of_nodes() == 0:
            return [], 0.0

        complement = nx.complement(self.graph)

        # Usamos pesos enteros escalados para evitar pérdida de precisión.
        scale = 1000

        for node in complement.nodes():
            weight = self.weights.get(node, 0.0)

            complement.nodes[node]["weight"] = int(
                round(weight * scale)
            )

        clique = nx.algorithms.clique.max_weight_clique(
            complement,
            weight="weight",
        )

        clique_nodes = clique[0]

        selected = sorted(
            clique_nodes,
            key=lambda x: str(x),
        )

        total_weight = sum(
            self.weights[node]
            for node in selected
        )

        return selected, float(total_weight)


# ============================================================================
# CLASSICAL REPAIR / PRUNING
# ============================================================================

class MWISPruner:
    """
    Reparación clásica de una solución que contiene conflictos.

    Si la solución cuántica contiene dos nodos conectados,
    se elimina el nodo con menor relación:

        weight / degree

    Esto convierte la salida cuántica en una solución MWIS factible.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    # ---------------------------------------------------------------------

    def repair(
        self,
        selected_nodes: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Devuelve:

            solución reparada
            nodos eliminados
        """

        selected = set(selected_nodes)
        pruned = []

        while True:

            conflicts = []

            for u, v in self.graph.edges():

                if u in selected and v in selected:
                    conflicts.append((u, v))

            if not conflicts:
                break

            # Seleccionar el primer conflicto de forma determinista.
            u, v = sorted(
                conflicts,
                key=lambda edge: (
                    str(edge[0]),
                    str(edge[1]),
                ),
            )[0]

            degree_u = max(
                self.graph.degree(u),
                1,
            )

            degree_v = max(
                self.graph.degree(v),
                1,
            )

            weight_u = float(
                self.graph.nodes[u].get(
                    "weight",
                    1.0,
                )
            )

            weight_v = float(
                self.graph.nodes[v].get(
                    "weight",
                    1.0,
                )
            )

            ratio_u = weight_u / degree_u
            ratio_v = weight_v / degree_v

            # Eliminamos el de menor eficiencia.
            if ratio_u < ratio_v:
                remove_node = u

            elif ratio_v < ratio_u:
                remove_node = v

            else:
                # Desempate determinista.
                remove_node = max(
                    [u, v],
                    key=lambda x: str(x),
                )

            selected.remove(remove_node)
            pruned.append(remove_node)

        repaired = sorted(
            selected,
            key=lambda x: str(x),
        )

        return repaired, pruned


# ============================================================================
# VISUALIZATION
# ============================================================================

class VisualizationAssetGenerator:
    """
    Genera activos visuales opcionales para QCentroid.
    """

    @staticmethod
    def generate_conflict_graph(
        graph: nx.Graph,
        output_dir: str = "additional_output",
    ) -> Optional[str]:

        try:
            import matplotlib

            matplotlib.use("Agg")

            import matplotlib.pyplot as plt

        except Exception as exc:

            logger.warning(
                "No se pudo importar matplotlib: %s",
                exc,
            )

            return None

        try:

            output_path = Path(output_dir)

            output_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            file_path = (
                output_path
                / "conflict_graph.png"
            )

            plt.figure(
                figsize=(8, 6)
            )

            if graph.number_of_nodes() > 0:

                pos = nx.spring_layout(
                    graph,
                    seed=42,
                )

                weights = [
                    graph.nodes[node].get(
                        "weight",
                        1.0,
                    )
                    for node in graph.nodes()
                ]

                nx.draw_networkx(
                    graph,
                    pos=pos,
                    with_labels=True,
                    node_size=1800,
                    node_color=weights,
                    cmap="viridis",
                )

            plt.title(
                "Railway Conflict Graph"
            )

            plt.axis("off")

            plt.tight_layout()

            plt.savefig(
                file_path,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close()

            return str(file_path)

        except Exception as exc:

            logger.warning(
                "No se pudo generar el grafo visual: %s",
                exc,
            )

            return None


# ============================================================================
# DATASET → CONFLICT GRAPH
# ============================================================================

def build_conflict_graph(
    input_data: Dict[str, Any],
) -> nx.Graph:
    """
    Construye el grafo de conflictos a partir de input_data.

    Formato esperado:

    {
        "nodes": [
            {
                "id": "C01",
                "weight": 86,
                ...
            }
        ],
        "edges": [
            ["C01", "C02"],
            ["C02", "C03"]
        ]
    }

    IMPORTANTE:
    Si el dataset contiene edges explícitos, estos tienen prioridad.

    Si no contiene edges, se puede inferir conflicto cuando dos nodos
    comparten viajes.
    """

    graph = nx.Graph()

    if not isinstance(input_data, dict):
        raise ValueError(
            "input_data debe ser un diccionario."
        )

    nodes_raw = input_data.get(
        "nodes",
        [],
    )

    edges_raw = input_data.get(
        "edges",
        [],
    )

    if not isinstance(nodes_raw, list):
        raise ValueError(
            "'nodes' debe ser una lista."
        )

    # ------------------------------------------------------------------
    # NODES
    # ------------------------------------------------------------------

    for node_data in nodes_raw:

        if not isinstance(node_data, dict):
            logger.warning(
                "Nodo ignorado por formato inválido: %r",
                node_data,
            )
            continue

        node_id = node_data.get("id")

        if node_id is None:
            logger.warning(
                "Nodo ignorado porque no tiene 'id': %r",
                node_data,
            )
            continue

        node_id = str(node_id)

        weight = node_data.get(
            "weight",
            1.0,
        )

        try:
            weight = float(weight)
        except Exception:
            logger.warning(
                "Peso inválido para %s. Se utiliza 1.0.",
                node_id,
            )
            weight = 1.0

        # Conservamos todos los atributos originales.
        attributes = dict(node_data)

        attributes["weight"] = weight

        graph.add_node(
            node_id,
            **attributes,
        )

    # ------------------------------------------------------------------
    # EXPLICIT EDGES
    # ------------------------------------------------------------------

    if edges_raw:

        if not isinstance(edges_raw, list):
            raise ValueError(
                "'edges' debe ser una lista."
            )

        for edge in edges_raw:

            # ----------------------------------------------------------
            # CORRECCIÓN PRINCIPAL
            #
            # Las aristas del dataset son:
            #
            # ["C01", "C02"]
            #
            # Por tanto:
            #
            # u = edge[0]
            # v = edge[1]
            #
            # NO edge[3].
            # ----------------------------------------------------------

            if (
                not isinstance(
                    edge,
                    (list, tuple),
                )
                or len(edge) != 2
            ):

                logger.warning(
                    "Arista ignorada (formato inválido): %r",
                    edge,
                )

                continue

            u = str(edge[0])
            v = str(edge[1])

            if (
                u not in graph
                or v not in graph
            ):

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

            graph.add_edge(
                u,
                v,
            )

    # ------------------------------------------------------------------
    # INFER CONFLICTS FROM SHARED TRIPS
    # ------------------------------------------------------------------

    else:

        trip_to_nodes = {}

        for node in graph.nodes():

            trips = graph.nodes[node].get(
                "trips",
                [],
            )

            if trips is None:
                trips = []

            if not isinstance(
                trips,
                (list, tuple),
            ):
                trips = [trips]

            for trip in trips:

                trip = str(trip)

                trip_to_nodes.setdefault(
                    trip,
                    [],
                ).append(node)

        for nodes in trip_to_nodes.values():

            for i in range(len(nodes)):

                for j in range(i + 1, len(nodes)):

                    u = nodes[i]
                    v = nodes[j]

                    if u != v:
                        graph.add_edge(
                            u,
                            v,
                        )

    logger.info(
        "Grafo construido: %d nodos, %d aristas.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    return graph


# ============================================================================
# QAOA CIRCUIT
# ============================================================================

def build_qaoa_circuit(
    objective: MWISObjective,
    depth: int = 2,
) -> "QuantumCircuit":
    """
    Construye un circuito QAOA-style.

    IMPORTANTE:
    Los parámetros gamma/beta utilizados aquí son deterministas.

    Por tanto, esta implementación constituye una ejecución QAOA con
    parámetros prefijados, NO un QAOA variacional completo con un
    optimizador clásico que busque los parámetros óptimos.
    """

    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Qiskit no está disponible."
        )

    n = len(objective.nodes)

    if n == 0:
        raise ValueError(
            "El problema no contiene nodos."
        )

    depth = max(
        int(depth),
        1,
    )

    circuit = QuantumCircuit(
        n,
        n,
    )

    # ------------------------------------------------------------------
    # INITIAL STATE
    # ------------------------------------------------------------------

    for qubit in range(n):
        circuit.h(qubit)

    # ------------------------------------------------------------------
    # QAOA LAYERS
    # ------------------------------------------------------------------

    for layer in range(depth):

        gamma = 0.05 / (
            layer + 1
        )

        beta = 0.25 / (
            layer + 1
        )

        # --------------------------------------------------------------
        # COST HAMILTONIAN - LINEAR TERMS
        # --------------------------------------------------------------

        for index, node in enumerate(
            objective.nodes
        ):

            weight = objective.weights[node]

            # Representación del término lineal.
            angle = (
                2.0
                * gamma
                * weight
            )

            circuit.rz(
                angle,
                index,
            )

        # --------------------------------------------------------------
        # COST HAMILTONIAN - CONFLICT TERMS
        # --------------------------------------------------------------

        for u, v in objective.graph.edges():

            i = objective.node_to_index[u]
            j = objective.node_to_index[v]

            angle = (
                2.0
                * gamma
                * objective.penalty
            )

            circuit.rzz(
                angle,
                i,
                j,
            )

        # --------------------------------------------------------------
        # MIXER
        # --------------------------------------------------------------

        for qubit in range(n):

            circuit.rx(
                2.0 * beta,
                qubit,
            )

    # ------------------------------------------------------------------
    # MEASUREMENT
    # ------------------------------------------------------------------

    circuit.measure(
        range(n),
        range(n),
    )

    return circuit


# ============================================================================
# QAOA RESULT SELECTION
# ============================================================================

def select_best_qaoa_bitstring(
    counts: Dict[str, int],
    objective: MWISObjective,
) -> Tuple[str, List[int], float]:
    """
    Selecciona la solución observada con menor energía QUBO.

    En caso de empate:

        mayor número de shots.

    Esto es importante porque la solución con más frecuencia no es
    necesariamente la solución de menor energía.
    """

    if not counts:
        raise ValueError(
            "QAOA no devolvió resultados."
        )

    candidates = []

    for bitstring, shots in counts.items():

        try:
            bits = objective.bitstring_to_bits(
                bitstring
            )

            energy = objective.qubo_energy(
                bits
            )

            candidates.append(
                (
                    energy,
                    -int(shots),
                    str(bitstring),
                    bits,
                )
            )

        except Exception as exc:

            logger.warning(
                "Bitstring inválido ignorado: %r (%s)",
                bitstring,
                exc,
            )

    if not candidates:
        raise RuntimeError(
            "No se encontraron bitstrings válidos."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    energy, _, bitstring, bits = candidates[0]

    return (
        bitstring,
        bits,
        float(energy),
    )


# ============================================================================
# BITSTRING → NODES
# ============================================================================

def bitstring_to_selected_nodes(
    bitstring: str,
    objective: MWISObjective,
) -> List[str]:
    """
    Convierte bitstring en lista de nodos seleccionados.
    """

    bits = objective.bitstring_to_bits(
        bitstring
    )

    selected = []

    for index, node in enumerate(
        objective.nodes
    ):

        if bits[index] == 1:
            selected.append(node)

    return selected


# ============================================================================
# COVERAGE
# ============================================================================

def calculate_coverage(
    graph: nx.Graph,
    selected_nodes: List[str],
) -> Dict[str, Any]:
    """
    Calcula cobertura de viajes.

    Un viaje se considera cubierto si aparece en al menos uno de los
    nodos seleccionados.
    """

    scheduled_trips = set()
    covered_trips = set()

    for node in graph.nodes():

        trips = graph.nodes[node].get(
            "trips",
            [],
        )

        if trips is None:
            trips = []

        if not isinstance(
            trips,
            (list, tuple),
        ):
            trips = [trips]

        for trip in trips:

            scheduled_trips.add(
                str(trip)
            )

    for node in selected_nodes:

        if node not in graph:
            continue

        trips = graph.nodes[node].get(
            "trips",
            [],
        )

        if trips is None:
            trips = []

        if not isinstance(
            trips,
            (list, tuple),
        ):
            trips = [trips]

        for trip in trips:

            covered_trips.add(
                str(trip)
            )

    total = len(
        scheduled_trips
    )

    covered = len(
        covered_trips
    )

    coverage_percent = (
        100.0 * covered / total
        if total > 0
        else 0.0
    )

    return {
        "scheduled_trips": total,
        "covered_trips": covered,
        "coverage_percent": coverage_percent,
        "scheduled_trip_ids": sorted(
            scheduled_trips
        ),
        "covered_trip_ids": sorted(
            covered_trips
        ),
    }


# ============================================================================
# OPERATIONAL METRICS
# ============================================================================

def calculate_operational_metrics(
    graph: nx.Graph,
    selected_nodes: List[str],
) -> Dict[str, Any]:
    """
    Calcula métricas operativas agregadas de los ciclos seleccionados.

    Campos utilizados si están disponibles:

        passenger_km
        empty_km
        operating_cost
    """

    passenger_km = 0.0
    empty_km = 0.0
    operating_cost = 0.0

    for node in selected_nodes:

        if node not in graph:
            continue

        attributes = graph.nodes[node]

        try:
            passenger_km += float(
                attributes.get(
                    "passenger_km",
                    0.0,
                )
            )
        except Exception:
            pass

        try:
            empty_km += float(
                attributes.get(
                    "empty_km",
                    0.0,
                )
            )
        except Exception:
            pass

        try:
            operating_cost += float(
                attributes.get(
                    "operating_cost",
                    0.0,
                )
            )
        except Exception:
            pass

    return {
        "total_passenger_km": passenger_km,
        "total_empty_km": empty_km,
        "total_operating_cost": operating_cost,
    }


# ============================================================================
# QUANTUM BACKEND - AER
# ============================================================================

def run_aer(
    circuit: "QuantumCircuit",
    shots: int,
) -> Tuple[Dict[str, int], str]:
    """
    Ejecuta el circuito localmente utilizando AerSimulator.
    """

    if not AER_AVAILABLE:
        raise RuntimeError(
            "Qiskit Aer no está disponible."
        )

    backend = AerSimulator()

    transpiled = transpile(
        circuit,
        backend,
    )

    result = backend.run(
        transpiled,
        shots=shots,
    ).result()

    counts = result.get_counts()

    return (
        counts,
        "AerSimulator",
    )


# ============================================================================
# QUANTUM BACKEND - IQM
# ============================================================================

def run_iqm(
    circuit: "QuantumCircuit",
    shots: int,
    token: str,
    server_url: str,
    quantum_computer: str,
) -> Tuple[Dict[str, int], str]:
    """
    Ejecuta el circuito sobre IQM cuando las librerías y credenciales
    están disponibles.
    """

    if not IQM_AVAILABLE:
        raise RuntimeError(
            "El paquete IQM Qiskit no está disponible."
        )

    if not token:
        raise RuntimeError(
            "No se ha proporcionado token IQM."
        )

    server_url = URLSanitizer.normalize(
        server_url
    )

    if not server_url:
        raise RuntimeError(
            "server_url no válido."
        )

    provider = IQMProvider(
        server_url,
        token=token,
    )

    backend = provider.get_backend(
        quantum_computer
    )

    transpiled = transpile(
        circuit,
        backend=backend,
    )

    result = backend.run(
        transpiled,
        shots=shots,
    ).result()

    counts = result.get_counts()

    return (
        counts,
        quantum_computer,
    )


# ============================================================================
# PARAMETER EXTRACTION
# ============================================================================

def get_parameter(
    name: str,
    solver_params: Optional[Dict[str, Any]],
    extra_arguments: Optional[Dict[str, Any]],
    environment_name: Optional[str] = None,
    default: Any = None,
) -> Any:
    """
    Obtiene parámetros respetando el orden:

        solver_params
        extra_arguments
        environment
        default
    """

    if isinstance(
        solver_params,
        dict,
    ):

        if name in solver_params:

            value = solver_params[name]

            if value is not None:
                return value

    if isinstance(
        extra_arguments,
        dict,
    ):

        if name in extra_arguments:

            value = extra_arguments[name]

            if value is not None:
                return value

    if environment_name:

        value = os.getenv(
            environment_name
        )

        if value is not None:
            return value

    return default


# ============================================================================
# SAFE NUMERIC CONVERSION
# ============================================================================

def safe_int(
    value: Any,
    default: int,
) -> int:

    try:
        return int(value)

    except Exception:
        return default


def safe_float(
    value: Any,
    default: float,
) -> float:

    try:
        return float(value)

    except Exception:
        return default


# ============================================================================
# MAIN Q-CENTROID ENTRY POINT
# ============================================================================

def run(
    input_data: Dict[str, Any],
    solver_params: Optional[Dict[str, Any]] = None,
    extra_arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Entry point principal de QCentroid.

    QCentroid llama a:

        run(
            input_data,
            solver_params,
            extra_arguments
        )

    `input_data` contiene el dataset proporcionado por la plataforma.
    """

    start_time = time.perf_counter()

    # =====================================================================
    # VALIDATE INPUT
    # =====================================================================

    if not isinstance(
        input_data,
        dict,
    ):
        raise ValueError(
            "input_data debe ser un objeto JSON/dict."
        )

    if "nodes" not in input_data:
        raise ValueError(
            "El dataset debe contener el campo 'nodes'."
        )

    # =====================================================================
    # PARAMETERS
    # =====================================================================

    penalty_value = get_parameter(
        "penalty",
        solver_params,
        extra_arguments,
        default=None,
    )

    if penalty_value is not None:
        penalty_value = safe_float(
            penalty_value,
            0.0,
        )

    qaoa_depth = safe_int(
        get_parameter(
            "qaoa_depth",
            solver_params,
            extra_arguments,
            default=2,
        ),
        2,
    )

    shots = safe_int(
        get_parameter(
            "shots",
            solver_params,
            extra_arguments,
            default=2048,
        ),
        2048,
    )

    shots = max(
        shots,
        1,
    )

    quantum_computer = str(
        get_parameter(
            "quantum_computer",
            solver_params,
            extra_arguments,
            default="emerald",
        )
    )

    server_url = str(
        get_parameter(
            "server_url",
            solver_params,
            extra_arguments,
            environment_name="IQM_SERVER_URL",
            default="https://resonance.iqm.tech/",
        )
    )

    iqm_token = get_parameter(
        "token",
        solver_params,
        extra_arguments,
        environment_name="IQM_TOKEN",
        default=None,
    )

    # =====================================================================
    # BUILD GRAPH
    # =====================================================================

    graph = build_conflict_graph(
        input_data
    )

    if graph.number_of_nodes() == 0:

        execution_time = (
            time.perf_counter()
            - start_time
        )

        return {
            "total_weight": 0.0,
            "optimality_gap_percent": 0.0,
            "coverage_percent": 0.0,
            "execution_time_seconds": execution_time,
            "total_operating_cost": 0.0,
            "selected_cycles": [],
            "is_feasible": True,
            "total_passenger_km": 0.0,
            "total_empty_km": 0.0,
            "scheduled_trips": 0,
            "covered_trips": 0,
            "backend_used": "none",
            "execution_metrics": {},
            "ground_truth_exact": {
                "selected_cycles": [],
                "total_weight": 0.0,
            },
            "nodes_pruned": [],
            "pruning_stats": {
                "num_pruned": 0,
            },
            "assets": [],
        }

    # =====================================================================
    # BUILD MWIS / QUBO
    # =====================================================================

    objective = MWISObjective(
        graph,
        penalty=penalty_value,
    )

    # =====================================================================
    # CLASSICAL GROUND TRUTH
    # =====================================================================

    exact_nodes, exact_weight = (
        objective.exact_solution()
    )

    logger.info(
        "Ground truth exacto: %s | peso=%.3f",
        exact_nodes,
        exact_weight,
    )

    # =====================================================================
    # BUILD QAOA
    # =====================================================================

    circuit = None
    counts = None
    backend_used = "classical_fallback"

    quantum_error = None

    if QISKIT_AVAILABLE:

        try:

            circuit = build_qaoa_circuit(
                objective,
                depth=qaoa_depth,
            )

        except Exception as exc:

            quantum_error = (
                f"Error construyendo QAOA: {exc}"
            )

            logger.warning(
                quantum_error
            )

    else:

        quantum_error = (
            "Qiskit no está disponible."
        )

        logger.warning(
            quantum_error
        )

    # =====================================================================
    # EXECUTE ON IQM IF CONFIGURED
    # =====================================================================

    if (
        circuit is not None
        and iqm_token
        and IQM_AVAILABLE
    ):

        try:

            logger.info(
                "Ejecutando QAOA en IQM: %s",
                quantum_computer,
            )

            counts, backend_used = run_iqm(
                circuit=circuit,
                shots=shots,
                token=str(iqm_token),
                server_url=server_url,
                quantum_computer=quantum_computer,
            )

            logger.info(
                "Ejecución IQM completada."
            )

        except Exception as exc:

            quantum_error = (
                f"IQM no disponible/falló: {exc}"
            )

            logger.warning(
                quantum_error
            )

    # =====================================================================
    # FALLBACK TO AER
    # =====================================================================

    if (
        counts is None
        and circuit is not None
        and AER_AVAILABLE
    ):

        try:

            logger.info(
                "Ejecutando QAOA con AerSimulator."
            )

            counts, backend_used = run_aer(
                circuit=circuit,
                shots=shots,
            )

            logger.info(
                "Ejecución Aer completada."
            )

        except Exception as exc:

            quantum_error = (
                f"AerSimulator falló: {exc}"
            )

            logger.warning(
                quantum_error
            )

    # =====================================================================
    # QUANTUM RESULT
    # =====================================================================

    if counts:

        try:

            best_bitstring, bits, raw_energy = (
                select_best_qaoa_bitstring(
                    counts,
                    objective,
                )
            )

            selected_nodes_raw = (
                bitstring_to_selected_nodes(
                    best_bitstring,
                    objective,
                )
            )

        except Exception as exc:

            logger.warning(
                "No se pudo interpretar la salida cuántica: %s",
                exc,
            )

            best_bitstring = None
            bits = None
            raw_energy = None
            selected_nodes_raw = []

    else:

        best_bitstring = None
        bits = None
        raw_energy = None
        selected_nodes_raw = []

    # =====================================================================
    # CLASSICAL FALLBACK
    # =====================================================================

    if not selected_nodes_raw:

        logger.warning(
            "No se obtuvo solución cuántica válida. "
            "Se utiliza el ground truth clásico como fallback."
        )

        selected_nodes_raw = list(
            exact_nodes
        )

        backend_used = (
            backend_used
            if backend_used != "classical_fallback"
            else "classical_fallback"
        )

    # =====================================================================
    # CLASSICAL REPAIR
    # =====================================================================

    pruner = MWISPruner(
        graph
    )

    selected_nodes, nodes_pruned = (
        pruner.repair(
            selected_nodes_raw
        )
    )

    # =====================================================================
    # VALIDATION
    # =====================================================================

    selected_bits = [
        1 if node in selected_nodes
        else 0
        for node in objective.nodes
    ]

    is_feasible = objective.is_feasible(
        selected_bits
    )

    total_weight = objective.solution_weight(
        selected_bits
    )

    qubo_energy = objective.qubo_energy(
        selected_bits
    )

    # =====================================================================
    # OPTIMALITY GAP
    # =====================================================================

    if exact_weight > 0:

        optimality_gap_percent = (
            100.0
            * (
                exact_weight
                - total_weight
            )
            / exact_weight
        )

    else:

        optimality_gap_percent = 0.0

    # Evitamos valores negativos por pequeños errores numéricos.
    optimality_gap_percent = max(
        0.0,
        float(
            optimality_gap_percent
        ),
    )

    # =====================================================================
    # COVERAGE
    # =====================================================================

    coverage = calculate_coverage(
        graph,
        selected_nodes,
    )

    # =====================================================================
    # OPERATIONAL METRICS
    # =====================================================================

    operational = (
        calculate_operational_metrics(
            graph,
            selected_nodes,
        )
    )

    # =====================================================================
    # VISUAL ASSET
    # =====================================================================

    assets = []

    try:

        asset_path = (
            VisualizationAssetGenerator
            .generate_conflict_graph(
                graph
            )
        )

        if asset_path:
            assets.append(
                asset_path
            )

    except Exception as exc:

        logger.warning(
            "No se pudo generar asset: %s",
            exc,
        )

    # =====================================================================
    # EXECUTION TIME
    # =====================================================================

    execution_time = (
        time.perf_counter()
        - start_time
    )

    # =====================================================================
    # EXECUTION METRICS
    # =====================================================================

    execution_metrics = {
        "shots": shots,
        "qaoa_depth": qaoa_depth,
        "num_qubits": graph.number_of_nodes(),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "penalty": objective.penalty,
        "qiskit_available": QISKIT_AVAILABLE,
        "aer_available": AER_AVAILABLE,
        "iqm_available": IQM_AVAILABLE,
        "quantum_execution_attempted": (
            circuit is not None
        ),
        "quantum_execution_successful": (
            counts is not None
        ),
        "best_bitstring": best_bitstring,
        "raw_qubo_energy": raw_energy,
        "validated_qubo_energy": qubo_energy,
    }

    if quantum_error:
        execution_metrics[
            "quantum_error"
        ] = quantum_error

    if counts:
        execution_metrics[
            "num_observed_bitstrings"
        ] = len(counts)

        execution_metrics[
            "max_observed_shots"
        ] = max(
            counts.values()
        )

    # =====================================================================
    # RESULT
    # =====================================================================

    result = {
        # ---------------------------------------------------------------
        # PRIMARY BENCHMARK METRICS
        # ---------------------------------------------------------------

        "total_weight": total_weight,

        "optimality_gap_percent": (
            optimality_gap_percent
        ),

        "coverage_percent": (
            coverage["coverage_percent"]
        ),

        "execution_time_seconds": (
            execution_time
        ),

        "total_operating_cost": (
            operational[
                "total_operating_cost"
            ]
        ),

        # ---------------------------------------------------------------
        # SOLUTION
        # ---------------------------------------------------------------

        "selected_cycles": selected_nodes,

        "is_feasible": is_feasible,

        # ---------------------------------------------------------------
        # OPERATIONAL METRICS
        # ---------------------------------------------------------------

        "total_passenger_km": (
            operational[
                "total_passenger_km"
            ]
        ),

        "total_empty_km": (
            operational[
                "total_empty_km"
            ]
        ),

        "scheduled_trips": (
            coverage[
                "scheduled_trips"
            ]
        ),

        "covered_trips": (
            coverage[
                "covered_trips"
            ]
        ),

        "scheduled_trip_ids": (
            coverage[
                "scheduled_trip_ids"
            ]
        ),

        "covered_trip_ids": (
            coverage[
                "covered_trip_ids"
            ]
        ),

        # ---------------------------------------------------------------
        # BACKEND
        # ---------------------------------------------------------------

        "backend_used": backend_used,

        # ---------------------------------------------------------------
        # EXECUTION
        # ---------------------------------------------------------------

        "execution_metrics": execution_metrics,

        # ---------------------------------------------------------------
        # CLASSICAL GROUND TRUTH
        # ---------------------------------------------------------------

        "ground_truth_exact": {
            "selected_cycles": exact_nodes,
            "total_weight": exact_weight,
        },

        # ---------------------------------------------------------------
        # REPAIR / PRUNING
        # ---------------------------------------------------------------

        "nodes_pruned": nodes_pruned,

        "pruning_stats": {
            "num_pruned": len(
                nodes_pruned
            ),
            "initial_selected": len(
                selected_nodes_raw
            ),
            "final_selected": len(
                selected_nodes
            ),
        },

        # ---------------------------------------------------------------
        # ASSETS
        # ---------------------------------------------------------------

        "assets": assets,
    }

    logger.info(
        "Resultado final: selected=%s weight=%.3f "
        "feasible=%s gap=%.3f%% coverage=%.3f%% "
        "backend=%s",
        selected_nodes,
        total_weight,
        is_feasible,
        optimality_gap_percent,
        coverage["coverage_percent"],
        backend_used,
    )

    return result
