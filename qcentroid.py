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
    IQM Emerald / AerSimulator
        ↓
    Solución candidata
        ↓
    Reparación clásica, si es necesaria
        ↓
    Validación clásica
        ↓
    Métricas operativas

IMPORTANTE
----------

- QCentroid suministra el dataset mediante `input_data`.
- El dataset NO está hardcodeado en este fichero.
- QCentroid suministra los parámetros mediante `extra_arguments`.

Formato esperado de extra_arguments:

{
    "iqm_token": "...",
    "quantum_computer": "emerald",
    "shots": 2048,
    "qaoa_depth": 2
}

Entrada principal:

    run(input_data, solver_params, extra_arguments)

NOTA SOBRE EL BENCHMARK
-----------------------

El ground truth exacto se calcula clásicamente únicamente para:

    - validar la solución;
    - calcular el optimality gap.

NO se utiliza el ground truth como sustituto de la solución cuántica.

Si falla IQM:

    - se puede utilizar AerSimulator si `allow_aer_fallback=True`;
    - el resultado queda identificado explícitamente como AerSimulator.

Si falla tanto IQM como Aer:

    - la ejecución termina con error;
    - no se fabrica una solución cuántica mediante el ground truth.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


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

logger = logging.getLogger(
    "qcentroid-user-log"
)

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
    Normaliza una URL de servidor IQM.

    No modifica rutas internas salvo la eliminación de barras finales.
    """

    @staticmethod
    def normalize(
        url: Optional[str],
    ) -> Optional[str]:

        if url is None:
            return None

        url = str(url).strip()

        if not url:
            return None

        return url.rstrip("/")


# ============================================================================
# MWIS OBJECTIVE
# ============================================================================

class MWISObjective:
    """
    Representación del problema Maximum Weight Independent Set.

    Para cada nodo i:

        x_i = 1  -> nodo seleccionado
        x_i = 0  -> nodo no seleccionado

    Función QUBO:

        E(x) =
            - Σ w_i x_i
            + λ Σ x_i x_j

    donde (i,j) son aristas del grafo de conflictos.

    Minimizar E equivale a maximizar el peso evitando conflictos.
    """

    def __init__(
        self,
        graph: nx.Graph,
        penalty: Optional[float] = None,
    ):

        self.graph = graph

        # Orden determinista.
        self.nodes = sorted(
            list(graph.nodes()),
            key=lambda x: str(x),
        )

        self.node_to_index = {
            node: index
            for index, node in enumerate(
                self.nodes
            )
        }

        self.weights: Dict[str, float] = {}

        for node in self.nodes:

            weight = graph.nodes[node].get(
                "weight",
                1.0,
            )

            try:
                weight = float(weight)

            except Exception:
                weight = 1.0

            self.weights[node] = weight

        max_weight = max(
            self.weights.values(),
            default=1.0,
        )

        # Penalización suficientemente superior al peso máximo.
        self.penalty = (
            float(penalty)
            if penalty is not None
            else 4.0 * max_weight
        )

    # ---------------------------------------------------------------------

    def bitstring_to_bits(
        self,
        bitstring: str,
    ) -> List[int]:
        """
        Convierte un bitstring de Qiskit al orden de self.nodes.

        Qiskit utiliza representación little-endian en los registros
        de medida, por lo que se invierte el string.
        """

        clean = str(
            bitstring
        ).replace(
            " ",
            "",
        )

        if not clean:
            raise ValueError(
                "Bitstring vacío."
            )

        if any(
            bit not in ("0", "1")
            for bit in clean
        ):
            raise ValueError(
                f"Bitstring inválido: {bitstring}"
            )

        bits = [
            int(bit)
            for bit in clean
        ]

        if len(bits) != len(
            self.nodes
        ):
            raise ValueError(
                "Longitud del bitstring "
                f"({len(bits)}) distinta del "
                f"número de nodos "
                f"({len(self.nodes)})."
            )

        return list(
            reversed(bits)
        )

    # ---------------------------------------------------------------------

    def qubo_energy(
        self,
        bits: List[int],
    ) -> float:
        """
        Calcula la energía QUBO.
        """

        if len(bits) != len(
            self.nodes
        ):
            raise ValueError(
                "Número de bits incompatible "
                "con el número de nodos."
            )

        energy = 0.0

        # Términos lineales.
        for index, node in enumerate(
            self.nodes
        ):

            x = bits[index]

            energy -= (
                self.weights[node]
                * x
            )

        # Penalizaciones de conflictos.
        for u, v in self.graph.edges():

            i = self.node_to_index[u]
            j = self.node_to_index[v]

            energy += (
                self.penalty
                * bits[i]
                * bits[j]
            )

        return float(
            energy
        )

    # ---------------------------------------------------------------------

    def solution_weight(
        self,
        bits: List[int],
    ) -> float:
        """
        Peso total de los nodos seleccionados.
        """

        total = 0.0

        for index, node in enumerate(
            self.nodes
        ):

            if bits[index]:
                total += (
                    self.weights[node]
                )

        return float(
            total
        )

    # ---------------------------------------------------------------------

    def is_feasible(
        self,
        bits: List[int],
    ) -> bool:
        """
        Comprueba si la solución es un conjunto independiente.
        """

        if len(bits) != len(
            self.nodes
        ):
            return False

        for u, v in self.graph.edges():

            i = self.node_to_index[u]
            j = self.node_to_index[v]

            if (
                bits[i] == 1
                and bits[j] == 1
            ):
                return False

        return True

    # ---------------------------------------------------------------------

    def exact_solution(
        self,
    ) -> Tuple[List[str], float]:
        """
        Calcula el MWIS exacto mediante Maximum Weight Clique
        sobre el grafo complemento.

        Se utiliza exclusivamente como ground truth clásico.

        Esta operación puede crecer exponencialmente y, por tanto,
        está destinada a datasets pequeños de benchmark.
        """

        if self.graph.number_of_nodes() == 0:
            return [], 0.0

        complement = nx.complement(
            self.graph
        )

        # Escalado para preservar precisión.
        scale = 1000

        for node in complement.nodes():

            weight = self.weights.get(
                node,
                0.0,
            )

            complement.nodes[node][
                "weight"
            ] = int(
                round(
                    weight * scale
                )
            )

        clique = (
            nx.algorithms.clique
            .max_weight_clique(
                complement,
                weight="weight",
            )
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

        return (
            selected,
            float(total_weight),
        )


# ============================================================================
# CLASSICAL REPAIR / PRUNING
# ============================================================================

class MWISPruner:
    """
    Reparación clásica de una solución que contiene conflictos.

    Si una solución contiene dos nodos conectados:

        u -- v

    se elimina el nodo con menor relación:

        weight / degree

    La reparación se ejecuta DESPUÉS de obtener la solución cuántica.
    """

    def __init__(
        self,
        graph: nx.Graph,
    ):

        self.graph = graph

    # ---------------------------------------------------------------------

    def repair(
        self,
        selected_nodes: List[str],
    ) -> Tuple[List[str], List[str]]:

        selected = set(
            selected_nodes
        )

        pruned: List[str] = []

        while True:

            conflicts = []

            for u, v in self.graph.edges():

                if (
                    u in selected
                    and v in selected
                ):
                    conflicts.append(
                        (u, v)
                    )

            if not conflicts:
                break

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

            ratio_u = (
                weight_u
                / degree_u
            )

            ratio_v = (
                weight_v
                / degree_v
            )

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

            selected.remove(
                remove_node
            )

            pruned.append(
                remove_node
            )

        repaired = sorted(
            selected,
            key=lambda x: str(x),
        )

        return (
            repaired,
            pruned,
        )


# ============================================================================
# VISUALIZATION
# ============================================================================

class VisualizationAssetGenerator:
    """
    Genera activos visuales opcionales.
    """

    @staticmethod
    def generate_conflict_graph(
        graph: nx.Graph,
        output_dir: str = "additional_output",
    ) -> Optional[str]:

        try:

            import matplotlib

            matplotlib.use(
                "Agg"
            )

            import matplotlib.pyplot as plt

        except Exception as exc:

            logger.warning(
                "No se pudo importar matplotlib: %s",
                exc,
            )

            return None

        try:

            output_path = Path(
                output_dir
            )

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

            plt.axis(
                "off"
            )

            plt.tight_layout()

            plt.savefig(
                file_path,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close()

            return str(
                file_path
            )

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
    Construye el grafo de conflictos.

    Formato:

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

    Si existen edges explícitos, se utilizan.

    Si no existen edges, se infieren conflictos mediante viajes
    compartidos.
    """

    graph = nx.Graph()

    if not isinstance(
        input_data,
        dict,
    ):
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

    if not isinstance(
        nodes_raw,
        list,
    ):
        raise ValueError(
            "'nodes' debe ser una lista."
        )

    # ------------------------------------------------------------------
    # NODES
    # ------------------------------------------------------------------

    for node_data in nodes_raw:

        if not isinstance(
            node_data,
            dict,
        ):

            logger.warning(
                "Nodo ignorado por formato inválido: %r",
                node_data,
            )

            continue

        node_id = node_data.get(
            "id"
        )

        if node_id is None:

            logger.warning(
                "Nodo ignorado porque no tiene 'id': %r",
                node_data,
            )

            continue

        node_id = str(
            node_id
        )

        weight = node_data.get(
            "weight",
            1.0,
        )

        try:

            weight = float(
                weight
            )

        except Exception:

            logger.warning(
                "Peso inválido para %s. Se utiliza 1.0.",
                node_id,
            )

            weight = 1.0

        attributes = dict(
            node_data
        )

        attributes[
            "weight"
        ] = weight

        graph.add_node(
            node_id,
            **attributes,
        )

    # ------------------------------------------------------------------
    # EXPLICIT EDGES
    # ------------------------------------------------------------------

    if edges_raw:

        if not isinstance(
            edges_raw,
            list,
        ):
            raise ValueError(
                "'edges' debe ser una lista."
            )

        for edge in edges_raw:

            # IMPORTANTE:
            #
            # El formato QCentroid es:
            #
            # ["C01", "C02"]
            #
            # Por tanto:
            #
            # edge[0] = C01
            # edge[1] = C02
            #
            # Nunca edge[3].

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

            u = str(
                edge[0]
            )

            v = str(
                edge[1]
            )

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

        trip_to_nodes: Dict[
            str,
            List[str],
        ] = {}

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

                trip = str(
                    trip
                )

                trip_to_nodes.setdefault(
                    trip,
                    [],
                ).append(
                    node
                )

        for nodes in trip_to_nodes.values():

            for i in range(
                len(nodes)
            ):

                for j in range(
                    i + 1,
                    len(nodes),
                ):

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
    Construye el circuito QAOA.

    IMPORTANTE:

    Esta implementación utiliza parámetros gamma/beta deterministas:

        gamma = 0.05 / (layer + 1)
        beta  = 0.25 / (layer + 1)

    Por tanto, se trata de QAOA con parámetros prefijados.

    No existe todavía un bucle de optimización clásico de gamma/beta.
    """

    if not QISKIT_AVAILABLE:

        raise RuntimeError(
            "Qiskit no está disponible."
        )

    n = len(
        objective.nodes
    )

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
    # INITIAL SUPERPOSITION
    # ------------------------------------------------------------------

    for qubit in range(n):

        circuit.h(
            qubit
        )

    # ------------------------------------------------------------------
    # QAOA LAYERS
    # ------------------------------------------------------------------

    for layer in range(
        depth
    ):

        gamma = (
            0.05
            / (layer + 1)
        )

        beta = (
            0.25
            / (layer + 1)
        )

        # --------------------------------------------------------------
        # LINEAR COST TERMS
        # --------------------------------------------------------------

        for index, node in enumerate(
            objective.nodes
        ):

            weight = (
                objective.weights[
                    node
                ]
            )

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
        # CONFLICT TERMS
        # --------------------------------------------------------------

        for u, v in (
            objective.graph.edges()
        ):

            i = (
                objective.node_to_index[
                    u
                ]
            )

            j = (
                objective.node_to_index[
                    v
                ]
            )

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
) -> Tuple[
    str,
    List[int],
    float,
]:
    """
    Selecciona el bitstring observado con menor energía QUBO.

    Desempate:

        mayor número de shots.
    """

    if not counts:

        raise ValueError(
            "QAOA no devolvió resultados."
        )

    candidates = []

    for bitstring, shots in (
        counts.items()
    ):

        try:

            bits = (
                objective
                .bitstring_to_bits(
                    bitstring
                )
            )

            energy = (
                objective
                .qubo_energy(
                    bits
                )
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
            "No se encontraron "
            "bitstrings válidos."
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    energy, _, bitstring, bits = (
        candidates[0]
    )

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

    bits = (
        objective
        .bitstring_to_bits(
            bitstring
        )
    )

    selected = []

    for index, node in enumerate(
        objective.nodes
    ):

        if bits[index] == 1:

            selected.append(
                node
            )

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

    Un viaje está cubierto si aparece en al menos uno de los ciclos
    seleccionados.
    """

    scheduled_trips = set()
    covered_trips = set()

    # Todos los viajes.
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

    # Viajes cubiertos.
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
        100.0
        * covered
        / total
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
    Calcula:

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

        attributes = graph.nodes[
            node
        ]

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
        "total_passenger_km": (
            passenger_km
        ),
        "total_empty_km": (
            empty_km
        ),
        "total_operating_cost": (
            operating_cost
        ),
    }


# ============================================================================
# AER BACKEND
# ============================================================================

def run_aer(
    circuit: "QuantumCircuit",
    shots: int,
) -> Tuple[
    Dict[str, int],
    str,
]:
    """
    Ejecuta el circuito en AerSimulator.
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

    if not counts:

        raise RuntimeError(
            "AerSimulator no devolvió counts."
        )

    return (
        counts,
        "AerSimulator",
    )


# ============================================================================
# IQM BACKEND
# ============================================================================

def run_iqm(
    circuit: "QuantumCircuit",
    shots: int,
    token: str,
    server_url: str,
    quantum_computer: str,
) -> Tuple[
    Dict[str, int],
    str,
]:
    """
    Ejecuta el circuito sobre IQM.

    Parámetros recibidos desde QCentroid:

        iqm_token
        quantum_computer
        shots

    La credencial nunca se registra en logs.
    """

    if not IQM_AVAILABLE:

        raise RuntimeError(
            "El paquete iqm.qiskit_iqm "
            "no está disponible."
        )

    if not token:

        raise RuntimeError(
            "No se ha proporcionado iqm_token."
        )

    normalized_url = (
        URLSanitizer.normalize(
            server_url
        )
    )

    if not normalized_url:

        raise RuntimeError(
            "server_url de IQM no es válida."
        )

    logger.info(
        "Configurando IQMProvider: server=%s device=%s",
        normalized_url,
        quantum_computer,
    )

    # IMPORTANTE:
    # El token se pasa al proveedor, pero nunca se imprime.
    provider = IQMProvider(
        normalized_url,
        token=token,
    )

    logger.info(
        "Obteniendo backend IQM: %s",
        quantum_computer,
    )

    backend = provider.get_backend(
        quantum_computer
    )

    logger.info(
        "Transpilando circuito para IQM %s.",
        quantum_computer,
    )

    transpiled = transpile(
        circuit,
        backend=backend,
    )

    logger.info(
        "Enviando circuito a IQM %s (%d shots).",
        quantum_computer,
        shots,
    )

    result = backend.run(
        transpiled,
        shots=shots,
    ).result()

    counts = result.get_counts()

    if not counts:

        raise RuntimeError(
            "IQM no devolvió counts."
        )

    logger.info(
        "Ejecución IQM completada: %d bitstrings observados.",
        len(counts),
    )

    return (
        counts,
        quantum_computer,
    )


# ============================================================================
# PARAMETER EXTRACTION
# ============================================================================

def get_parameter(
    name: str,
    solver_params: Optional[
        Dict[str, Any]
    ],
    extra_arguments: Optional[
        Dict[str, Any]
    ],
    environment_name: Optional[str] = None,
    default: Any = None,
) -> Any:
    """
    Prioridad:

        1. solver_params
        2. extra_arguments
        3. environment
        4. default
    """

    if isinstance(
        solver_params,
        dict,
    ):

        value = solver_params.get(
            name
        )

        if value is not None:
            return value

    if isinstance(
        extra_arguments,
        dict,
    ):

        value = extra_arguments.get(
            name
        )

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
# SAFE CONVERSIONS
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


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    """
    Conversión robusta de booleanos procedentes de JSON/configuración.
    """

    if value is None:
        return default

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        str,
    ):

        normalized = (
            value.strip()
            .lower()
        )

        if normalized in (
            "true",
            "1",
            "yes",
            "y",
            "on",
        ):
            return True

        if normalized in (
            "false",
            "0",
            "no",
            "n",
            "off",
        ):
            return False

    return default


# ============================================================================
# MAIN Q-CENTROID ENTRY POINT
# ============================================================================

def run(
    input_data: Dict[str, Any],
    solver_params: Optional[
        Dict[str, Any]
    ] = None,
    extra_arguments: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Entry point de QCentroid.

    QCentroid llama:

        run(
            input_data,
            solver_params,
            extra_arguments
        )
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
    # READ Q-CENTROID PARAMETERS
    # =====================================================================

    # ---------------------------------------------------------------
    # IQM TOKEN
    #
    # QCentroid utiliza explícitamente:
    #
    #     "iqm_token"
    #
    # NO "token".
    # ---------------------------------------------------------------

    iqm_token = get_parameter(
        "iqm_token",
        solver_params,
        extra_arguments,
        environment_name="IQM_TOKEN",
        default=None,
    )

    # ---------------------------------------------------------------
    # QUANTUM COMPUTER
    # ---------------------------------------------------------------

    quantum_computer = str(
        get_parameter(
            "quantum_computer",
            solver_params,
            extra_arguments,
            default="emerald",
        )
    ).strip()

    # ---------------------------------------------------------------
    # SHOTS
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # QAOA DEPTH
    # ---------------------------------------------------------------

    qaoa_depth = safe_int(
        get_parameter(
            "qaoa_depth",
            solver_params,
            extra_arguments,
            default=2,
        ),
        2,
    )

    qaoa_depth = max(
        qaoa_depth,
        1,
    )

    # ---------------------------------------------------------------
    # PENALTY
    # ---------------------------------------------------------------

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

        if penalty_value <= 0:

            penalty_value = None

    # ---------------------------------------------------------------
    # IQM SERVER URL
    # ---------------------------------------------------------------
    #
    # Se puede proporcionar:
    #
    #   iqm_server_url
    #
    # o:
    #
    #   server_url
    #
    # o:
    #
    #   IQM_SERVER_URL
    #
    # El valor por defecto corresponde al endpoint identificado por
    # QCentroid en la configuración del dispositivo Emerald.
    # ---------------------------------------------------------------

    server_url = get_parameter(
        "iqm_server_url",
        solver_params,
        extra_arguments,
        environment_name="IQM_SERVER_URL",
        default=None,
    )

    if not server_url:

        server_url = get_parameter(
            "server_url",
            solver_params,
            extra_arguments,
            environment_name="IQM_SERVER_URL",
            default=None,
        )

    if not server_url:

        server_url = (
            "https://cocos.resonance.meetiqm.com/emerald"
        )

    server_url = str(
        server_url
    )

    # ---------------------------------------------------------------
    # AER FALLBACK
    # ---------------------------------------------------------------
    #
    # Por defecto FALSE.
    #
    # Esto es deliberado:
    # si queremos un benchmark hardware, un fallo de Emerald no debe
    # convertirse silenciosamente en una ejecución de Aer.
    #
    # Se puede activar explícitamente mediante:
    #
    #     "allow_aer_fallback": true
    # ---------------------------------------------------------------

    allow_aer_fallback = safe_bool(
        get_parameter(
            "allow_aer_fallback",
            solver_params,
            extra_arguments,
            default=False,
        ),
        False,
    )

    # =====================================================================
    # LOG CONFIGURATION
    # =====================================================================

    logger.info(
        "Quantum configuration: "
        "device=%s, shots=%d, qaoa_depth=%d, "
        "IQM token available=%s, IQM SDK available=%s, "
        "Aer available=%s, Aer fallback=%s",
        quantum_computer,
        shots,
        qaoa_depth,
        bool(iqm_token),
        IQM_AVAILABLE,
        AER_AVAILABLE,
        allow_aer_fallback,
    )

    logger.info(
        "IQM server configured: %s",
        URLSanitizer.normalize(
            server_url
        ),
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
    # MWIS / QUBO
    # =====================================================================

    objective = MWISObjective(
        graph,
        penalty=penalty_value,
    )

    logger.info(
        "MWIS/QUBO configured: nodes=%d edges=%d penalty=%.6f",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        objective.penalty,
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
    # BUILD QAOA CIRCUIT
    # =====================================================================

    if not QISKIT_AVAILABLE:

        raise RuntimeError(
            "Qiskit no está disponible. "
            "No es posible construir el circuito QAOA."
        )

    try:

        circuit = build_qaoa_circuit(
            objective,
            depth=qaoa_depth,
        )

    except Exception as exc:

        raise RuntimeError(
            f"No se pudo construir el circuito QAOA: {exc}"
        ) from exc

    logger.info(
        "Circuito QAOA construido: qubits=%d depth=%d",
        len(objective.nodes),
        qaoa_depth,
    )

    # =====================================================================
    # EXECUTION
    # =====================================================================

    counts = None
    backend_used = None
    quantum_error = None
    quantum_execution_attempted = False
    quantum_execution_successful = False

    # =====================================================================
    # PRIMARY EXECUTION: IQM
    # =====================================================================

    if not iqm_token:

        quantum_error = (
            "QCentroid no proporcionó 'iqm_token'."
        )

        logger.error(
            quantum_error
        )

    elif not IQM_AVAILABLE:

        quantum_error = (
            "IQM SDK no está disponible "
            "en el entorno del solver."
        )

        logger.error(
            quantum_error
        )

    else:

        quantum_execution_attempted = True

        try:

            logger.info(
                "==================================================="
            )

            logger.info(
                "Ejecutando QAOA en IQM %s.",
                quantum_computer,
            )

            logger.info(
                "Shots: %d | QAOA depth: %d",
                shots,
                qaoa_depth,
            )

            counts, backend_used = run_iqm(
                circuit=circuit,
                shots=shots,
                token=str(iqm_token),
                server_url=server_url,
                quantum_computer=quantum_computer,
            )

            quantum_execution_successful = True

            logger.info(
                "QAOA ejecutado correctamente en IQM %s.",
                quantum_computer,
            )

            logger.info(
                "==================================================="
            )

        except Exception as exc:

            quantum_error = (
                f"IQM execution failed: {exc}"
            )

            logger.error(
                quantum_error,
                exc_info=True,
            )

    # =====================================================================
    # OPTIONAL FALLBACK: AER
    # =====================================================================

    if (
        counts is None
        and allow_aer_fallback
    ):

        if not AER_AVAILABLE:

            logger.error(
                "Aer fallback solicitado, pero "
                "Qiskit Aer no está disponible."
            )

        else:

            logger.warning(
                "IQM no produjo resultado. "
                "Se activa explícitamente el fallback AerSimulator."
            )

            try:

                counts, backend_used = run_aer(
                    circuit=circuit,
                    shots=shots,
                )

                quantum_execution_successful = True

                logger.info(
                    "QAOA ejecutado mediante AerSimulator."
                )

            except Exception as exc:

                quantum_error = (
                    f"AerSimulator fallback failed: {exc}"
                )

                logger.error(
                    quantum_error,
                    exc_info=True,
                )

    # =====================================================================
    # NO QUANTUM RESULT
    # =====================================================================

    if counts is None:

        execution_time = (
            time.perf_counter()
            - start_time
        )

        raise RuntimeError(
            "No se obtuvo resultado cuántico. "
            f"backend={quantum_computer}; "
            f"error={quantum_error}; "
            f"allow_aer_fallback={allow_aer_fallback}; "
            f"execution_time={execution_time:.3f}s"
        )

    # =====================================================================
    # DECODE QUANTUM RESULT
    # =====================================================================

    try:

        (
            best_bitstring,
            bits,
            raw_energy,
        ) = select_best_qaoa_bitstring(
            counts,
            objective,
        )

        selected_nodes_raw = (
            bitstring_to_selected_nodes(
                best_bitstring,
                objective,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "No se pudo interpretar la salida "
            f"cuántica: {exc}"
        ) from exc

    logger.info(
        "Mejor bitstring observado: %s",
        best_bitstring,
    )

    logger.info(
        "Nodos seleccionados por QAOA: %s",
        selected_nodes_raw,
    )

    logger.info(
        "Energía QUBO observada: %.6f",
        raw_energy,
    )

    # =====================================================================
    # CLASSICAL REPAIR
    # =====================================================================

    pruner = MWISPruner(
        graph
    )

    (
        selected_nodes,
        nodes_pruned,
    ) = pruner.repair(
        selected_nodes_raw
    )

    if nodes_pruned:

        logger.warning(
            "La solución QAOA contenía conflictos. "
            "Nodos eliminados durante reparación: %s",
            nodes_pruned,
        )

    else:

        logger.info(
            "La solución QAOA ya era factible. "
            "No fue necesaria reparación."
        )

    # =====================================================================
    # VALIDATION
    # =====================================================================

    selected_bits = [
        1
        if node in selected_nodes
        else 0
        for node in objective.nodes
    ]

    is_feasible = (
        objective.is_feasible(
            selected_bits
        )
    )

    total_weight = (
        objective.solution_weight(
            selected_bits
        )
    )

    qubo_energy = (
        objective.qubo_energy(
            selected_bits
        )
    )

    if not is_feasible:

        raise RuntimeError(
            "La solución después de la reparación "
            "clásica sigue siendo infactible."
        )

    logger.info(
        "Solución validada: selected=%s weight=%.3f feasible=%s",
        selected_nodes,
        total_weight,
        is_feasible,
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

        "num_qubits": (
            graph.number_of_nodes()
        ),

        "num_nodes": (
            graph.number_of_nodes()
        ),

        "num_edges": (
            graph.number_of_edges()
        ),

        "penalty": (
            objective.penalty
        ),

        "qiskit_available": (
            QISKIT_AVAILABLE
        ),

        "aer_available": (
            AER_AVAILABLE
        ),

        "iqm_available": (
            IQM_AVAILABLE
        ),

        "quantum_execution_attempted": (
            quantum_execution_attempted
        ),

        "quantum_execution_successful": (
            quantum_execution_successful
        ),

        "backend_used": (
            backend_used
        ),

        "requested_quantum_computer": (
            quantum_computer
        ),

        "best_bitstring": (
            best_bitstring
        ),

        "raw_qubo_energy": (
            raw_energy
        ),

        "validated_qubo_energy": (
            qubo_energy
        ),

        "num_observed_bitstrings": (
            len(counts)
        ),

        "max_observed_shots": (
            max(
                counts.values()
            )
        ),

        "aer_fallback_enabled": (
            allow_aer_fallback
        ),

        "nodes_pruned": (
            len(nodes_pruned)
        ),

        "execution_time_seconds": (
            execution_time
        ),
    }

    if quantum_error:

        execution_metrics[
            "quantum_error"
        ] = quantum_error

    # =====================================================================
    # RESULT
    # =====================================================================

    result = {

        # ---------------------------------------------------------------
        # PRIMARY BENCHMARK METRICS
        # ---------------------------------------------------------------

        "total_weight": (
            total_weight
        ),

        "optimality_gap_percent": (
            optimality_gap_percent
        ),

        "coverage_percent": (
            coverage[
                "coverage_percent"
            ]
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

        "selected_cycles": (
            selected_nodes
        ),

        "is_feasible": (
            is_feasible
        ),

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

        "backend_used": (
            backend_used
        ),

        # ---------------------------------------------------------------
        # EXECUTION
        # ---------------------------------------------------------------

        "execution_metrics": (
            execution_metrics
        ),

        # ---------------------------------------------------------------
        # CLASSICAL GROUND TRUTH
        #
        # Se utiliza para validación, NO como sustituto de QAOA.
        # ---------------------------------------------------------------

        "ground_truth_exact": {

            "selected_cycles": (
                exact_nodes
            ),

            "total_weight": (
                exact_weight
            ),
        },

        # ---------------------------------------------------------------
        # REPAIR / PRUNING
        # ---------------------------------------------------------------

        "nodes_pruned": (
            nodes_pruned
        ),

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

        "assets": (
            assets
        ),
    }

    # =====================================================================
    # FINAL LOG
    # =====================================================================

    logger.info(
        "Resultado final: "
        "selected=%s "
        "weight=%.3f "
        "feasible=%s "
        "gap=%.3f%% "
        "coverage=%.3f%% "
        "backend=%s "
        "execution_time=%.3fs",
        selected_nodes,
        total_weight,
        is_feasible,
        optimality_gap_percent,
        coverage[
            "coverage_percent"
        ],
        backend_used,
        execution_time,
    )

    return result
