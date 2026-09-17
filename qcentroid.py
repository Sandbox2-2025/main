"""
===============================================================
QCentroid PoC
Railway Rolling Stock Cycle Selection
MWIS + QUBO + QAOA + IQM Resonance
===============================================================

OBJETIVO
--------
Seleccionar un conjunto de ciclos de material rodante que:

    1. No tengan conflictos entre sí.
    2. Maximicen el valor total de los ciclos seleccionados.

El problema se formula como:

    Maximum Weighted Independent Set (MWIS)

Cada nodo representa un ciclo ferroviario.
Cada arista representa un conflicto entre dos ciclos.

La función QUBO utilizada es:

    H(x) = - sum(w_i * x_i)
           + lambda * sum(x_i * x_j)

donde:

    x_i = 1 -> ciclo seleccionado
    x_i = 0 -> ciclo no seleccionado

El primer término maximiza el peso.

El segundo término penaliza la selección simultánea
de ciclos que presentan conflicto.

ARQUITECTURA DE EJECUCIÓN
-------------------------

    Dataset JSON
          |
          v
    Validación de datos
          |
          v
    Grafo de conflictos
          |
          v
    Definición MWIS
          |
          v
    Construcción QUBO
          |
          v
    Circuito QAOA
          |
          v
    IQM Resonance / Emerald
          |
          v
    Bitstrings
          |
          v
    Evaluación QUBO
          |
          v
    Reparación determinista
          |
          v
    Validación contra solución exacta
          |
          v
    Métricas + assets
          |
          v
    Resultado QCentroid

IMPORTANTE
----------
QCentroid proporciona mediante `extra_arguments`:

    iqm_token
    quantum_computer
    shots
    qaoa_depth

Por tanto, el solver NO contiene credenciales.

Si `quantum_computer` está configurado como `emerald`,
el solver ejecuta obligatoriamente contra IQM.

No se realiza fallback silencioso a AerSimulator.
Esto es deliberado: un benchmark hardware debe poder
distinguir inequívocamente una ejecución real de IQM
de una simulación clásica.
"""

# ============================================================
# PASO 1. IMPORTACIONES
# ============================================================

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Tuple


# ------------------------------------------------------------
# Matplotlib
# ------------------------------------------------------------
# QCentroid se ejecuta en un entorno sin interfaz gráfica.
# Por eso utilizamos el backend no interactivo "Agg".
# ------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# NetworkX
# ------------------------------------------------------------

import networkx as nx


# ------------------------------------------------------------
# Qiskit
# ------------------------------------------------------------

try:
    from qiskit import QuantumCircuit, transpile
except ImportError:
    QuantumCircuit = None
    transpile = None


# ------------------------------------------------------------
# Qiskit Aer
# ------------------------------------------------------------
# Se mantiene disponible únicamente como herramienta técnica
# opcional si el usuario solicita explícitamente use_iqm=False.
# NO se utiliza como fallback automático.
# ------------------------------------------------------------

try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None


# ------------------------------------------------------------
# IQM
# ------------------------------------------------------------

try:
    from iqm.qiskit_iqm import IQMProvider
except ImportError:
    IQMProvider = None


# ============================================================
# PASO 2. CONFIGURACIÓN DE LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# PASO 3. SANITIZACIÓN BÁSICA
# ============================================================

class URLSanitizer:
    """
    Pequeña utilidad para normalizar la URL del servidor IQM.

    QCentroid puede proporcionar una URL de dispositivo o una
    URL de servidor dependiendo de la configuración.

    Ejemplos:

        https://cocos.resonance.meetiqm.com
        https://cocos.resonance.meetiqm.com/
        https://cocos.resonance.meetiqm.com/emerald
    """

    @staticmethod
    def normalize_iqm_url(url: str) -> str:
        """
        Normaliza la URL IQM para su utilización por IQMProvider.

        Si la URL termina en el nombre del ordenador cuántico,
        se elimina dicho componente.
        """

        if not url:
            return url

        url = str(url).strip().rstrip("/")

        # Si QCentroid entrega directamente:
        # https://.../emerald
        #
        # IQMProvider normalmente necesita el endpoint del servidor.
        parts = url.split("/")

        if parts[-1].lower() in {
            "emerald",
            "garnet",
            "deneb",
            "adonis",
        }:
            url = "/".join(parts[:-1])

        return url


# ============================================================
# PASO 4. OBJETIVO MWIS
# ============================================================

class MWISObjective:
    """
    Define el objetivo de optimización.

    Si cada nodo contiene explícitamente:

        weight

    se utiliza ese valor como objetivo principal.

    Si no existe weight, se utiliza una función de fallback:

        2 * passenger_km - empty_km

    passenger_km, empty_km y operating_cost son métricas
    operacionales. No forman parte del objetivo cuando
    existe un weight explícito.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph

        self.weights = {}

        for node, data in graph.nodes(data=True):

            # ------------------------------------------------
            # OBJETIVO EXPLÍCITO
            # ------------------------------------------------

            if "weight" in data:
                weight = float(data["weight"])

            # ------------------------------------------------
            # OBJETIVO DE FALLBACK
            # ------------------------------------------------

            else:
                passenger_km = float(data.get("passenger_km", 0.0))
                empty_km = float(data.get("empty_km", 0.0))

                weight = (
                    2.0 * passenger_km
                    - empty_km
                )

            self.weights[node] = weight

    def total_weight(self, selected_nodes: List[str]) -> float:
        """
        Calcula el peso total de una solución.
        """

        return sum(
            self.weights[node]
            for node in selected_nodes
        )


# ============================================================
# PASO 5. CONSTRUCCIÓN DEL GRAFO
# ============================================================

def build_conflict_graph(input_data: Dict[str, Any]) -> nx.Graph:
    """
    Construye el grafo de conflictos a partir del JSON.

    Input:

        {
            "nodes": [...],
            "edges": [...]
        }

    Cada nodo representa un ciclo.

    Cada arista:

        [C01, C02]

    significa que C01 y C02 no pueden seleccionarse
    simultáneamente.
    """

    graph = nx.Graph()

    nodes = input_data.get("nodes", [])
    edges = input_data.get("edges", [])

    # --------------------------------------------------------
    # Añadir nodos
    # --------------------------------------------------------

    for node_data in nodes:

        if not isinstance(node_data, dict):
            raise ValueError(
                "Cada elemento de 'nodes' debe ser un objeto."
            )

        if "id" not in node_data:
            raise ValueError(
                "Cada nodo debe contener un campo 'id'."
            )

        node_id = str(node_data["id"])

        graph.add_node(
            node_id,
            **node_data
        )

    # --------------------------------------------------------
    # Añadir aristas
    # --------------------------------------------------------

    for edge in edges:

        if not isinstance(edge, (list, tuple)):
            raise ValueError(
                "Cada arista debe ser una lista de dos nodos."
            )

        if len(edge) != 2:
            raise ValueError(
                f"Arista inválida: {edge}. "
                "Debe contener exactamente dos nodos."
            )

        u = str(edge[0])
        v = str(edge[1])

        if u not in graph:
            raise ValueError(
                f"La arista referencia un nodo inexistente: {u}"
            )

        if v not in graph:
            raise ValueError(
                f"La arista referencia un nodo inexistente: {v}"
            )

        graph.add_edge(u, v)

    return graph


# ============================================================
# PASO 6. CONSTRUCCIÓN DEL QUBO
# ============================================================

def calculate_penalty(weights: Dict[str, float]) -> float:
    """
    Calcula el parámetro lambda del QUBO.

    Utilizamos:

        lambda = 4 * max(weight)

    Esto garantiza una penalización suficientemente grande
    respecto al beneficio individual de seleccionar un nodo.

    Para el dataset de validación:

        max(weight) = 91

        lambda = 364
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


def qubo_energy(
    graph: nx.Graph,
    weights: Dict[str, float],
    selected_nodes: List[str],
    penalty: float,
) -> float:
    """
    Evalúa directamente la energía QUBO de una solución.

        H(x) =
            - sum(weight_i * x_i)
            + lambda * sum(x_i*x_j)

    Una solución independiente no tiene términos de penalización.

    Por tanto, para una solución factible:

        H(x) = - total_weight
    """

    selected = set(selected_nodes)

    energy = 0.0

    # --------------------------------------------------------
    # Término objetivo
    # --------------------------------------------------------

    for node in graph.nodes:

        if node in selected:
            energy -= weights[node]

    # --------------------------------------------------------
    # Término de penalización
    # --------------------------------------------------------

    for u, v in graph.edges:

        if u in selected and v in selected:
            energy += penalty

    return float(energy)


# ============================================================
# PASO 7. CONSTRUCCIÓN DEL CIRCUITO QAOA
# ============================================================

def build_qaoa_circuit(
    graph: nx.Graph,
    weights: Dict[str, float],
    penalty: float,
    qaoa_depth: int,
) -> QuantumCircuit:
    """
    Construye un circuito QAOA básico.

    Para cada qubit:

        qubit i <-> nodo i

    Inicialización:
        |+> para todos los qubits.

    Cada capa QAOA contiene:

        1. Cost unitary
        2. Mixer unitary

    NOTA
    ----
    Los parámetros gamma y beta son actualmente valores
    heurísticos/fijados para el PoC.

    No se pretende demostrar que sean parámetros variacionales
    óptimos para cualquier instancia.
    """

    if QuantumCircuit is None:
        raise RuntimeError(
            "Qiskit no está disponible en el entorno de ejecución."
        )

    qaoa_depth = max(
        1,
        int(qaoa_depth)
    )

    nodes = list(graph.nodes)

    n_qubits = len(nodes)

    circuit = QuantumCircuit(
        n_qubits,
        n_qubits
    )

    # --------------------------------------------------------
    # Estado inicial |+>
    # --------------------------------------------------------

    for qubit in range(n_qubits):
        circuit.h(qubit)

    # --------------------------------------------------------
    # Parámetros heurísticos
    # --------------------------------------------------------

    gamma = 0.15
    beta = 0.20

    node_to_qubit = {
        node: index
        for index, node in enumerate(nodes)
    }

    # --------------------------------------------------------
    # Capas QAOA
    # --------------------------------------------------------

    for _ in range(qaoa_depth):

        # ====================================================
        # Cost unitary
        # ====================================================

        # Objetivo:
        #
        #     - weight_i * x_i
        #
        # Para este PoC utilizamos RZ como implementación
        # sencilla del término diagonal.

        for node in nodes:

            qubit = node_to_qubit[node]

            angle = (
                2.0
                * gamma
                * weights[node]
            )

            circuit.rz(
                angle,
                qubit
            )

        # ----------------------------------------------------
        # Penalización por conflictos
        # ----------------------------------------------------

        for u, v in graph.edges:

            qubit_u = node_to_qubit[u]
            qubit_v = node_to_qubit[v]

            # Implementación de una interacción ZZ.
            #
            # CNOT -> RZ -> CNOT

            circuit.cx(
                qubit_u,
                qubit_v
            )

            circuit.rz(
                2.0 * gamma * penalty,
                qubit_v
            )

            circuit.cx(
                qubit_u,
                qubit_v
            )

        # ====================================================
        # Mixer unitary
        # ====================================================

        for qubit in range(n_qubits):

            circuit.rx(
                2.0 * beta,
                qubit
            )

    # --------------------------------------------------------
    # Medición
    # --------------------------------------------------------

    circuit.measure(
        range(n_qubits),
        range(n_qubits)
    )

    return circuit


# ============================================================
# PASO 8. EJECUCIÓN CUÁNTICA
# ============================================================

def execute_qaoa(
    circuit: QuantumCircuit,
    extra_arguments: Dict[str, Any],
) -> Tuple[Dict[str, int], str, Any]:
    """
    Ejecuta el circuito QAOA.

    FUENTE DE CONFIGURACIÓN
    -----------------------

    QCentroid entrega:

        extra_arguments["iqm_token"]
        extra_arguments["quantum_computer"]
        extra_arguments["shots"]

    No se utiliza un token embebido en el código.

    BACKEND
    -------

    Si use_iqm=True, se utiliza obligatoriamente IQM.

    Por defecto:

        use_iqm = True

    Si IQM no está disponible o la autenticación falla,
    se genera una excepción.

    Esto es intencionado.

    No queremos que un benchmark configurado para hardware
    real termine accidentalmente ejecutándose en AerSimulator.
    """

    # ========================================================
    # 8.1. Leer parámetros entregados por QCentroid
    # ========================================================

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

    use_iqm = bool(
        extra_arguments.get(
            "use_iqm",
            True
        )
    )

    # ========================================================
    # 8.2. Token IQM
    # ========================================================

    # IMPORTANTE:
    #
    # QCentroid ya entrega el token mediante:
    #
    #     extra_arguments["iqm_token"]
    #
    # No se imprime nunca el valor.

    iqm_token = extra_arguments.get(
        "iqm_token"
    )

    if use_iqm:

        logger.info(
            "QAOA backend requested: IQM"
        )

        logger.info(
            "IQM quantum computer: %s",
            quantum_computer
        )

        logger.info(
            "QAOA shots: %d",
            shots
        )

        # ====================================================
        # 8.3. Comprobar SDK IQM
        # ====================================================

        if IQMProvider is None:

            raise RuntimeError(
                "IQMProvider no está disponible. "
                "El entorno QCentroid debe tener instalado "
                "qiskit-iqm."
            )

        # ====================================================
        # 8.4. Comprobar credencial
        # ====================================================

        if not iqm_token:

            raise RuntimeError(
                "QCentroid no ha proporcionado 'iqm_token' "
                "en extra_arguments. "
                "No se realizará fallback a AerSimulator."
            )

        # ====================================================
        # 8.5. Resolver servidor IQM
        # ========================================================

        #
        # QCentroid puede proporcionar la URL mediante:
        #
        #   iqm_server_url
        #   iqm_url
        #
        # También permitimos variable de entorno como respaldo
        # técnico.
        #

        iqm_url = (
            extra_arguments.get("iqm_server_url")
            or extra_arguments.get("iqm_url")
            or os.getenv("IQM_SERVER_URL")
        )

        # ----------------------------------------------------
        # En el entorno Resonance utilizado por este PoC,
        # si QCentroid no proporciona explícitamente el endpoint,
        # utilizamos el endpoint público de Resonance.
        #
        # El ordenador cuántico concreto sigue siendo controlado
        # por `quantum_computer`.
        # ----------------------------------------------------

        if not iqm_url:

            iqm_url = (
                "https://cocos.resonance.meetiqm.com"
            )

        iqm_url = (
            URLSanitizer
            .normalize_iqm_url(iqm_url)
        )

        logger.info(
            "IQM server endpoint: %s",
            iqm_url
        )

        # ====================================================
        # 8.6. Crear Provider IQM
        # ====================================================

        try:

            provider = IQMProvider(
                iqm_url,
                quantum_computer=quantum_computer,
                token=iqm_token,
            )

        except TypeError:

            # ------------------------------------------------
            # Compatibilidad con versiones del SDK que puedan
            # manejar la autenticación de forma diferente.
            # ------------------------------------------------

            try:

                provider = IQMProvider(
                    iqm_url,
                    quantum_computer=quantum_computer,
                    token=iqm_token,
                )

            except Exception as exc:

                raise RuntimeError(
                    "No se pudo inicializar IQMProvider: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        except Exception as exc:

            raise RuntimeError(
                "No se pudo inicializar IQMProvider: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # ====================================================
        # 8.7. Obtener backend real
        # ====================================================

        try:

            backend = provider.get_backend()

        except Exception as exc:

            raise RuntimeError(
                "No se pudo obtener el backend IQM. "
                f"Quantum computer solicitado: "
                f"{quantum_computer}. "
                f"Detalle: {type(exc).__name__}: {exc}"
            ) from exc

        logger.info(
            "IQM backend successfully obtained: %s",
            backend
        )

        # ====================================================
        # 8.8. Transpilación para IQM
        # ====================================================

        try:

            transpiled_circuit = transpile(
                circuit,
                backend=backend
            )

        except Exception as exc:

            raise RuntimeError(
                "Error durante la transpilación del circuito "
                "para IQM: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        logger.info(
            "Circuit transpiled successfully for IQM."
        )

        # ====================================================
        # 8.9. Envío a hardware
        # ====================================================

        try:

            job = backend.run(
                transpiled_circuit,
                shots=shots
            )

        except Exception as exc:

            raise RuntimeError(
                "Error enviando el circuito al backend IQM. "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        logger.info(
            "QAOA job submitted to IQM."
        )

        # ====================================================
        # 8.10. Esperar resultado
        # ====================================================

        try:

            result = job.result()

        except Exception as exc:

            raise RuntimeError(
                "Error obteniendo el resultado del job IQM. "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # ====================================================
        # 8.11. Obtener bitstrings
        # ====================================================

        try:

            counts = result.get_counts()

        except Exception as exc:

            raise RuntimeError(
                "No se pudieron obtener los counts del resultado "
                "IQM: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # ====================================================
        # 8.12. Obtener Job ID
        # ====================================================

        job_id = None

        try:

            job_id = job.job_id()

        except Exception:

            try:
                job_id = job.id()

            except Exception:
                job_id = None

        logger.info(
            "IQM job completed. Job ID: %s",
            job_id
        )

        return (
            counts,
            f"IQM Resonance ({quantum_computer})",
            job_id,
        )

    # ========================================================
    # 8.13. Simulador SOLO si se solicita explícitamente
    # ========================================================

    logger.warning(
        "IQM execution explicitly disabled. "
        "Using local AerSimulator."
    )

    if AerSimulator is None:

        raise RuntimeError(
            "AerSimulator no está disponible."
        )

    simulator = AerSimulator()

    transpiled_circuit = transpile(
        circuit,
        simulator
    )

    job = simulator.run(
        transpiled_circuit,
        shots=shots
    )

    result = job.result()

    counts = result.get_counts()

    return (
        counts,
        "Qiskit AerSimulator",
        None,
    )


# ============================================================
# PASO 9. DECODIFICACIÓN DE BITSTRING
# ============================================================

def bitstring_to_selected_nodes(
    bitstring: str,
    nodes: List[str],
) -> List[str]:
    """
    Convierte un bitstring QAOA en nodos seleccionados.

    Qiskit devuelve los bits en orden little-endian respecto
    al registro clásico. Por ello se invierte el string antes
    de asociarlo a la lista de nodos.

    Ejemplo:

        nodes = [C01, C02, C03, C04, C05]

        bitstring = 00101

    Después de invertir:

        10100

    Seleccionaría:

        C01, C03

    """

    clean = (
        str(bitstring)
        .replace(" ", "")
    )

    clean = clean[::-1]

    selected = []

    for index, node in enumerate(nodes):

        if index >= len(clean):
            break

        if clean[index] == "1":
            selected.append(node)

    return selected


# ============================================================
# PASO 10. EVALUAR TODOS LOS BITSTRINGS
# ============================================================

def select_best_qaoa_bitstring(
    counts: Dict[str, int],
    graph: nx.Graph,
    weights: Dict[str, float],
    penalty: float,
) -> Tuple[str, List[str], float]:
    """
    Selecciona el mejor bitstring observado.

    IMPORTANTE
    ----------

    No se selecciona simplemente el bitstring más frecuente.

    Para cada bitstring:

        1. Se convierte a solución.
        2. Se calcula la energía QUBO.
        3. Se compara la energía.
        4. En caso de empate se utilizan los shots.

    El criterio correcto es:

        menor energía QUBO = mejor solución
    """

    nodes = list(graph.nodes)

    best_bitstring = None
    best_nodes = []
    best_energy = float("inf")
    best_shots = -1

    for bitstring, shots_observed in counts.items():

        selected_nodes = bitstring_to_selected_nodes(
            bitstring,
            nodes
        )

        energy = qubo_energy(
            graph,
            weights,
            selected_nodes,
            penalty
        )

        if (
            energy < best_energy
            or (
                math.isclose(
                    energy,
                    best_energy
                )
                and shots_observed > best_shots
            )
        ):

            best_bitstring = bitstring
            best_nodes = selected_nodes
            best_energy = energy
            best_shots = shots_observed

    if best_bitstring is None:

        raise RuntimeError(
            "QAOA no devolvió ningún bitstring."
        )

    return (
        best_bitstring,
        best_nodes,
        float(best_energy),
    )


# ============================================================
# PASO 11. REPARACIÓN DE SOLUCIONES
# ============================================================

class MWISPruner:
    """
    Repara una solución que contenga conflictos.

    Estrategia:

        - Mientras exista un conflicto:
        - identificar los nodos conflictivos
        - eliminar el nodo con peor relación:

              weight / degree

    Es una heurística determinista.

    No sustituye al optimizador.
    Se utiliza como mecanismo de robustez para convertir
    una solución QAOA potencialmente no factible en una
    solución MWIS factible.
    """

    def __init__(
        self,
        graph: nx.Graph,
        weights: Dict[str, float],
    ):
        self.graph = graph
        self.weights = weights

    def prune_solution(
        self,
        selected_nodes: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Devuelve:

            solución reparada
            nodos eliminados
        """

        selected = set(selected_nodes)

        pruned_nodes = []

        while True:

            conflicts = []

            for u, v in self.graph.edges:

                if (
                    u in selected
                    and v in selected
                ):

                    conflicts.append(
                        (u, v)
                    )

            if not conflicts:
                break

            conflicting_nodes = set()

            for u, v in conflicts:

                conflicting_nodes.add(u)
                conflicting_nodes.add(v)

            # ------------------------------------------------
            # Calcular score weight / degree
            # ------------------------------------------------

            candidate = None
            candidate_score = float("inf")

            for node in conflicting_nodes:

                degree = max(
                    1,
                    self.graph.degree(node)
                )

                score = (
                    self.weights[node]
                    / degree
                )

                if score < candidate_score:

                    candidate = node
                    candidate_score = score

            if candidate is None:
                break

            selected.remove(candidate)

            pruned_nodes.append(candidate)

        return (
            list(selected),
            pruned_nodes,
        )


# ============================================================
# PASO 12. SOLUCIÓN EXACTA CLÁSICA
# ============================================================

def calculate_exact_ground_truth(
    graph: nx.Graph,
    weights: Dict[str, float],
) -> Tuple[List[str], float]:
    """
    Calcula la solución MWIS exacta mediante:

        MWIS(G) = Maximum Weight Clique(complement(G))

    Se utiliza NetworkX.

    NOTA
    ----

    Esta referencia exacta está pensada para instancias pequeñas
    de validación y benchmarking.

    No debe interpretarse como un método escalable a grandes
    instancias ferroviarias.
    """

    if len(graph) == 0:

        return [], 0.0

    complement = nx.complement(graph)

    # --------------------------------------------------------
    # NetworkX max_weight_clique utiliza pesos enteros.
    # --------------------------------------------------------

    for node in complement.nodes:

        complement.nodes[node]["weight"] = int(
            round(
                weights[node]
            )
        )

    clique, clique_weight = nx.max_weight_clique(
        complement,
        weight="weight"
    )

    selected_nodes = list(clique)

    # --------------------------------------------------------
    # Recalcular utilizando los pesos float originales.
    # --------------------------------------------------------

    total_weight = sum(
        weights[node]
        for node in selected_nodes
    )

    return (
        selected_nodes,
        float(total_weight),
    )


# ============================================================
# PASO 13. MÉTRICAS OPERACIONALES
# ============================================================

def calculate_operational_metrics(
    graph: nx.Graph,
    selected_nodes: List[str],
) -> Dict[str, float]:
    """
    Calcula métricas operacionales de la solución seleccionada.

    Estas métricas NO modifican el objetivo MWIS cuando existe
    un campo `weight` explícito.

    Se utilizan para interpretar la solución desde el punto
    de vista ferroviario.
    """

    selected = set(selected_nodes)

    passenger_km = 0.0
    empty_km = 0.0
    operating_cost = 0.0

    covered_trips = set()

    for node in selected:

        data = graph.nodes[node]

        passenger_km += float(
            data.get(
                "passenger_km",
                0.0
            )
        )

        empty_km += float(
            data.get(
                "empty_km",
                0.0
            )
        )

        operating_cost += float(
            data.get(
                "operating_cost",
                0.0
            )
        )

        for trip in data.get(
            "trips",
            []
        ):

            covered_trips.add(
                str(trip)
            )

    return {
        "total_passenger_km": float(
            passenger_km
        ),
        "total_empty_km": float(
            empty_km
        ),
        "total_operating_cost": float(
            operating_cost
        ),
        "covered_trips": len(
            covered_trips
        ),
        "_covered_trip_ids": covered_trips,
    }


# ============================================================
# PASO 14. VISUALIZACIÓN
# ============================================================

class VisualizationAssetGenerator:
    """
    Genera un PNG del grafo de conflictos.

    QCentroid recoge automáticamente los archivos generados
    dentro de:

        additional_output/

    El archivo no tiene que incluirse dentro del JSON.
    """

    @staticmethod
    def generate_conflict_graph(
        graph: nx.Graph,
        selected_nodes: List[str],
    ) -> str:

        # ----------------------------------------------------
        # Directorio de assets de QCentroid
        # ----------------------------------------------------

        output_dir = (
            "additional_output"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        path = os.path.join(
            output_dir,
            "conflict_graph.png"
        )

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        plt.figure(
            figsize=(9, 7),
            dpi=150
        )

        pos = nx.spring_layout(
            graph,
            seed=42
        )

        # ----------------------------------------------------
        # Colores
        # ----------------------------------------------------
        # Se utiliza la API estándar de NetworkX.
        # Los nodos seleccionados se diferencian visualmente.
        # ----------------------------------------------------

        node_colors = []

        selected = set(
            selected_nodes
        )

        for node in graph.nodes:

            if node in selected:
                node_colors.append(
                    "green"
                )
            else:
                node_colors.append(
                    "lightgray"
                )

        nx.draw_networkx_edges(
            graph,
            pos,
            alpha=0.6
        )

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=900
        )

        nx.draw_networkx_labels(
            graph,
            pos,
            font_weight="bold"
        )

        plt.title(
            "Railway Rolling Stock Conflict Graph"
        )

        plt.axis("off")

        plt.savefig(
            path,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            "Conflict graph visualization saved to %s",
            path
        )

        return path


# ============================================================
# PASO 15. VALIDACIÓN DE INPUT
# ============================================================

def validate_input_data(
    input_data: Dict[str, Any],
) -> None:
    """
    Comprueba que el dataset contiene la estructura mínima
    necesaria para ejecutar el solver.
    """

    if not isinstance(
        input_data,
        dict
    ):

        raise ValueError(
            "Input data debe ser un objeto JSON."
        )

    if "nodes" not in input_data:

        raise ValueError(
            "Input data debe contener 'nodes'."
        )

    if "edges" not in input_data:

        raise ValueError(
            "Input data debe contener 'edges'."
        )

    if not isinstance(
        input_data["nodes"],
        list
    ):

        raise ValueError(
            "'nodes' debe ser una lista."
        )

    if not isinstance(
        input_data["edges"],
        list
    ):

        raise ValueError(
            "'edges' debe ser una lista."
        )


# ============================================================
# PASO 16. FUNCIÓN PRINCIPAL QCentroid
# ============================================================

def run(
    input_data: Dict[str, Any],
    solver_params: Dict[str, Any],
    extra_arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Punto de entrada principal del solver para QCentroid.

    QCentroid llama a:

        run(
            input_data,
            solver_params,
            extra_arguments
        )

    La función devuelve un diccionario JSON-compatible.
    """

    # ========================================================
    # 16.1. Inicio del cronómetro
    # ========================================================

    execution_start = time.perf_counter()

    logger.info(
        "==================================================="
    )

    logger.info(
        "Railway MWIS + QAOA solver started."
    )

    logger.info(
        "==================================================="
    )

    # ========================================================
    # 16.2. Validación del input
    # ========================================================

    validate_input_data(
        input_data
    )

    # ========================================================
    # 16.3. Leer parámetros QCentroid
    # ========================================================

    #
    # QCentroid pasa:
    #
    #   iqm_token
    #   quantum_computer
    #   shots
    #   qaoa_depth
    #

    shots = int(
        extra_arguments.get(
            "shots",
            2048
        )
    )

    qaoa_depth = int(
        extra_arguments.get(
            "qaoa_depth",
            2
        )
    )

    quantum_computer = str(
        extra_arguments.get(
            "quantum_computer",
            "emerald"
        )
    )

    logger.info(
        "QCentroid parameters received:"
    )

    logger.info(
        "  quantum_computer = %s",
        quantum_computer
    )

    logger.info(
        "  shots = %d",
        shots
    )

    logger.info(
        "  qaoa_depth = %d",
        qaoa_depth
    )

    # --------------------------------------------------------
    # No mostramos nunca el token.
    # --------------------------------------------------------

    logger.info(
        "  iqm_token = [PROVIDED]"
        if extra_arguments.get("iqm_token")
        else
        "  iqm_token = [NOT PROVIDED]"
    )

    # ========================================================
    # 16.4. Construir grafo
    # ========================================================

    graph = build_conflict_graph(
        input_data
    )

    logger.info(
        "Conflict graph created:"
    )

    logger.info(
        "  nodes = %d",
        graph.number_of_nodes()
    )

    logger.info(
        "  conflicts = %d",
        graph.number_of_edges()
    )

    # ========================================================
    # 16.5. Crear objetivo MWIS
    # ========================================================

    objective = MWISObjective(
        graph
    )

    weights = objective.weights

    logger.info(
        "Optimization objective: "
        "maximize sum(weight_i * x_i)"
    )

    # ========================================================
    # 16.6. Calcular penalización QUBO
    # ========================================================

    penalty = calculate_penalty(
        weights
    )

    logger.info(
        "QUBO penalty lambda = %.6f",
        penalty
    )

    # ========================================================
    # 16.7. Calcular ground truth exacto
    # ========================================================

    logger.info(
        "Calculating exact classical ground truth..."
    )

    exact_nodes, exact_weight = (
        calculate_exact_ground_truth(
            graph,
            weights
        )
    )

    logger.info(
        "Exact solution = %s",
        exact_nodes
    )

    logger.info(
        "Exact total weight = %.6f",
        exact_weight
    )

    # ========================================================
    # 16.8. Construir circuito QAOA
    # ========================================================

    logger.info(
        "Building QAOA circuit..."
    )

    circuit = build_qaoa_circuit(
        graph,
        weights,
        penalty,
        qaoa_depth
    )

    logger.info(
        "QAOA circuit created:"
    )

    logger.info(
        "  qubits = %d",
        circuit.num_qubits
    )

    logger.info(
        "  depth = %s",
        circuit.depth()
    )

    # ========================================================
    # 16.9. Ejecutar QAOA
    # ========================================================

    logger.info(
        "Executing QAOA..."
    )

    counts, backend_used, job_id = (
        execute_qaoa(
            circuit,
            extra_arguments
        )
    )

    logger.info(
        "QAOA backend used: %s",
        backend_used
    )

    logger.info(
        "QAOA job ID: %s",
        job_id
    )

    logger.info(
        "Number of measured bitstrings: %d",
        len(counts)
    )

    # ========================================================
    # 16.10. Seleccionar mejor bitstring
    # ========================================================

    (
        best_bitstring,
        initial_solution,
        initial_energy,
    ) = select_best_qaoa_bitstring(
        counts,
        graph,
        weights,
        penalty
    )

    logger.info(
        "Best observed bitstring: %s",
        best_bitstring
    )

    logger.info(
        "Initial QUBO energy: %.6f",
        initial_energy
    )

    logger.info(
        "Initial selected cycles: %s",
        initial_solution
    )

    # ========================================================
    # 16.11. Reparar solución si presenta conflictos
    # ========================================================

    pruner = MWISPruner(
        graph,
        weights
    )

    (
        selected_nodes,
        pruned_nodes,
    ) = pruner.prune_solution(
        initial_solution
    )

    if pruned_nodes:

        logger.info(
            "Pruning removed conflicting nodes: %s",
            pruned_nodes
        )

    else:

        logger.info(
            "No pruning required."
        )

    # ========================================================
    # 16.12. Comprobar factibilidad
    # ========================================================

    is_feasible = nx.is_independent_set(
        graph,
        selected_nodes
    )

    if not is_feasible:

        raise RuntimeError(
            "La solución final no es un independent set válido."
        )

    # ========================================================
    # 16.13. Peso final
    # ========================================================

    total_weight = objective.total_weight(
        selected_nodes
    )

    # ========================================================
    # 16.14. Métricas operacionales
    # ========================================================

    operational = (
        calculate_operational_metrics(
            graph,
            selected_nodes
        )
    )

    # ========================================================
    # 16.15. Cobertura de viajes
    # ========================================================

    all_trips = set()

    for node in graph.nodes:

        for trip in graph.nodes[node].get(
            "trips",
            []
        ):

            all_trips.add(
                str(trip)
            )

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

    else:

        coverage_rate = 0.0

    coverage_percent = (
        100.0
        * coverage_rate
    )

    # ========================================================
    # 16.16. Gap de optimalidad
    # ========================================================

    if exact_weight != 0:

        optimality_gap_percent = (
            100.0
            * (
                exact_weight
                - total_weight
            )
            / abs(exact_weight)
        )

    else:

        optimality_gap_percent = 0.0

    # Evitamos pequeños errores numéricos negativos.

    if abs(
        optimality_gap_percent
    ) < 1e-12:

        optimality_gap_percent = 0.0

    # ========================================================
    # 16.17. Generar asset gráfico
    # ========================================================

    conflict_graph_path = (
        VisualizationAssetGenerator
        .generate_conflict_graph(
            graph,
            selected_nodes
        )
    )

    # ========================================================
    # 16.18. Tiempo de ejecución
    # ========================================================

    execution_time_seconds = (
        time.perf_counter()
        - execution_start
    )

    # ========================================================
    # 16.19. Estadísticas de pruning
    # ========================================================

    pruning_stats = {
        "nodes_pruned_count": len(
            pruned_nodes
        ),
        "pruned_nodes_list": list(
            pruned_nodes
        ),
    }

    # ========================================================
    # 16.20. Resultado final
    # ========================================================

    #
    # IMPORTANTE PARA QCentroid
    #
    # Las cinco métricas de benchmark están en el nivel raíz:
    #
    #   total_weight
    #   optimality_gap_percent
    #   coverage_percent
    #   execution_time_seconds
    #   total_operating_cost
    #
    # Esto permite que QCentroid las utilice directamente
    # como Output Benchmark Metrics.
    #

    result = {

        # ----------------------------------------------------
        # BENCHMARK METRICS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SOLUCIÓN
        # ----------------------------------------------------

        "selected_cycles": list(
            selected_nodes
        ),

        "is_feasible": bool(
            is_feasible
        ),

        # ----------------------------------------------------
        # COBERTURA
        # ----------------------------------------------------

        "scheduled_trips": int(
            scheduled_trips
        ),

        "covered_trips": int(
            covered_trips
        ),

        "coverage_rate": float(
            coverage_rate
        ),

        # ----------------------------------------------------
        # MÉTRICAS OPERACIONALES
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
        # BACKEND
        # ----------------------------------------------------

        "backend_used": str(
            backend_used
        ),

        "job_id": job_id,

        # ----------------------------------------------------
        # QAOA
        # ----------------------------------------------------

        "initial_bitstring_min_qubo": str(
            best_bitstring
        ),

        "initial_qubo_energy": float(
            initial_energy
        ),

        "initial_solution_size": int(
            len(initial_solution)
        ),

        "final_solution_size": int(
            len(selected_nodes)
        ),

        "shots": int(
            shots
        ),

        "qaoa_depth": int(
            qaoa_depth
        ),

        "penalty": float(
            penalty
        ),

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        "ground_truth_exact": {

            "selected_cycles": list(
                exact_nodes
            ),

            "total_weight": float(
                exact_weight
            ),

            "optimality_gap_percent": float(
                optimality_gap_percent
            ),
        },

        # ----------------------------------------------------
        # PRUNING
        # ----------------------------------------------------

        "nodes_pruned": int(
            len(pruned_nodes)
        ),

        "pruning_stats": pruning_stats,

        # ----------------------------------------------------
        # ASSETS
        # ----------------------------------------------------

        "assets": {

            "conflict_graph_png": (
                conflict_graph_path
            )

        },

        # ----------------------------------------------------
        # DETALLE DE EJECUCIÓN
        # ----------------------------------------------------

        "execution_metrics": {

            "execution_time_seconds": float(
                execution_time_seconds
            ),

            "final_solution_size": int(
                len(selected_nodes)
            ),

            "initial_bitstring_min_qubo": str(
                best_bitstring
            ),

            "initial_qubo_energy": float(
                initial_energy
            ),

            "initial_solution_size": int(
                len(initial_solution)
            ),

            "job_id": job_id,

            "penalty": float(
                penalty
            ),

            "qaoa_depth": int(
                qaoa_depth
            ),

            "shots": int(
                shots
            ),
        },
    }

    # ========================================================
    # 16.21. Logging final
    # ========================================================

    logger.info(
        "==================================================="
    )

    logger.info(
        "FINAL SOLUTION"
    )

    logger.info(
        "Selected cycles: %s",
        selected_nodes
    )

    logger.info(
        "Total weight: %.6f",
        total_weight
    )

    logger.info(
        "Coverage: %.2f%%",
        coverage_percent
    )

    logger.info(
        "Operating cost: %.2f",
        operational[
            "total_operating_cost"
        ]
    )

    logger.info(
        "Optimality gap: %.6f%%",
        optimality_gap_percent
    )

    logger.info(
        "Backend: %s",
        backend_used
    )

    logger.info(
        "Job ID: %s",
        job_id
    )

    logger.info(
        "Execution time: %.6f seconds",
        execution_time_seconds
    )

    logger.info(
        "==================================================="
    )

    # ========================================================
    # 16.22. Devolver resultado a QCentroid
    # ========================================================

    return result
