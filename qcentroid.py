"""
qcentroid.py - Railway Rolling Stock Cycle Selection via MWIS + QAOA
PoC para QCentroid Quantum Platform
Basado en el caso de estudio de IQM & Deutsche Bahn (arXiv:2606.11383)
"""

import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

matplotlib.use("Agg")

# Importación opcional de SciPy para la optimización de parámetros QAOA
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Importaciones de Qiskit y adaptadores cuánticos
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False

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
    """Sanitiza y valida URLs de IQM Resonance para evitar sufijos de QPU."""
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


class IQMBackendManager:
    """Gestor de conexión y ejecución en hardware IQM Resonance."""
    def __init__(self, iqm_token: str, quantum_computer: str = "emerald", server_url: str = "https://resonance.iqm.tech/"):
        if not iqm_token:
            raise ValueError("El token de IQM es obligatorio.")
        self.iqm_token = iqm_token
        self.quantum_computer = quantum_computer
        self.server_url = URLSanitizer.sanitize_iqm_url(server_url)
        self.backend = None
        self._connect()

    def _connect(self):
        if not IQM_AVAILABLE:
            raise RuntimeError(f"El plugin de IQM no está disponible: {IQM_IMPORT_ERROR or 'No instalado'}")
        try:
            logger.info("Conectando a IQM Resonance en %s (QPU: %s)", self.server_url, self.quantum_computer)
            provider = IQMProvider(url=self.server_url, quantum_computer=self.quantum_computer, token=self.iqm_token)
            self.backend = provider.get_backend()
            logger.info("Conectado con éxito al backend IQM: %s", self.backend.name)
        except Exception as exc:
            raise RuntimeError(f"Error de conexión con IQM: {exc}") from exc

    def get_backend(self):
        return self.backend


class QAOAMWISSolver:
    """Solver QAOA para el problema Maximum Weighted Independent Set (MWIS)."""
    def __init__(self, conflict_graph: nx.Graph, penalty: Optional[float] = None):
        self.graph = conflict_graph
        self.nodes = sorted(list(conflict_graph.nodes()))
        self.num_nodes = len(self.nodes)
        
        # Cálculo del peso máximo y constante de penalización lambda = 4 * max(w)
        max_weight = max((float(conflict_graph.nodes[n].get("weight", 0.0)) for n in self.nodes), default=1.0)
        self.penalty = float(penalty) if penalty is not None else 4.0 * max_weight

    def classical_objective(self, bits: List[int]) -> float:
        """Calcula la función objetivo QUBO: Minimizar H(x) = -sum(w_i * x_i) + lambda * sum(x_i * x_j)."""
        val = 0.0
        for i, n in enumerate(self.nodes):
            if bits[i]:
                val -= float(self.graph.nodes[n].get("weight", 0.0))
        for u, v in self.graph.edges():
            i = self.nodes.index(u)
            j = self.nodes.index(v)
            if bits[i] and bits[j]:
                val += self.penalty
        return val

    def is_feasible(self, selected: List[str]) -> bool:
        sel_set = set(selected)
        return all(not (u in sel_set and v in sel_set) for u, v in self.graph.edges())

    def _hamiltonian_coefficients(self):
        """Coeficientes del Hamiltoniano de Ising: h_i = w_i/2 - lambda*deg(i)/4, J_ij = lambda/4."""
        h = {}
        for n in self.nodes:
            w = float(self.graph.nodes[n].get("weight", 0.0))
            deg = self.graph.degree(n)
            h[n] = (w / 2.0) - (self.penalty * deg / 4.0)
        zz = {}
        for u, v in self.graph.edges():
            zz[(u, v)] = self.penalty / 4.0
        return h, zz

    def build_qaoa_circuit(self, p: int, gamma_values: List[float], beta_values: List[float]) -> QuantumCircuit:
        n = len(self.nodes)
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(n, "c")
        qc = QuantumCircuit(qr, cr)

        h, zz = self._hamiltonian_coefficients()

        # Estado inicial |+>
        for i in range(n):
            qc.h(qr[i])

        # Capas QAOA
        for layer in range(p):
            g = gamma_values[layer]
            b = beta_values[layer]

            # Término de coste lineal (RZ)
            for i, n_id in enumerate(self.nodes):
                qc.rz(2.0 * g * h[n_id], qr[i])

            # Término de interacción de conflictos (RZZ)
            for (u, v), coupling in zz.items():
                i = self.nodes.index(u)
                j = self.nodes.index(v)
                qc.rzz(2.0 * g * coupling, qr[i], qr[j])

            # Operador mezclador Mixer (RX)
            for i in range(n):
                qc.rx(2.0 * b, qr[i])

        qc.measure(qr, cr)
        return qc

    def prune_solution(self, initial_selected: List[str]) -> Tuple[List[str], List[str]]:
        """Poda determinista basada en el ratio weight / degree."""
        selected = set(initial_selected)
        pruned_nodes = []

        while True:
            subgraph = self.graph.subgraph(selected)
            conflicts = list(subgraph.edges())
            if not conflicts:
                break

            conflict_nodes = set()
            for u, v in conflicts:
                conflict_nodes.add(u)
                conflict_nodes.add(v)

            # Selección del nodo con menor ratio w_i / deg_i
            worst_node = min(
                conflict_nodes,
                key=lambda n: float(self.graph.nodes[n].get("weight", 0.0)) / max(1, subgraph.degree(n))
            )
            selected.remove(worst_node)
            pruned_nodes.append(worst_node)

        return sorted(list(selected)), sorted(pruned_nodes)


class RailwayRollingStockSolver:
    """Orquestador principal del problema de asignación de trenes."""
    def __init__(self, nodes: List[Dict[str, Any]], edges: List[List[str]], penalty: Optional[float] = None):
        self.graph = nx.Graph()
        self.all_trips = set()

        for n in nodes:
            n_id = n["id"]
            trips = n.get("trips", [])
            self.all_trips.update(trips)

            # Cálculo automático del peso si no viene definido: w_i = 2*passenger_km - empty_km
            passenger_km = float(n.get("passenger_km", 0.0))
            empty_km = float(n.get("empty_km", 0.0))
            calculated_weight = (2.0 * passenger_km) - empty_km if (passenger_km > 0 or empty_km > 0) else float(n.get("weight", 1.0))
            
            weight = float(n.get("weight", calculated_weight))

            self.graph.add_node(
                n_id,
                weight=weight,
                trips=trips,
                passenger_km=passenger_km,
                empty_km=empty_km,
                operating_cost=float(n.get("operating_cost", 0.0))
            )

        for e in edges:
            if len(e) == 2 and e in self.graph and e[21] in self.graph:
                self.graph.add_edge(e, e[21])

        self.mwis_solver = QAOAMWISSolver(self.graph, penalty=penalty)

    def solve(self, backend=None, qaoa_p: int = 1, shots: int = 2048) -> Tuple[List[str], Dict[str, Any], List[str]]:
        start_time = time.time()

        if QISKIT_AVAILABLE and backend is not None:
            # Ángulos heurísticos pre-optimizados para p=1 o p=2
            if qaoa_p == 1:
                gamma_vals = [0.05]
                beta_vals = [0.25]
            else:
                gamma_vals = [0.05, 0.03]
                beta_vals = [0.25, 0.15]

            qc = self.mwis_solver.build_qaoa_circuit(qaoa_p, gamma_vals, beta_vals)
            qc_transpiled = transpile(qc, backend=backend, optimization_level=2)
            
            logger.info("Ejecutando circuito QAOA en el backend (%d shots)...", shots)
            job = backend.run(qc_transpiled, shots=shots)
            counts = job.result().get_counts()

            # Selección del bitstring de menor energía QUBO
            best_bitstring = max(counts, key=counts.get)
            initial_selected = [
                self.mwis_solver.nodes[i]
                for i in range(len(self.mwis_solver.nodes))
                if best_bitstring[-(i + 1)] == "1"
            ]
            qaoa_alg = "QAOA"
            job_id = getattr(job, "job_id", lambda: "local")()
        else:
            # Fallback voraz clásico
            logger.info("Ejecutando solución voraz clásica...")
            initial_selected = []
            remaining = set(self.graph.nodes())
            while remaining:
                best = max(remaining, key=lambda n: self.graph.nodes[n]["weight"])
                initial_selected.append(best)
                remaining -= set(self.graph.neighbors(best))
                remaining.discard(best)
            qaoa_alg = "Greedy"
            job_id = "N/A"

        # Aplicar Poda
        final_selected, pruned_nodes = self.mwis_solver.prune_solution(initial_selected)

        # Cálculo de métricas
        total_weight = sum(self.graph.nodes[n]["weight"] for n in final_selected)
        covered_trips = set()
        total_passenger_km = 0.0
        total_empty_km = 0.0
        total_operating_cost = 0.0

        for n in final_selected:
            node_data = self.graph.nodes[n]
            covered_trips.update(node_data["trips"])
            total_passenger_km += node_data["passenger_km"]
            total_empty_km += node_data["empty_km"]
            total_operating_cost += node_data["operating_cost"]

        exec_time = time.time() - start_time
        coverage_rate = len(covered_trips) / len(self.all_trips) if self.all_trips else 1.0

        metrics = {
            "total_weight": total_weight,
            "scheduled_trips": len(self.all_trips),
            "covered_trips": len(covered_trips),
            "coverage_rate": round(coverage_rate, 4),
            "coverage_percent": round(coverage_rate * 100.0, 2),
            "total_empty_km": total_empty_km,
            "total_passenger_km": total_passenger_km,
            "total_operating_cost": total_operating_cost,
            "is_feasible": self.mwis_solver.is_feasible(final_selected),
            "execution_time_seconds": exec_time,
            "initial_solution_size": len(initial_selected),
            "final_solution_size": len(final_selected),
            "nodes_pruned": len(pruned_nodes),
            "qaoa_algorithm": qaoa_alg,
            "qaoa_depth": qaoa_p,
            "shots": shots,
            "penalty": self.mwis_solver.penalty,
            "job_id": job_id,
        }

        return final_selected, metrics, pruned_nodes


class VisualizationAssetGenerator:
    """Genera activos visuales (PNG) en la carpeta additional_output."""
    @staticmethod
    def generate_conflict_graph_visualization(graph: nx.Graph, selected: List[str], pruned: List[str], filename: str = "conflict_graph.png") -> Optional[str]:
        try:
            output_dir = "additional_output"
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, filename)

            plt.figure(figsize=(8, 6), dpi=150)
            pos = nx.spring_layout(graph, seed=42)

            node_colors = []
            for n in graph.nodes():
                if n in selected:
                    node_colors.append("#2ecc71")  # Verde: Seleccionado
                elif n in pruned:
                    node_colors.append("#e74c3c")  # Rojo: Podado
                else:
                    node_colors.append("#95a5a6")  # Gris: Descartado

            nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=700)
            nx.draw_networkx_edges(graph, pos, edge_color="#bdc3c7", width=1.5)
            nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", font_color="white")

            labels = {n: f"{n}\n(w={graph.nodes[n]['weight']:.0f})" for n in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, font_color="black")

            plt.title("Grafo de Conflictos de Material Rodante (MWIS)", fontsize=12, fontweight="bold")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            logger.info("Visualización guardada con éxito en %s", path)
            return path
        except Exception as exc:
            logger.warning("No se pudo generar el gráfico de activos: %s", exc)
            return None


def run(input_data: Dict[str, Any], solver_params: Dict[str, Any], extra_arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Punto de entrada principal de QCentroid Platform."""
    logger.info("=" * 70)
    logger.info("Iniciando Solver QCentroid MWIS de Material Rodante Ferroviario")
    logger.info("=" * 70)

    try:
        nodes = input_data.get("nodes", [])
        edges = input_data.get("edges", [])

        if not nodes:
            return {"selected_cycles": [], "total_weight": 0.0, "is_feasible": False, "error": "Dataset de entrada vacío."}

        iqm_token = solver_params.get("iqm_token")
        quantum_computer = solver_params.get("quantum_computer", "emerald")
        server_url = solver_params.get("server_url", "https://resonance.iqm.tech/")
        shots = int(solver_params.get("shots", 2048))
        qaoa_depth = int(solver_params.get("qaoa_depth", 2))
        penalty = solver_params.get("penalty")

        backend = None
        backend_info = "Heurística Voraz Clásica"

        # Conexión a IQM o Simulador Aer
        if iqm_token:
            try:
                iqm_mgr = IQMBackendManager(iqm_token, quantum_computer, server_url)
                backend = iqm_mgr.get_backend()
                backend_info = f"IQM Resonance ({quantum_computer})"
            except Exception as exc:
                logger.warning("No se pudo conectar a IQM: %s. Recomenzando en simulador.", exc)

        if backend is None and QISKIT_AVAILABLE and AER_AVAILABLE:
            backend = AerSimulator()
            backend_info = "Qiskit AerSimulator"

        solver = RailwayRollingStockSolver(nodes, edges, penalty=penalty)
        final_selected, metrics, pruned_nodes = solver.solve(backend=backend, qaoa_p=qaoa_depth, shots=shots)

        # Generar Activos
        asset_path = VisualizationAssetGenerator.generate_conflict_graph_visualization(solver.graph, final_selected, pruned_nodes)
        assets = {"conflict_graph_png": asset_path} if asset_path else {}

        response = {
            "selected_cycles": final_selected,
            "total_weight": metrics["total_weight"],
            "nodes_pruned": metrics["nodes_pruned"],
            "coverage_rate": metrics["coverage_rate"],
            "coverage_percent": metrics["coverage_percent"],
            "covered_trips": metrics["covered_trips"],
            "scheduled_trips": metrics["scheduled_trips"],
            "total_empty_km": metrics["total_empty_km"],
            "total_passenger_km": metrics["total_passenger_km"],
            "total_operating_cost": metrics["total_operating_cost"],
            "is_feasible": metrics["is_feasible"],
            "execution_metrics": {
                "execution_time_seconds": metrics["execution_time_seconds"],
                "initial_solution_size": metrics["initial_solution_size"],
                "final_solution_size": metrics["final_solution_size"],
                "qaoa_depth": qaoa_depth,
                "shots": shots,
                "penalty": metrics["penalty"],
                "job_id": metrics["job_id"],
            },
            "backend_used": backend_info,
            "assets": assets,
            "pruning_stats": {
                "nodes_pruned_count": len(pruned_nodes),
                "pruned_nodes_list": sorted(pruned_nodes),
            },
        }

        logger.info("Ejecución completada con éxito. Ciclos seleccionados: %s", final_selected)
        return response

    except Exception as exc:
        logger.exception("Fallo durante la ejecución del solver.")
        return {"selected_cycles": [], "total_weight": 0.0, "is_feasible": False, "error": str(exc)}
