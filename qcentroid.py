"""
qcentroid.py
Railway Rolling Stock Cycle Selection via Maximum Weighted Independent Set (MWIS)

QCentroid / IQM Proof of Concept
--------------------------------
Input:
{
  "nodes": [
    {"id": "C01", "weight": 85.0, "trips": ["T01", "T02"]}
  ],
  "edges": [
    ["C01", "C02"]
  ]
}

The implementation:
1. Validates the MWIS graph.
2. Builds a mathematically consistent MWIS cost Hamiltonian.
3. Optimizes QAOA parameters on a classical simulator when available.
4. Executes the optimized circuit on IQM Resonance when a token is supplied.
5. Selects the most frequent measured bitstring.
6. Applies deterministic feasibility pruning.
7. Calculates actual trip coverage, selected weight and empty-km metrics when
   optional node attributes are supplied.
8. Produces a conflict-graph PNG asset.

Notes:
- QAOA parameters are optimized classically; IQM is used for the final circuit
  execution. This avoids trying to optimize continuous parameters directly on
  NISQ hardware for this PoC.
- MWIS is encoded as a minimization problem:
      H(x) = -sum_i w_i x_i + P sum_(i,j in E) x_i x_j
  where x_i in {0,1}.
- The conflict penalty P is chosen strictly larger than the largest node
  weight by default, so selecting both endpoints of an edge is never
  beneficial in the classical objective.

Changelog vs. the original PoC:
- FIX: `nx.is_independent(...)` does not exist in NetworkX's public API and
  raised AttributeError at runtime. Replaced with a direct, dependency-free
  independence check equivalent to `QAOAMWISSolver.is_feasible`.
- FIX: `iqm.qiskit_iqm` now raises RuntimeError (not ImportError) when the
  obsolete `qiskit-iqm` distribution is installed instead of the current
  `iqm-client[qiskit]` package. The import fallback below now catches both,
  so the solver degrades gracefully to AerSimulator / greedy instead of
  crashing.
"""

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Set, Optional

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Quantum imports with fallback
#
# NOTE: `qiskit-iqm` is obsolete. The current package is `iqm-client[qiskit]`,
# but it still exposes the same `iqm.qiskit_iqm` import path. If the obsolete
# `qiskit-iqm` distribution happens to be installed, importing it raises a
# RuntimeError (not an ImportError) telling you to migrate. We catch both so
# this module never crashes on import regardless of which package is present.
IQM_IMPORT_ERROR: Optional[str] = None
try:
    from iqm.qiskit_iqm import IQMProvider
    IQM_AVAILABLE = True
except ImportError:
    IQM_AVAILABLE = False
except RuntimeError as _iqm_exc:
    IQM_AVAILABLE = False
    IQM_IMPORT_ERROR = str(_iqm_exc)

try:
    from qiskit import (
        QuantumCircuit,
        QuantumRegister,
        ClassicalRegister,
        transpile,
    )
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False


logger = logging.getLogger("qcentroid-user-log")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

if IQM_IMPORT_ERROR:
    logger.warning(
        "El paquete 'qiskit-iqm' instalado está obsoleto y no se pudo importar "
        "(%s). Instala el sustituto oficial con: "
        "pip uninstall -y qiskit-iqm && pip install --force-reinstall \"iqm-client[qiskit]\". "
        "El solver seguirá funcionando con AerSimulator/heurística voraz mientras tanto.",
        IQM_IMPORT_ERROR,
    )


@dataclass
class CycleNode:
    """Represents a railway rolling-stock cycle candidate."""
    id: str
    weight: float
    trips: Optional[List[str]] = None
    empty_km: Optional[float] = None
    passenger_km: Optional[float] = None
    operating_cost: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConflictEdge:
    """Represents a conflict between two cycles."""
    cycle_1: str
    cycle_2: str


class URLSanitizer:
    """Sanitize and validate IQM Resonance base URLs."""

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

    @staticmethod
    def validate_iqm_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            path = parsed.path.strip("/").lower()
            return not any(qpu in path.split("/") for qpu in URLSanitizer.KNOWN_QPUS)
        except Exception:
            return False


class IQMBackendManager:
    """Manages IQM Resonance connection and circuit execution."""

    def __init__(
        self,
        iqm_token: str,
        quantum_computer: str = "emerald",
        server_url: str = "https://resonance.iqm.tech/",
    ):
        if not iqm_token:
            raise ValueError("IQM token is required.")

        self.iqm_token = iqm_token
        self.quantum_computer = quantum_computer
        self.server_url = URLSanitizer.sanitize_iqm_url(server_url)

        if not URLSanitizer.validate_iqm_url(self.server_url):
            raise ValueError(f"Invalid IQM base URL: {self.server_url}")

        self.backend = None
        self._connect()

    def _connect(self):
        if not IQM_AVAILABLE:
            raise RuntimeError(
                "IQM Qiskit plugin is not installed or is obsolete. "
                "Install the current package with: pip install \"iqm-client[qiskit]\" "
                "(uninstall the obsolete 'qiskit-iqm' first if present)."
            )

        try:
            logger.info("Connecting to IQM Resonance at %s", self.server_url)
            logger.info("Quantum computer: %s", self.quantum_computer)

            provider = IQMProvider(
                url=self.server_url,
                quantum_computer=self.quantum_computer,
                token=self.iqm_token,
            )
            self.backend = provider.get_backend()

            logger.info("Connected to IQM backend: %s", self.backend.name)
            if hasattr(self.backend, "num_qubits"):
                logger.info("Backend qubits: %s", self.backend.num_qubits)

        except Exception as exc:
            raise RuntimeError(f"IQM connection failed: {exc}") from exc

    def get_backend(self):
        if self.backend is None:
            raise RuntimeError("IQM backend is not connected.")
        return self.backend

    def run_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        if self.backend is None:
            raise RuntimeError("Backend is not connected.")

        try:
            qc_transpiled = transpile(
                circuit,
                backend=self.backend,
                optimization_level=3,
            )

            logger.info(
                "Executing optimized QAOA circuit on IQM (%s shots)...",
                shots,
            )
            job = self.backend.run(qc_transpiled, shots=shots)
            job_id = job.job_id() if hasattr(job, "job_id") else "Unknown"

            logger.info("IQM job submitted: %s", job_id)
            result = job.result()
            counts = result.get_counts()

            return {
                "counts": counts,
                "backend_name": self.backend.name,
                "shots": shots,
                "quantum_computer": self.quantum_computer,
                "job_id": job_id,
                "success": True,
            }
        except Exception as exc:
            raise RuntimeError(f"IQM execution error: {exc}") from exc


class VisualizationAssetGenerator:
    """Generate conflict graph visualization for QCentroid assets."""

    def __init__(self, output_dir: str = "additional_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_conflict_graph_visualization(
        self,
        conflict_graph: nx.Graph,
        selected_cycles: List[str],
        pruned_nodes: Set[str],
    ) -> Optional[str]:
        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            pos = nx.spring_layout(
                conflict_graph,
                k=0.5,
                iterations=50,
                seed=42,
            )

            selected_set = set(selected_cycles)
            pruned_set = set(pruned_nodes)
            node_colors = []
            node_sizes = []

            for node in conflict_graph.nodes():
                if node in selected_set:
                    node_colors.append("#2ecc71")
                    node_sizes.append(1000)
                elif node in pruned_set:
                    node_colors.append("#e74c3c")
                    node_sizes.append(700)
                else:
                    node_colors.append("#bdc3c7")
                    node_sizes.append(700)

            nx.draw_networkx_edges(
                conflict_graph,
                pos,
                ax=ax,
                edge_color="#95a5a6",
                width=1.5,
                alpha=0.6,
            )
            nx.draw_networkx_nodes(
                conflict_graph,
                pos,
                ax=ax,
                node_color=node_colors,
                node_size=node_sizes,
                edgecolors="#2c3e50",
                linewidths=2.0,
            )

            labels = {
                node: f"{node}\n(w={conflict_graph.nodes[node].get('weight', 0):.2f})"
                for node in conflict_graph.nodes()
            }
            nx.draw_networkx_labels(
                conflict_graph,
                pos,
                labels,
                ax=ax,
                font_size=9,
                font_weight="bold",
            )

            green_patch = mpatches.Patch(
                color="#2ecc71",
                label="Selected cycles (MWIS solution)",
            )
            red_patch = mpatches.Patch(
                color="#e74c3c",
                label="Pruned cycles",
            )
            gray_patch = mpatches.Patch(
                color="#bdc3c7",
                label="Unselected cycles",
            )
            ax.legend(
                handles=[green_patch, red_patch, gray_patch],
                loc="upper left",
                fontsize=11,
                framealpha=0.95,
            )

            ax.set_title(
                "Railway Rolling Stock Conflict Graph\n"
                "Maximum Weighted Independent Set (MWIS)",
                fontsize=15,
                fontweight="bold",
                pad=20,
            )
            ax.text(
                0.5,
                -0.05,
                f"Nodes: {conflict_graph.number_of_nodes()} | "
                f"Conflicts: {conflict_graph.number_of_edges()}",
                ha="center",
                transform=ax.transAxes,
                fontsize=10,
                style="italic",
            )
            ax.axis("off")
            plt.tight_layout()

            output_path = os.path.join(
                self.output_dir,
                "conflict_graph.png",
            )
            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)
            return output_path

        except Exception as exc:
            logger.error("Visualization failed: %s", exc)
            plt.close("all")
            return None


class MWISPruner:
    """
    Deterministic feasibility repair.

    Given a potentially infeasible measured solution, repeatedly removes the
    lower-weight endpoint of the first internal conflict until the solution
    becomes independent.
    """

    def __init__(self, conflict_graph: nx.Graph):
        self.conflict_graph = conflict_graph.copy()
        self.pruned_nodes: Set[str] = set()
        self.pruning_history: List[Tuple[str, str, float]] = []

    def prune(self, selected_cycles: List[str]) -> Tuple[List[str], int]:
        current_solution = set(selected_cycles)

        while True:
            conflicts = [
                (u, v)
                for u, v in self.conflict_graph.edges()
                if u in current_solution and v in current_solution
            ]

            if not conflicts:
                break

            node1, node2 = conflicts[0]
            weight1 = float(self.conflict_graph.nodes[node1].get("weight", 0.0))
            weight2 = float(self.conflict_graph.nodes[node2].get("weight", 0.0))

            if weight1 <= weight2:
                removed = node1
            else:
                removed = node2

            current_solution.discard(removed)
            self.pruned_nodes.add(removed)
            self.pruning_history.append(
                (node1, node2, min(weight1, weight2))
            )

        return sorted(current_solution), len(self.pruned_nodes)

    def get_pruning_stats(self) -> Dict[str, Any]:
        return {
            "nodes_pruned": len(self.pruned_nodes),
            "pruning_iterations": len(self.pruning_history),
            "pruned_nodes": sorted(self.pruned_nodes),
        }


class QAOAMWISSolver:
    """
    QAOA solver for MWIS.

    The objective is:
        minimize H(x) = -sum(w_i*x_i) + P*sum(x_i*x_j)

    P is deliberately larger than the maximum node weight by default.
    """

    def __init__(
        self,
        conflict_graph: nx.Graph,
        backend=None,
        num_qubits: Optional[int] = None,
        penalty: Optional[float] = None,
    ):
        self.conflict_graph = conflict_graph
        self.backend = backend
        self.nodes = list(conflict_graph.nodes())
        self.num_nodes = len(self.nodes)
        self.num_qubits = num_qubits or self.num_nodes

        if self.num_qubits != self.num_nodes:
            raise ValueError(
                "This PoC uses one qubit per graph node. "
                "num_qubits must equal the number of nodes."
            )

        max_weight = max(
            (float(conflict_graph.nodes[n].get("weight", 0.0)) for n in self.nodes),
            default=1.0,
        )
        self.penalty = float(penalty) if penalty is not None else 1.25 * max_weight

        if self.penalty <= max_weight:
            raise ValueError(
                "Conflict penalty must be strictly greater than the maximum node weight."
            )

    def classical_objective(self, bits: List[int]) -> float:
        """Return the MWIS minimization objective for a binary solution."""
        value = 0.0
        for i, node in enumerate(self.nodes):
            if bits[i]:
                value -= float(self.conflict_graph.nodes[node].get("weight", 0.0))

        for u, v in self.conflict_graph.edges():
            i = self.nodes.index(u)
            j = self.nodes.index(v)
            if bits[i] and bits[j]:
                value += self.penalty

        return value

    def is_feasible(self, selected: List[str]) -> bool:
        selected_set = set(selected)
        return all(
            not (u in selected_set and v in selected_set)
            for u, v in self.conflict_graph.edges()
        )

    def _hamiltonian_coefficients(self):
        """
        Expand H(x) using x=(1-Z)/2.

        H = constant + sum_i h_i Z_i + sum_(i,j) J_ij Z_i Z_j

        h_i = w_i/2 - P*degree_i/4
        J_ij = P/4
        """
        h = {}
        for node in self.nodes:
            weight = float(self.conflict_graph.nodes[node].get("weight", 0.0))
            degree = self.conflict_graph.degree(node)
            h[node] = weight / 2.0 - self.penalty * degree / 4.0

        zz = {}
        for u, v in self.conflict_graph.edges():
            zz[(u, v)] = self.penalty / 4.0

        return h, zz

    def _build_qaoa_circuit(
        self,
        p: int = 1,
        gamma_values=None,
        beta_values=None,
        measure: bool = True,
    ):
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is not available.")

        n = len(self.nodes)
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(n, "c") if measure else None

        qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)

        h, zz = self._hamiltonian_coefficients()

        if gamma_values is None:
            gamma_values = [
                Parameter(f"gamma_{layer}") for layer in range(p)
            ]
        if beta_values is None:
            beta_values = [
                Parameter(f"beta_{layer}") for layer in range(p)
            ]

        for i in range(n):
            qc.h(qr[i])

        for layer in range(p):
            gamma = gamma_values[layer]
            beta = beta_values[layer]

            # exp(-i gamma h_i Z_i) -> RZ(2*gamma*h_i)
            for i, node in enumerate(self.nodes):
                qc.rz(
                    2.0 * gamma * h[node],
                    qr[i],
                )

            # exp(-i gamma J_ij Z_i Z_j) -> RZZ(2*gamma*J_ij)
            for (u, v), coupling in zz.items():
                i = self.nodes.index(u)
                j = self.nodes.index(v)
                qc.rzz(
                    2.0 * gamma * coupling,
                    qr[i],
                    qr[j],
                )

            # Standard X mixer: exp(-i beta X) -> RX(2*beta)
            for i in range(n):
                qc.rx(2.0 * beta, qr[i])

        if measure:
            for i in range(n):
                qc.measure(qr[i], cr[i])

        return qc

    def _counts_to_solution(self, counts: Dict[str, int]) -> Tuple[List[str], str]:
        best_bitstring = max(counts, key=counts.get)
        # Qiskit strings are displayed highest classical bit first.
        selected = [
            self.nodes[i]
            for i in range(len(self.nodes))
            if best_bitstring[-(i + 1)] == "1"
        ]
        return selected, best_bitstring

    def _simulate_counts(self, gamma, beta, p, shots=256):
        if not (QISKIT_AVAILABLE and AER_AVAILABLE):
            return None

        qc = self._build_qaoa_circuit(
            p=p,
            gamma_values=list(gamma),
            beta_values=list(beta),
            measure=True,
        )

        simulator = AerSimulator()
        result = simulator.run(qc, shots=shots).result()
        return result.get_counts()

    def _expected_cost_from_counts(self, counts: Dict[str, int]) -> float:
        total = sum(counts.values())
        if total == 0:
            return float("inf")

        expectation = 0.0
        for bitstring, count in counts.items():
            bits = [
                1 if bitstring[-(i + 1)] == "1" else 0
                for i in range(len(self.nodes))
            ]
            expectation += self.classical_objective(bits) * (count / total)

        return expectation

    def _optimize_parameters(self, p: int, shots: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Optimize QAOA parameters on a classical simulator.

        This is intentional for the PoC: the resulting parameters are then
        sent to IQM for the final hardware execution.
        """
        if not (QISKIT_AVAILABLE and AER_AVAILABLE):
            logger.warning(
                "Qiskit Aer unavailable; using deterministic fallback parameters."
            )
            return (
                np.full(p, np.pi / 4.0),
                np.full(p, np.pi / 8.0),
                float("nan"),
            )

        rng = np.random.default_rng(42)
        x0 = np.concatenate([
            rng.uniform(0.0, 2.0 * np.pi, p),
            rng.uniform(0.0, np.pi, p),
        ])

        eval_shots = max(128, min(shots, 512))

        def objective(x):
            gamma = x[:p]
            beta = x[p:]
            counts = self._simulate_counts(
                gamma,
                beta,
                p,
                shots=eval_shots,
            )
            return self._expected_cost_from_counts(counts)

        if SCIPY_AVAILABLE:
            result = minimize(
                objective,
                x0,
                method="COBYLA",
                options={"maxiter": max(40, 20 * p)},
            )
            x = result.x
            value = float(result.fun)
        else:
            # Lightweight deterministic random search fallback.
            best_x = x0
            best_value = objective(x0)

            for _ in range(30):
                candidate = np.concatenate([
                    rng.uniform(0.0, 2.0 * np.pi, p),
                    rng.uniform(0.0, np.pi, p),
                ])
                candidate_value = objective(candidate)
                if candidate_value < best_value:
                    best_x = candidate
                    best_value = candidate_value

            x = best_x
            value = float(best_value)

        return x[:p], x[p:], value

    def solve_qaoa(self, p: int = 1, shots: int = 1024) -> Dict[str, Any]:
        if not QISKIT_AVAILABLE or self.backend is None:
            logger.warning(
                "Quantum hardware unavailable; using classical greedy fallback."
            )
            return self.solve_greedy()

        gamma, beta, training_cost = self._optimize_parameters(
            p=p,
            shots=shots,
        )

        logger.info(
            "Optimized QAOA parameters: gamma=%s beta=%s",
            np.round(gamma, 5).tolist(),
            np.round(beta, 5).tolist(),
        )

        qc = self._build_qaoa_circuit(
            p=p,
            gamma_values=gamma,
            beta_values=beta,
            measure=True,
        )

        try:
            if isinstance(self.backend, AerSimulator):
                result = self.backend.run(qc, shots=shots).result()
                counts = result.get_counts()
                backend_name = "Qiskit AerSimulator"
                job_id = "local"
            else:
                qc_t = transpile(
                    qc,
                    backend=self.backend,
                    optimization_level=3,
                )
                job = self.backend.run(qc_t, shots=shots)
                job_id = job.job_id() if hasattr(job, "job_id") else "Unknown"
                result = job.result()
                counts = result.get_counts()
                backend_name = getattr(self.backend, "name", "IQM")

            selected, best_bitstring = self._counts_to_solution(counts)

            return {
                "solution": selected,
                "counts": counts,
                "best_bitstring": best_bitstring,
                "algorithm": "QAOA",
                "qaoa_depth": p,
                "gamma": gamma.tolist(),
                "beta": beta.tolist(),
                "training_cost": training_cost,
                "backend_name": backend_name,
                "job_id": job_id,
            }

        except Exception as exc:
            logger.error(
                "QAOA execution failed: %s. Falling back to greedy.",
                exc,
            )
            return self.solve_greedy()

    def solve_greedy(self) -> Dict[str, Any]:
        selected = []
        remaining = set(self.conflict_graph.nodes())

        while remaining:
            best_node = max(
                remaining,
                key=lambda n: self.conflict_graph.nodes[n].get("weight", 0.0),
            )
            selected.append(best_node)
            remaining -= set(self.conflict_graph.neighbors(best_node))
            remaining.discard(best_node)

        return {
            "solution": selected,
            "algorithm": "greedy",
            "is_feasible": True,
        }


class RailwayRollingStockSolver:
    """Main QCentroid solver orchestrator."""

    def __init__(
        self,
        nodes: List[Dict],
        edges: List[List[str]],
        penalty: Optional[float] = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self.penalty_override = penalty
        self._validate_input()
        self.conflict_graph = self._build_conflict_graph()
        self.execution_log: Dict[str, Any] = {}

    def _validate_input(self):
        if not isinstance(self.nodes, list) or not self.nodes:
            raise ValueError("Input must contain a non-empty 'nodes' list.")
        if not isinstance(self.edges, list):
            raise ValueError("Input must contain an 'edges' list.")

        ids = []
        for node in self.nodes:
            if not isinstance(node, dict):
                raise ValueError("Every node must be a JSON object.")

            node_id = node.get("id")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("Every node requires a non-empty string 'id'.")

            weight = node.get("weight")
            if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
                raise ValueError(f"Invalid weight for node {node_id}.")
            if float(weight) <= 0:
                raise ValueError(f"Weight must be positive for node {node_id}.")

            trips = node.get("trips", [])
            if trips is not None and not isinstance(trips, list):
                raise ValueError(f"'trips' must be a list for node {node_id}.")

            if node_id in ids:
                raise ValueError(f"Duplicate node ID: {node_id}")
            ids.append(node_id)

        valid_ids = set(ids)

        for edge in self.edges:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError(
                    "Each edge must be a two-element list: [cycle_1, cycle_2]."
                )
            u, v = edge
            if u not in valid_ids or v not in valid_ids:
                raise ValueError(
                    f"Edge references unknown cycle: [{u}, {v}]"
                )
            if u == v:
                raise ValueError(f"Self-conflict is not allowed: {u}")

    def _build_conflict_graph(self) -> nx.Graph:
        G = nx.Graph()

        for node in self.nodes:
            G.add_node(
                node["id"],
                weight=float(node["weight"]),
                trips=node.get("trips", []),
                empty_km=node.get("empty_km"),
                passenger_km=node.get("passenger_km"),
                operating_cost=node.get("operating_cost"),
            )

        for edge in self.edges:
            G.add_edge(edge[0], edge[1])

        return G

    def _calculate_metrics(self, selected_cycles: List[str]) -> Dict[str, Any]:
        selected_set = set(selected_cycles)

        total_weight = sum(
            float(self.conflict_graph.nodes[n].get("weight", 0.0))
            for n in selected_set
        )

        all_trips = set()
        covered_trips = set()
        for node in self.nodes:
            all_trips.update(node.get("trips", []) or [])

        for node_id in selected_set:
            covered_trips.update(
                self.conflict_graph.nodes[node_id].get("trips", []) or []
            )

        coverage_rate = (
            len(covered_trips) / len(all_trips)
            if all_trips
            else None
        )

        empty_values = [
            self.conflict_graph.nodes[n].get("empty_km")
            for n in selected_set
            if self.conflict_graph.nodes[n].get("empty_km") is not None
        ]

        passenger_values = [
            self.conflict_graph.nodes[n].get("passenger_km")
            for n in selected_set
            if self.conflict_graph.nodes[n].get("passenger_km") is not None
        ]

        cost_values = [
            self.conflict_graph.nodes[n].get("operating_cost")
            for n in selected_set
            if self.conflict_graph.nodes[n].get("operating_cost") is not None
        ]

        return {
            "selected_cycles": len(selected_set),
            "total_weight": total_weight,
            "scheduled_trips": len(all_trips),
            "covered_trips": len(covered_trips),
            "coverage_rate": coverage_rate,
            "coverage_percent": (
                coverage_rate * 100.0 if coverage_rate is not None else None
            ),
            "total_empty_km": sum(empty_values) if empty_values else None,
            "total_passenger_km": sum(passenger_values) if passenger_values else None,
            "total_operating_cost": sum(cost_values) if cost_values else None,
            # FIX: `nx.is_independent` does not exist in NetworkX's public API
            # (AttributeError at runtime in the original PoC). Replaced with a
            # direct independence check equivalent to
            # QAOAMWISSolver.is_feasible.
            "is_feasible": all(
                not (u in selected_set and v in selected_set)
                for u, v in self.conflict_graph.edges()
            ),
        }

    def solve(
        self,
        backend=None,
        qaoa_p: int = 1,
        shots: int = 1000,
        penalty: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, Any], Set[str]]:

        start_time = time.time()

        if qaoa_p < 1:
            raise ValueError("qaoa_depth must be >= 1.")
        if shots < 1:
            raise ValueError("shots must be >= 1.")

        selected_penalty = (
            penalty
            if penalty is not None
            else self.penalty_override
        )

        logger.info(
            "Starting railway rolling-stock MWIS solver: %d cycles, %d conflicts.",
            self.conflict_graph.number_of_nodes(),
            self.conflict_graph.number_of_edges(),
        )

        solver = QAOAMWISSolver(
            self.conflict_graph,
            backend=backend,
            penalty=selected_penalty,
        )

        qaoa_result = solver.solve_qaoa(
            p=qaoa_p,
            shots=shots,
        )
        initial_solution = qaoa_result.get("solution", [])

        logger.info(
            "Initial QAOA/heuristic solution: %d cycles.",
            len(initial_solution),
        )

        pruner = MWISPruner(self.conflict_graph)
        final_solution, pruned_count = pruner.prune(initial_solution)

        metrics = self._calculate_metrics(final_solution)
        metrics.update({
            "execution_time_seconds": time.time() - start_time,
            "initial_solution_size": len(initial_solution),
            "final_solution_size": len(final_solution),
            "nodes_pruned": pruned_count,
            "qaoa_algorithm": qaoa_result.get("algorithm", "qaoa"),
            "qaoa_depth": qaoa_p,
            "shots": shots,
            "penalty": solver.penalty,
            "initial_bitstring": qaoa_result.get("best_bitstring"),
            "training_cost": qaoa_result.get("training_cost"),
            "job_id": qaoa_result.get("job_id"),
        })

        self.execution_log = metrics

        logger.info(
            "Final solution: %d cycles; total weight %.4f; feasible=%s.",
            len(final_solution),
            metrics["total_weight"],
            metrics["is_feasible"],
        )

        return final_solution, metrics, pruner.pruned_nodes


def run(
    input_data: Dict[str, Any],
    solver_params: Dict[str, Any],
    extra_arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    QCentroid entry point.

    Required input_data:
      - nodes
      - edges

    Optional node fields:
      - trips
      - empty_km
      - passenger_km
      - operating_cost

    solver_params:
      - iqm_token
      - quantum_computer
      - server_url
      - shots
      - qaoa_depth
      - penalty
    """

    logger.info("=" * 80)
    logger.info("QCentroid - Railway Rolling Stock MWIS Solver")
    logger.info("=" * 80)

    try:
        nodes = input_data.get("nodes", [])
        edges = input_data.get("edges", [])

        if not nodes:
            return {
                "selected_cycles": [],
                "total_weight": 0.0,
                "nodes_pruned": 0,
                "coverage_rate": 0.0,
                "error": "No input nodes",
            }

        iqm_token = solver_params.get("iqm_token")
        quantum_computer = solver_params.get(
            "quantum_computer",
            "emerald",
        )
        server_url = solver_params.get(
            "server_url",
            "https://resonance.iqm.tech/",
        )
        shots = int(solver_params.get("shots", 1024))
        qaoa_depth = int(solver_params.get("qaoa_depth", 1))
        penalty = solver_params.get("penalty")

        backend = None
        backend_info = "Classical Greedy Heuristic"

        # Prefer IQM if credentials are supplied.
        if iqm_token:
            try:
                iqm_manager = IQMBackendManager(
                    iqm_token=iqm_token,
                    quantum_computer=quantum_computer,
                    server_url=server_url,
                )
                backend = iqm_manager.get_backend()
                backend_info = f"IQM Resonance ({quantum_computer})"
            except Exception as exc:
                logger.warning("IQM connection failed: %s", exc)

        # If IQM is unavailable, use local Aer when possible.
        if backend is None and QISKIT_AVAILABLE and AER_AVAILABLE:
            backend = AerSimulator()
            backend_info = "Qiskit AerSimulator"

        solver = RailwayRollingStockSolver(
            nodes,
            edges,
            penalty=penalty,
        )

        selected_cycles, metrics, pruned_nodes = solver.solve(
            backend=backend,
            qaoa_p=qaoa_depth,
            shots=shots,
            penalty=penalty,
        )

        asset_generator = VisualizationAssetGenerator()
        assets = {}

        graph_path = asset_generator.generate_conflict_graph_visualization(
            solver.conflict_graph,
            selected_cycles,
            pruned_nodes,
        )

        if graph_path:
            assets["conflict_graph_png"] = graph_path

        response = {
            "selected_cycles": selected_cycles,
            "total_weight": metrics.get("total_weight", 0.0),
            "nodes_pruned": metrics.get("nodes_pruned", 0),
            "coverage_rate": metrics.get("coverage_rate"),
            "coverage_percent": metrics.get("coverage_percent"),
            "covered_trips": metrics.get("covered_trips"),
            "scheduled_trips": metrics.get("scheduled_trips"),
            "total_empty_km": metrics.get("total_empty_km"),
            "total_passenger_km": metrics.get("total_passenger_km"),
            "total_operating_cost": metrics.get("total_operating_cost"),
            "is_feasible": metrics.get("is_feasible"),
            "execution_metrics": {
                "execution_time_seconds": metrics.get(
                    "execution_time_seconds",
                    0.0,
                ),
                "initial_solution_size": metrics.get(
                    "initial_solution_size",
                    0,
                ),
                "final_solution_size": metrics.get(
                    "final_solution_size",
                    0,
                ),
                "qaoa_depth": qaoa_depth,
                "shots": shots,
                "penalty": metrics.get("penalty"),
                "training_cost": metrics.get("training_cost"),
                "job_id": metrics.get("job_id"),
            },
            "backend_used": backend_info,
            "assets": assets,
            "pruning_stats": {
                "nodes_pruned": len(pruned_nodes),
                "pruned_nodes": sorted(pruned_nodes),
            },
        }

        logger.info("Solver execution completed successfully.")
        return response

    except Exception as exc:
        logger.exception("Solver execution failed.")
        return {
            "selected_cycles": [],
            "total_weight": 0.0,
            "nodes_pruned": 0,
            "coverage_rate": None,
            "is_feasible": False,
            "backend_used": "ERROR",
            "error": str(exc),
        }


if __name__ == "__main__":
    # Caso de uso: Planificación de Material Rodante (Rolling Stock Planning)
    # Corredor de 5 ciudades alemanas (Berlín-Hamburgo-Frankfurt-Colonia-Múnich),
    # modelo simplificado de 5 nodos basado en la red de Deutsche Bahn.
    #
    # Grafo de conflictos en anillo: cada ciclo comparte exactamente un trip
    # con cada uno de sus dos vecinos (C01-C02-C03-C04-C05-C01), por lo que
    # la solución óptima (MWIS) selecciona nodos alternos no adyacentes.
    # Solución esperada: C03 + C05 (peso total 179.0).
    example_input = {
        "nodes": [
            {
                "id": "C01",
                "weight": 86.0,
                "trips": ["T01", "T02"],
                "passenger_km": 8200,
                "empty_km": 80,
                "operating_cost": 1180,
            },
            {
                "id": "C02",
                "weight": 78.0,
                "trips": ["T02", "T03"],
                "passenger_km": 7600,
                "empty_km": 120,
                "operating_cost": 1210,
            },
            {
                "id": "C03",
                "weight": 91.0,
                "trips": ["T03", "T04"],
                "passenger_km": 9000,
                "empty_km": 60,
                "operating_cost": 1160,
            },
            {
                "id": "C04",
                "weight": 73.0,
                "trips": ["T04", "T05"],
                "passenger_km": 7100,
                "empty_km": 150,
                "operating_cost": 1240,
            },
            {
                "id": "C05",
                "weight": 88.0,
                "trips": ["T05", "T01"],
                "passenger_km": 8500,
                "empty_km": 70,
                "operating_cost": 1190,
            },
        ],
        "edges": [
            ["C01", "C02"],  # Conflicto en T02 (Hamburgo -> Frankfurt)
            ["C02", "C03"],  # Conflicto en T03 (Frankfurt -> Colonia)
            ["C03", "C04"],  # Conflicto en T04 (Colonia -> Múnich)
            ["C04", "C05"],  # Conflicto en T05 (Múnich -> Berlín)
            ["C05", "C01"],  # Conflicto en T01 (Berlín -> Hamburgo)
        ],
    }

    example_params = {
        "iqm_token": None,             # Simulador local Aer / IQM Resonance
        "quantum_computer": "emerald",
        "server_url": "https://resonance.iqm.tech/",
        "shots": 2048,                 # Más shots -> mayor precisión estadística
        "qaoa_depth": 2,               # Profundidad p de QAOA
        "penalty": 95.0,               # Ligeramente por encima de max_weight (91.0)
    }

    nodes = example_input["nodes"]
    edges = example_input["edges"]
    print(f"{len(nodes)} ciclos candidatos, {len(edges)} conflictos.")

    result = run(example_input, example_params, {})
    print(json.dumps(result, indent=2, default=str))
