"""
QCentroid Platform Solver: Railway Rolling Stock Cycle Selection via Maximum Weighted Independent Set Optimization
Proof of Concept Implementation for IQM & Deutsche Bahn Use Case

This module implements a quantum-classical hybrid solver for the Maximum Weighted Independent Set (MWIS) problem
on a conflict graph of railway rolling stock cycles, with automatic constraint pruning for NISQ-era hardware resilience.

Compliance Level: QCentroid Platform v1.0 with IQM Resonance Integration (Fixed URL Handling)
"""

import json
import logging
import time
import os
import re
from typing import Dict, List, Tuple, Any, Set, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, urlunparse

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

# Quantum imports with fallback
try:
    from iqm.qiskit_iqm import IQMProvider
    IQM_AVAILABLE = True
except ImportError:
    IQM_AVAILABLE = False

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, execute
    from qiskit.circuit import Parameter
    from qiskit.primitives import Sampler
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Initialize QCentroid logger
logger = logging.getLogger("qcentroid-user-log")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class CycleNode:
    """Represents a railway rolling stock cycle candidate."""
    id: str
    weight: float
    trips: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConflictEdge:
    """Represents a conflict between two cycles (shared trip)."""
    cycle_1: str
    cycle_2: str


class URLSanitizer:
    """
    Defensive URL sanitizer for IQM Resonance server URLs.
    Removes quantum computer names from URL to prevent ClientConfigurationError.
    """
    
    @staticmethod
    def sanitize_iqm_url(raw_url: str) -> str:
        """
        Sanitize IQM server URL by removing quantum computer suffixes.
        
        Args:
            raw_url: Raw URL that may contain quantum computer name
                    (e.g., "https://resonance.iqm.tech/emerald" or "https://resonance.iqm.tech/sirius")
        
        Returns:
            Clean base URL without quantum computer name
            (e.g., "https://resonance.iqm.tech/")
        
        Examples:
            >>> URLSanitizer.sanitize_iqm_url("https://resonance.iqm.tech/emerald")
            'https://resonance.iqm.tech/'
            
            >>> URLSanitizer.sanitize_iqm_url("https://resonance.iqm.tech/sirius/")
            'https://resonance.iqm.tech/'
            
            >>> URLSanitizer.sanitize_iqm_url("https://resonance.iqm.tech/")
            'https://resonance.iqm.tech/'
        """
        if not raw_url:
            return "https://resonance.iqm.tech/"
        
        # Remove trailing slashes for consistent processing
        url = raw_url.rstrip("/")
        
        # List of known quantum computer names to remove
        known_qpus = ["emerald", "sirius", "garnet", "sapphire", "ruby", "diamond"]
        
        # Check if URL ends with any known QPU name
        for qpu in known_qpus:
            # Match pattern: /qpu_name at the end
            if url.endswith(f"/{qpu}"):
                url = url[:-len(f"/{qpu}")]
                logger.info(f"Removed quantum computer suffix '/{qpu}' from URL")
                break
        
        # Use regex for more flexible matching (case-insensitive)
        # Pattern: URL ending with /word (where word is not a file extension)
        url_pattern = r"^(.*?)(?:/[a-zA-Z]+)?$"
        match = re.match(url_pattern, url)
        
        if match:
            base = match.group(1)
            # Verify it ends with .tech or .com or similar (to avoid removing legitimate path components)
            if base and any(base.endswith(domain) for domain in [".tech", ".com", ".io", ".net", ".org"]):
                url = base
        
        # Ensure URL ends with /
        if not url.endswith("/"):
            url += "/"
        
        logger.debug(f"Sanitized URL: {url}")
        return url
    
    @staticmethod
    def validate_iqm_url(url: str) -> bool:
        """
        Validate that URL is a proper base URL without quantum computer name.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid base URL, False otherwise
        """
        try:
            parsed = urlparse(url)
            
            # Should have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Path should not contain quantum computer names
            path = parsed.path.strip("/")
            if path and any(qpu in path.lower() for qpu in ["emerald", "sirius", "garnet", "sapphire", "ruby", "diamond"]):
                return False
            
            return True
        except Exception:
            return False


class IQMBackendManager:
    """
    Manages connection to IQM Resonance hardware with proper URL handling.
    Handles token authentication, quantum computer selection, and circuit execution.
    """
    
    def __init__(
        self,
        iqm_token: str,
        quantum_computer: str = "emerald",
        server_url: str = "https://resonance.iqm.tech/"
    ):
        """
        Initialize IQM Resonance backend connection.
        
        Args:
            iqm_token: Authentication token for IQM API
            quantum_computer: Quantum computer name ('emerald', 'sirius', etc.)
            server_url: IQM Resonance server base URL (will be sanitized)
            
        Raises:
            ValueError: If token is not provided
            RuntimeError: If connection to IQM hardware fails
        """
        if not iqm_token:
            raise ValueError("Error: IQM token is required but was not provided. Set 'iqm_token' in solver_params.")
        
        self.iqm_token = iqm_token
        self.quantum_computer = quantum_computer
        
        # Sanitize URL to remove any quantum computer suffix
        self.server_url = URLSanitizer.sanitize_iqm_url(server_url)
        
        if not URLSanitizer.validate_iqm_url(self.server_url):
            raise ValueError(f"Invalid IQM server URL after sanitization: {self.server_url}")
        
        self.backend = None
        self._connect()
    
    def _connect(self):
        """
        Establish connection to IQM Resonance hardware.
        
        Raises:
            RuntimeError: If connection to IQM hardware fails
        """
        if not IQM_AVAILABLE:
            raise RuntimeError("IQM Qiskit plugin is not installed. Install with: pip install iqm-client[qiskit]")
        
        try:
            logger.info(f"Connecting to IQM Resonance at {self.server_url}...")
            logger.info(f"Quantum computer: {self.quantum_computer}")
            logger.info(f"Server URL (sanitized): {self.server_url}")
            
            # CRITICAL FIX: Pass base URL and quantum_computer separately
            # This prevents ClientConfigurationError about URL containing quantum computer name
            provider = IQMProvider(
                url=self.server_url,           # Base URL WITHOUT quantum computer name
                quantum_computer=self.quantum_computer,  # Quantum computer name as separate parameter
                token=self.iqm_token
            )
            
            self.backend = provider.get_backend()
            logger.info(f"✓ Successfully connected to IQM Resonance backend: {self.backend.name}")
            
            # Log backend properties if available
            try:
                if hasattr(self.backend, 'num_qubits'):
                    logger.info(f"✓ Backend properties: {self.backend.num_qubits} qubits available")
            except Exception:
                pass  # Backend properties may not always be accessible
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to connect to IQM Resonance: {error_msg}")
            
            # Provide helpful error messages
            if "ClientConfigurationError" in error_msg and "quantum computer name" in error_msg:
                logger.error("This is likely a URL formatting issue. Ensure:")
                logger.error("  1. server_url does NOT include quantum computer name (e.g., no '/emerald' suffix)")
                logger.error("  2. quantum_computer parameter is set separately")
            
            raise RuntimeError(f"IQM connection failed: {error_msg}")
    
    def get_backend(self):
        """
        Return the connected backend.
        
        Returns:
            IQM backend object
            
        Raises:
            RuntimeError: If backend is not connected
        """
        if self.backend is None:
            raise RuntimeError("Backend is not connected. Call _connect() first.")
        return self.backend
    
    def run_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        use_timeslot: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a quantum circuit on IQM Resonance hardware.
        
        Args:
            circuit: Qiskit QuantumCircuit to execute
            shots: Number of measurement shots
            use_timeslot: Whether to use timeslot optimization
            
        Returns:
            Dictionary with measurement counts and metadata
            
        Raises:
            RuntimeError: If circuit execution fails
        """
        if self.backend is None:
            raise RuntimeError("Backend is not connected.")
        
        try:
            logger.info(f"Transpiling circuit for IQM backend ({self.quantum_computer})...")
            qc_transpiled = transpile(circuit, backend=self.backend, optimization_level=3)
            
            logger.info(f"Executing circuit with {shots} shots on IQM Resonance...")
            job = self.backend.run(qc_transpiled, shots=shots)
            
            job_id = job.job_id() if hasattr(job, 'job_id') else 'Unknown'
            logger.info(f"Job submitted: {job_id}")
            
            result = job.result()
            counts = result.get_counts()
            
            total_results = sum(counts.values())
            logger.info(f"✓ Execution completed. Received {total_results} measurement results.")
            
            return {
                "counts": counts,
                "backend_name": self.backend.name,
                "shots": shots,
                "quantum_computer": self.quantum_computer,
                "job_id": job_id,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Circuit execution on IQM failed: {str(e)}")
            raise RuntimeError(f"IQM execution error: {str(e)}")


class VisualizationAssetGenerator:
    """
    Generates visual assets for QCentroid Platform dashboard (Assets tab).
    Handles PNG rendering, HTML reports, and data summaries.
    """
    
    def __init__(self, output_dir: str = "additional_output"):
        """
        Initialize the asset generator.
        
        Args:
            output_dir: Directory for saving generated assets
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")
    
    def generate_conflict_graph_visualization(
        self,
        conflict_graph: nx.Graph,
        selected_cycles: List[str],
        pruned_nodes: Set[str]
    ) -> str:
        """
        Render the conflict graph with visual differentiation of selected/pruned nodes.
        
        Args:
            conflict_graph: NetworkX graph of cycle conflicts
            selected_cycles: List of selected cycle IDs (highlighted in green)
            pruned_nodes: Set of pruned cycle IDs (highlighted in red)
            
        Returns:
            Path to saved PNG file
        """
        try:
            self._ensure_output_dir()
            
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            # Compute layout
            pos = nx.spring_layout(conflict_graph, k=0.5, iterations=50, seed=42)
            
            # Determine node colors
            selected_set = set(selected_cycles)
            pruned_set = set(pruned_nodes)
            node_colors = []
            node_sizes = []
            
            for node in conflict_graph.nodes():
                if node in selected_set:
                    node_colors.append('#2ecc71')  # Green - Selected
                    node_sizes.append(1000)         # Larger
                elif node in pruned_set:
                    node_colors.append('#e74c3c')  # Red - Pruned
                    node_sizes.append(700)          # Medium
                else:
                    node_colors.append('#bdc3c7')  # Gray - Not selected
                    node_sizes.append(700)          # Medium
            
            # Draw edges
            nx.draw_networkx_edges(
                conflict_graph, pos, ax=ax,
                edge_color='#95a5a6', width=1.5, alpha=0.6
            )
            
            # Draw nodes with size variation
            nx.draw_networkx_nodes(
                conflict_graph, pos, ax=ax,
                node_color=node_colors,
                node_size=node_sizes,
                edgecolors='#2c3e50',
                linewidths=2.5
            )
            
            # Draw labels with cycle ID and weight
            labels = {
                node: f"{node}\n(w={conflict_graph.nodes[node].get('weight', 0):.2f})"
                for node in conflict_graph.nodes()
            }
            nx.draw_networkx_labels(conflict_graph, pos, labels, ax=ax, font_size=9, font_weight='bold')
            
            # Create legend
            green_patch = mpatches.Patch(color='#2ecc71', label='Selected Cycles (MWIS Solution)')
            red_patch = mpatches.Patch(color='#e74c3c', label='Pruned Cycles (Conflict Removed)')
            gray_patch = mpatches.Patch(color='#bdc3c7', label='Unselected Cycles')
            ax.legend(handles=[green_patch, red_patch, gray_patch], loc='upper left', fontsize=11, framealpha=0.95)
            
            # Title and labels
            ax.set_title(
                'Railway Rolling Stock Conflict Graph\nMaximum Weighted Independent Set (MWIS) Solution',
                fontsize=15, fontweight='bold', pad=20
            )
            ax.text(0.5, -0.05, f'Total Nodes: {conflict_graph.number_of_nodes()} | Total Conflicts: {conflict_graph.number_of_edges()}',
                    ha='center', transform=ax.transAxes, fontsize=10, style='italic', color='#34495e')
            
            ax.axis('off')
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(self.output_dir, "conflict_graph.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"✓ Conflict graph visualization saved to: {output_path}")
            plt.close(fig)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to generate conflict graph visualization: {e}")
            plt.close('all')
            return None


class MWISPruner:
    """
    Iterative constraint pruning engine for Maximum Weighted Independent Set solutions.
    Removes conflicting nodes by greedily eliminating the node with minimum weight in each conflict pair.
    """
    
    def __init__(self, conflict_graph: nx.Graph):
        """
        Initialize the pruner with the conflict graph.
        
        Args:
            conflict_graph: NetworkX graph where edges represent conflicts between cycles
        """
        self.conflict_graph = conflict_graph.copy()
        self.pruned_nodes: Set[str] = set()
        self.pruning_history: List[Tuple[str, str, float]] = []
    
    def prune(self, selected_cycles: List[str]) -> Tuple[List[str], int]:
        """
        Iteratively prune the selected cycles until no conflicts remain.
        
        Args:
            selected_cycles: Initial list of cycle IDs from quantum solution
            
        Returns:
            Tuple of (pruned_cycle_list, number_of_nodes_pruned)
        """
        current_solution = set(selected_cycles)
        iteration = 0
        
        while True:
            iteration += 1
            # Find all conflicting edges within current solution
            conflicts = []
            for edge in self.conflict_graph.edges():
                node1, node2 = edge
                if node1 in current_solution and node2 in current_solution:
                    conflicts.append((node1, node2))
            
            if not conflicts:
                # No conflicts remain; solution is valid
                break
            
            # Select first conflict and remove lower-weight node
            node1, node2 = conflicts[0]
            weight1 = self.conflict_graph.nodes[node1].get('weight', 0.0)
            weight2 = self.conflict_graph.nodes[node2].get('weight', 0.0)
            
            if weight1 <= weight2:
                removed_node = node1
            else:
                removed_node = node2
            
            current_solution.discard(removed_node)
            self.pruned_nodes.add(removed_node)
            self.pruning_history.append((node1, node2, float(max(weight1, weight2))))
            
            logger.info(f"Pruning iteration {iteration}: removed cycle '{removed_node}' (weight={min(weight1, weight2):.4f}) due to conflict.")
        
        pruned_count = len(self.pruned_nodes)
        return list(current_solution), pruned_count
    
    def get_pruning_stats(self) -> Dict[str, Any]:
        """Return pruning statistics."""
        return {
            "nodes_pruned": len(self.pruned_nodes),
            "pruning_iterations": len(self.pruning_history),
            "pruned_nodes": list(self.pruned_nodes)
        }


class QAOAMWISSolver:
    """
    Quantum Approximate Optimization Algorithm (QAOA) solver for MWIS on railway conflict graphs.
    Falls back to classical heuristics if quantum hardware is unavailable.
    """
    
    def __init__(self, conflict_graph: nx.Graph, backend=None, num_qubits: Optional[int] = None):
        """
        Initialize QAOA solver.
        
        Args:
            conflict_graph: NetworkX graph representing cycle conflicts
            backend: Quantum backend (IQM or Qiskit simulator). None for fallback to greedy.
            num_qubits: Override for number of qubits (useful for subgraph selection)
        """
        self.conflict_graph = conflict_graph
        self.backend = backend
        self.num_nodes = len(conflict_graph.nodes())
        self.num_qubits = num_qubits or self.num_nodes
    
    def solve_qaoa(self, p: int = 1, shots: int = 1000) -> Dict[str, Any]:
        """
        Solve MWIS using QAOA circuit.
        
        Args:
            p: QAOA depth (number of alternating mixer/problem Hamiltonian layers)
            shots: Number of shots for measurement
            
        Returns:
            Dictionary with solution bitstring and measured counts
        """
        if not QISKIT_AVAILABLE or self.backend is None:
            logger.warning("Qiskit or backend unavailable; falling back to greedy heuristic.")
            return self.solve_greedy()
        
        try:
            nodes_list = list(self.conflict_graph.nodes())
            n = len(nodes_list)
            
            # Build QAOA circuit
            qc = self._build_qaoa_circuit(nodes_list, p)
            
            # Transpile for backend
            if self.backend:
                qc_t = transpile(qc, backend=self.backend, optimization_level=3)
            else:
                qc_t = qc
            
            # Execute
            job = execute(qc_t, backend=self.backend or AerSimulator(), shots=shots)
            result = job.result()
            counts = result.get_counts(qc_t)
            
            # Extract best solution
            best_bitstring = max(counts, key=counts.get)
            selected = [nodes_list[i] for i in range(n) if best_bitstring[-(i+1)] == '1']
            
            return {
                "solution": selected,
                "counts": counts,
                "best_bitstring": best_bitstring
            }
        
        except Exception as e:
            logger.error(f"QAOA execution failed: {e}. Falling back to greedy heuristic.")
            return self.solve_greedy()
    
    def _build_qaoa_circuit(self, nodes: List[str], p: int = 1) -> QuantumCircuit:
        """
        Build QAOA circuit for MWIS.
        
        The problem Hamiltonian encodes cycle weights on Z terms,
        and penalties for conflicting cycles (edges).
        """
        n = len(nodes)
        qr = QuantumRegister(n, 'q')
        cr = ClassicalRegister(n, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Initial superposition
        for i in range(n):
            qc.h(qr[i])
        
        # QAOA layers
        for layer in range(p):
            # Problem Hamiltonian (apply ZZ gates for edges, Z gates for weights)
            gamma = Parameter(f'gamma_{layer}')
            for i, node in enumerate(nodes):
                weight = self.conflict_graph.nodes[node].get('weight', 0.0)
                qc.rz(2 * gamma * weight, qr[i])
            
            for edge in self.conflict_graph.edges():
                u, v = edge
                if u in nodes and v in nodes:
                    i, j = nodes.index(u), nodes.index(v)
                    # Penalty for selecting both conflicting nodes
                    qc.rzz(2 * gamma, qr[i], qr[j])
            
            # Mixer Hamiltonian
            beta = Parameter(f'beta_{layer}')
            for i in range(n):
                qc.rx(2 * beta, qr[i])
        
        # Measurement
        for i in range(n):
            qc.measure(qr[i], cr[i])
        
        return qc
    
    def solve_greedy(self) -> Dict[str, Any]:
        """
        Greedy heuristic for MWIS: iteratively select maximum-weight nodes
        that don't conflict with already selected nodes.
        """
        selected = []
        remaining = set(self.conflict_graph.nodes())
        
        while remaining:
            # Select node with highest weight
            best_node = max(remaining, key=lambda n: self.conflict_graph.nodes[n].get('weight', 0.0))
            selected.append(best_node)
            
            # Remove neighbors (conflicting nodes)
            neighbors = set(self.conflict_graph.neighbors(best_node))
            remaining -= neighbors
            remaining.discard(best_node)
        
        return {
            "solution": selected,
            "algorithm": "greedy",
            "is_feasible": True
        }


class RailwayRollingStockSolver:
    """
    Main solver orchestrator for the Railway Rolling Stock Cycle Selection problem.
    Integrates graph construction, quantum solving, constraint pruning, and visualization.
    """
    
    def __init__(self, nodes: List[Dict], edges: List[List[str]]):
        """
        Initialize the solver.
        
        Args:
            nodes: List of node dictionaries with 'id', 'weight', and optional 'trips'
            edges: List of edge pairs [id1, id2] representing conflicts
        """
        self.nodes = nodes
        self.edges = edges
        self.conflict_graph = self._build_conflict_graph()
        self.execution_log = {}
    
    def _build_conflict_graph(self) -> nx.Graph:
        """
        Construct the conflict graph from nodes and edges.
        
        Returns:
            NetworkX Graph with node weights and edge attributes
        """
        G = nx.Graph()
        
        # Add nodes with weights
        for node in self.nodes:
            node_id = node.get('id')
            weight = node.get('weight', 0.0)
            trips = node.get('trips', [])
            G.add_node(node_id, weight=weight, trips=trips)
        
        # Add edges (conflicts)
        for edge in self.edges:
            if len(edge) == 2:
                id1, id2 = edge[0], edge[1]
                if id1 in G.nodes() and id2 in G.nodes():
                    G.add_edge(id1, id2)
        
        return G
    
    def solve(self, backend=None, qaoa_p: int = 1, shots: int = 1000) -> Tuple[List[str], Dict[str, Any], Set[str]]:
        """
        Execute the complete solving pipeline.
        
        Args:
            backend: Quantum backend for QAOA (None for greedy fallback)
            qaoa_p: QAOA depth parameter
            shots: Number of quantum shots
            
        Returns:
            Tuple of (selected_cycle_ids, solver_metrics, pruned_nodes_set)
        """
        start_time = time.time()
        
        logger.info(f"Starting MWIS solver for Railway Rolling Stock Planning.")
        logger.info(f"Conflict graph: {self.conflict_graph.number_of_nodes()} cycles, {self.conflict_graph.number_of_edges()} conflicts.")
        
        # Solve using QAOA
        solver = QAOAMWISSolver(self.conflict_graph, backend=backend)
        qaoa_result = solver.solve_qaoa(p=qaoa_p, shots=shots)
        initial_solution = qaoa_result.get("solution", [])
        
        logger.info(f"Initial quantum solution: {len(initial_solution)} cycles selected.")
        
        # Prune invalid configurations
        pruner = MWISPruner(self.conflict_graph)
        final_solution, pruned_count = pruner.prune(initial_solution)
        
        # Calculate metrics
        total_weight = sum(self.conflict_graph.nodes[node_id].get('weight', 0.0) for node_id in final_solution)
        coverage_rate = total_weight / sum(node.get('weight', 0.0) for node in self.nodes) if self.nodes else 0.0
        
        execution_time = time.time() - start_time
        
        # Log results
        logger.info(f"Final solution: {len(final_solution)} cycles with total weight {total_weight:.4f}.")
        logger.info(f"Nodes pruned: {pruned_count}.")
        logger.info(f"Coverage rate: {coverage_rate:.4f} ({coverage_rate*100:.2f}%).")
        logger.info(f"Execution time: {execution_time:.4f}s.")
        
        self.execution_log = {
            "execution_time_seconds": execution_time,
            "initial_solution_size": len(initial_solution),
            "final_solution_size": len(final_solution),
            "nodes_pruned": pruned_count,
            "total_weight": total_weight,
            "coverage_rate": coverage_rate,
            "qaoa_algorithm": qaoa_result.get("algorithm", "qaoa"),
            "pruning_stats": pruner.get_pruning_stats()
        }
        
        return final_solution, self.execution_log, pruner.pruned_nodes


def run(input_data: Dict[str, Any], solver_params: Dict[str, Any], extra_arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for the QCentroid Platform.
    
    QCentroid Contract Compliance with IQM Resonance Integration (Fixed URL Handling):
    - Accepts input_data with 'nodes' and 'edges' keys
    - Reads 'iqm_token', 'quantum_computer', 'server_url' from solver_params
    - Sanitizes server_url to remove quantum computer suffixes
    - Integrates with IQM Resonance hardware via IQMProvider
    - Performs constraint pruning to ensure 100% feasible solution
    - Generates visualization assets for QCentroid dashboard
    - Returns JSON-serializable dictionary with solution and metrics
    
    Args:
        input_data: Dictionary with keys:
            - 'nodes': List[Dict] with 'id' (str), 'weight' (float), optional 'trips' (List[str])
            - 'edges': List[List[str]] with conflict pairs
        
        solver_params: Configuration parameters:
            - 'iqm_token': (Optional) IQM API token for hardware access
            - 'quantum_computer': (Optional) Quantum computer name ('emerald', 'sirius', etc., default: 'emerald')
            - 'server_url': (Optional) IQM server base URL (default: 'https://resonance.iqm.tech/')
                           Note: URL will be sanitized to remove quantum computer suffixes
            - 'shots': (Optional) Number of quantum measurement shots (default 1024)
            - 'qaoa_depth': (Optional) QAOA circuit depth p (default 1)
            - 'use_timeslot': (Optional) Use IQM timeslot optimization (default False)
        
        extra_arguments: Runtime arguments injected by QCentroid platform
    
    Returns:
        Dictionary with keys:
        - 'selected_cycles': List[str] of cycle IDs in final solution
        - 'total_weight': float, sum of weights of selected cycles
        - 'nodes_pruned': int, number of cycles removed during pruning
        - 'coverage_rate': float, ratio of solution weight to total available weight
        - 'execution_metrics': dict with detailed timing and solver stats
        - 'backend_used': str, information about backend used
        - 'assets': dict with paths to generated visualization assets
    """
    
    logger.info("=" * 80)
    logger.info("QCentroid Platform - Railway Rolling Stock MWIS Solver")
    logger.info("=" * 80)
    
    # Extract and validate input data
    nodes = input_data.get('nodes', [])
    edges = input_data.get('edges', [])
    
    if not nodes:
        logger.error("No nodes provided in input_data.")
        return {
            "selected_cycles": [],
            "total_weight": 0.0,
            "nodes_pruned": 0,
            "coverage_rate": 0.0,
            "error": "No input nodes"
        }
    
    logger.info(f"Input: {len(nodes)} cycles, {len(edges)} conflict constraints")
    
    # Extract solver parameters
    iqm_token = solver_params.get('iqm_token')
    quantum_computer = solver_params.get('quantum_computer', 'emerald')
    server_url = solver_params.get('server_url', 'https://resonance.iqm.tech/')
    shots = solver_params.get('shots', 1024)
    qaoa_depth = solver_params.get('qaoa_depth', 1)
    use_timeslot = solver_params.get('use_timeslot', False)
    
    # Initialize backend and determine backend type
    backend = None
    backend_info = "Qiskit Simulator (Fallback)"
    
    if iqm_token:
        # Try to connect to IQM Resonance
        try:
            logger.info("Initializing IQM Resonance backend...")
            logger.debug(f"Raw server_url: {server_url}")
            logger.debug(f"Quantum computer: {quantum_computer}")
            
            # IQMBackendManager will handle URL sanitization internally
            iqm_manager = IQMBackendManager(
                iqm_token=iqm_token,
                quantum_computer=quantum_computer,
                server_url=server_url
            )
            backend = iqm_manager.get_backend()
            backend_info = f"IQM Resonance ({quantum_computer})"
            logger.info(f"✓ Successfully connected to IQM Resonance backend")
            
        except Exception as e:
            logger.warning(f"IQM Resonance connection failed: {e}")
            logger.info("Falling back to Qiskit Simulator...")
            if QISKIT_AVAILABLE:
                backend = AerSimulator()
                backend_info = "Qiskit AerSimulator (Fallback from IQM Error)"
            else:
                logger.warning("Qiskit not available. Using classical greedy solver.")
                backend_info = "Classical Greedy Heuristic"
    else:
        # No IQM token provided, use Qiskit simulator or greedy
        if QISKIT_AVAILABLE:
            logger.info("Using Qiskit AerSimulator (no IQM token provided).")
            backend = AerSimulator()
            backend_info = "Qiskit AerSimulator"
        else:
            logger.warning("Neither IQM nor Qiskit available. Using classical greedy solver.")
            backend_info = "Classical Greedy Heuristic"
    
    # Create solver instance
    solver = RailwayRollingStockSolver(nodes, edges)
    
    # Solve
    selected_cycles, metrics, pruned_nodes_set = solver.solve(backend=backend, qaoa_p=qaoa_depth, shots=shots)
    
    # Generate visualization assets
    logger.info("Generating visualization assets for QCentroid dashboard...")
    asset_generator = VisualizationAssetGenerator(output_dir="additional_output")
    assets = {}
    
    try:
        # Generate conflict graph PNG
        graph_path = asset_generator.generate_conflict_graph_visualization(
            solver.conflict_graph,
            selected_cycles,
            pruned_nodes_set
        )
        if graph_path:
            assets['conflict_graph_png'] = graph_path
    except Exception as e:
        logger.error(f"Error generating conflict graph visualization: {e}")
    
    # Build response
    response = {
        "selected_cycles": selected_cycles,
        "total_weight": metrics.get('total_weight', 0.0),
        "nodes_pruned": metrics.get('nodes_pruned', 0),
        "coverage_rate": metrics.get('coverage_rate', 0.0),
        "execution_metrics": {
            "execution_time_seconds": metrics.get('execution_time_seconds', 0.0),
            "initial_solution_size": metrics.get('initial_solution_size', 0),
            "final_solution_size": metrics.get('final_solution_size', 0),
            "qaoa_depth": qaoa_depth,
            "shots": shots,
            "backend_type": "iqm_resonance" if iqm_token and "IQM" in backend_info else "qiskit_simulator"
        },
        "backend_used": backend_info,
        "assets": assets
    }
    
    logger.info("=" * 80)
    logger.info("Solver execution completed successfully.")
    logger.info(f"Backend: {backend_info}")
    logger.info(f"Generated {len(assets)} visualization assets in 'additional_output' directory.")
    logger.info("=" * 80)
    
    return response


if __name__ == "__main__":
    """
    Example execution for local testing.
    """
    # Sample input data: 5 cycles with conflicts
    example_input = {
        "nodes": [
            {"id": "cycle_A", "weight": 100.0, "trips": ["trip_1", "trip_2"]},
            {"id": "cycle_B", "weight": 85.0, "trips": ["trip_2", "trip_3"]},
            {"id": "cycle_C", "weight": 95.0, "trips": ["trip_4", "trip_5"]},
            {"id": "cycle_D", "weight": 110.0, "trips": ["trip_1", "trip_6"]},
            {"id": "cycle_E", "weight": 70.0, "trips": ["trip_7"]},
        ],
        "edges": [
            ["cycle_A", "cycle_B"],  # Conflict: share trip_2
            ["cycle_A", "cycle_D"],  # Conflict: share trip_1
            ["cycle_B", "cycle_C"],  # Conflict: indirect via system constraints
        ]
    }
    
    # Example 1: Using Qiskit Simulator (no IQM token)
    example_params_simulator = {
        "iqm_token": None,  # Use Qiskit simulator
        "shots": 500,
        "qaoa_depth": 1
    }
    
    # Example 2: Using IQM Resonance (requires valid token)
    # Note: server_url will be automatically sanitized to remove any quantum computer suffix
    example_params_iqm = {
        "iqm_token": "your_actual_iqm_token_here",
        "quantum_computer": "emerald",
        "server_url": "https://resonance.iqm.tech/",  # Clean base URL (no /emerald suffix)
        "shots": 1024,
        "qaoa_depth": 1,
        "use_timeslot": False
    }
    
    result = run(example_input, example_params_simulator, {})
    
    print("\n" + "=" * 80)
    print("SOLUTION SUMMARY")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))
