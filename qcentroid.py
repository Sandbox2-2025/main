"""
QCentroid Platform Solver: Railway Rolling Stock Cycle Selection via Maximum Weighted Independent Set Optimization
Proof of Concept Implementation for IQM & Deutsche Bahn Use Case

This module implements a quantum-classical hybrid solver for the Maximum Weighted Independent Set (MWIS) problem
on a conflict graph of railway rolling stock cycles, with automatic constraint pruning for NISQ-era hardware resilience.

Compliance Level: QCentroid Platform v1.0
"""

import json
import logging
import time
import os
from typing import Dict, List, Tuple, Any, Set, Optional
from dataclasses import dataclass, asdict

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
    
    def solve(self, backend=None, qaoa_p: int = 1, shots: int = 1000) -> Tuple[List[str], Dict[str, Any]]:
        """
        Execute the complete solving pipeline.
        
        Args:
            backend: Quantum backend for QAOA (None for greedy fallback)
            qaoa_p: QAOA depth parameter
            shots: Number of quantum shots
            
        Returns:
            Tuple of (selected_cycle_ids, solver_metrics)
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
        
        return final_solution, self.execution_log
    
    def visualize_conflict_graph(self, selected_cycles: List[str], output_dir: str = "additional_output"):
        """
        Render and save the conflict graph with selected/pruned nodes highlighted.
        
        Args:
            selected_cycles: List of selected cycle IDs (shown in green)
            output_dir: Directory for output images
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Layout
        pos = nx.spring_layout(self.conflict_graph, k=0.5, iterations=50, seed=42)
        
        # Node colors: green for selected, red for pruned, gray for others
        selected_set = set(selected_cycles)
        node_colors = []
        for node in self.conflict_graph.nodes():
            if node in selected_set:
                node_colors.append('#2ecc71')  # Green
            else:
                node_colors.append('#e74c3c')  # Red (pruned/not selected)
        
        # Draw edges
        nx.draw_networkx_edges(self.conflict_graph, pos, ax=ax, edge_color='#95a5a6', width=1.5, alpha=0.6)
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            self.conflict_graph, pos, ax=ax, node_color=node_colors, 
            node_size=800, edgecolors='#2c3e50', linewidths=2
        )
        
        # Draw labels with node IDs and weights
        labels = {
            node: f"{node}\n(w={self.conflict_graph.nodes[node].get('weight', 0):.2f})"
            for node in self.conflict_graph.nodes()
        }
        nx.draw_networkx_labels(self.conflict_graph, pos, labels, ax=ax, font_size=8, font_weight='bold')
        
        # Legend
        green_patch = mpatches.Patch(color='#2ecc71', label='Selected Cycles')
        red_patch = mpatches.Patch(color='#e74c3c', label='Pruned/Not Selected Cycles')
        ax.legend(handles=[green_patch, red_patch], loc='upper left', fontsize=11, framealpha=0.9)
        
        ax.set_title(
            'Railway Rolling Stock Conflict Graph\n(Maximum Weighted Independent Set Solution)',
            fontsize=14, fontweight='bold', pad=20
        )
        ax.axis('off')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "conflict_graph.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Conflict graph visualization saved to: {output_path}")
        plt.close()


def run(input_data: Dict[str, Any], solver_params: Dict[str, Any], extra_arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for the QCentroid Platform.
    
    QCentroid Contract Compliance:
    - Accepts input_data with 'nodes' and 'edges' keys
    - Reads 'iqm_token', 'shots', 'subgraph_size' from solver_params
    - Integrates with IQM Resonance or falls back to Qiskit simulator
    - Performs constraint pruning to ensure 100% feasible solution
    - Returns JSON-serializable dictionary with solution and metrics
    
    Args:
        input_data: Dictionary with keys:
            - 'nodes': List[Dict] with 'id' (str), 'weight' (float), optional 'trips' (List[str])
            - 'edges': List[List[str]] with conflict pairs
        
        solver_params: Optional parameters:
            - 'iqm_token': IQM API token for hardware access
            - 'shots': Number of quantum measurement shots (default 1000)
            - 'qaoa_depth': QAOA circuit depth p (default 1)
            - 'subgraph_size': Subgraph size for distributed solving (default None)
        
        extra_arguments: Runtime arguments injected by QCentroid platform
    
    Returns:
        Dictionary with keys:
        - 'selected_cycles': List[str] of cycle IDs in final solution
        - 'total_weight': float, sum of weights of selected cycles
        - 'nodes_pruned': int, number of cycles removed during pruning
        - 'coverage_rate': float, ratio of solution weight to total available weight
        - 'execution_metrics': dict with detailed timing and solver stats
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
    shots = solver_params.get('shots', 1000)
    qaoa_depth = solver_params.get('qaoa_depth', 1)
    subgraph_size = solver_params.get('subgraph_size')
    
    # Initialize backend
    backend = None
    if iqm_token and IQM_AVAILABLE:
        try:
            logger.info("Attempting to connect to IQM Resonance hardware...")
            provider = IQMProvider(iqm_token)
            backend = provider.get_backend()
            logger.info("✓ Connected to IQM Resonance hardware.")
        except Exception as e:
            logger.warning(f"IQM hardware unavailable ({e}). Falling back to Qiskit simulator.")
            backend = None
    elif QISKIT_AVAILABLE:
        logger.info("Using Qiskit AerSimulator (IQM token not provided or IQM unavailable).")
        backend = AerSimulator()
    else:
        logger.warning("Neither IQM nor Qiskit available. Using classical greedy solver.")
    
    # Create solver instance
    solver = RailwayRollingStockSolver(nodes, edges)
    
    # Solve
    selected_cycles, metrics = solver.solve(backend=backend, qaoa_p=qaoa_depth, shots=shots)
    
    # Generate visualization
    solver.visualize_conflict_graph(selected_cycles)
    
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
            "backend_type": "iqm_resonance" if iqm_token else "qiskit_simulator"
        }
    }
    
    logger.info("=" * 80)
    logger.info("Solver execution completed successfully.")
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
    
    example_params = {
        "iqm_token": None,  # Use simulator
        "shots": 500,
        "qaoa_depth": 1
    }
    
    result = run(example_input, example_params, {})
    
    print("\n" + "=" * 80)
    print("SOLUTION SUMMARY")
    print("=" * 80)
    print(json.dumps(result, indent=2))
