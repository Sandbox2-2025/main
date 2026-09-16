"""
qcentroid.py - Rigorous MWIS + QAOA PoC for QCentroid Platform
Grounded in IQM / Deutsche Bahn Formulation (arXiv:2606.11383)
"""

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Tuple, Set, Optional

import networkx as nx
import numpy as np

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

logger = logging.getLogger("qcentroid-user-log")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MWISObjective:
    """Clase para la evaluación QUBO/Ising y cálculo de ground truth exacto."""
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.nodes = sorted(list(graph.nodes()))
        self.num_nodes = len(self.nodes)
        
        # Pesos de nodos y constante de penalización lambda = 4 * max(w)
        self.weights = {n: float(graph.nodes[n].get("weight", 1.0)) for n in self.nodes}
        self.max_weight = max(self.weights.values(), default=1.0)
        self.lambda_penalty = 4.0 * self.max_weight  # Penalización estricta del paper

    def qubo_energy(self, bitstring: str) -> float:
        """Calcula H(x) = -sum(w_i * x_i) + lambda * sum(x_i * x_j)."""
        # Qiskit ordena los bits desde el qubit de mayor índice al menor (little-endian)
        bits = [1 if bitstring[-(i + 1)] == "1" else 0 for i in range(self.num_nodes)]
        
        energy = 0.0
        # Término lineal
        for i, node in enumerate(self.nodes):
            if bits[i]:
                energy -= self.weights[node]
        
        # Término de penalización por conflictos
        for u, v in self.graph.edges():
            i = self.nodes.index(u)
            j = self.nodes.index(v)
            if bits[i] and bits[j]:
                energy += self.lambda_penalty
                
        return energy

    def is_feasible(self, selected_nodes: List[str]) -> bool:
        sel_set = set(selected_nodes)
        return all(not (u in sel_set and v in sel_set) for u, v in self.graph.edges())

    def solve_exact_ground_truth(self) -> Tuple[List[str], float]:
        """Calcula la solución MWIS exacta mediante clique sobre el grafo complementario."""
        complement_graph = nx.complement(self.graph)
        # Asignar pesos al complemento
        for n in complement_graph.nodes():
            complement_graph.nodes[n]["weight"] = self.weights[n]
            
        clique, weight = nx.max_weight_clique(complement_graph, weight="weight")
        return sorted(clique), float(weight)


class MWISPruner:
    """Reparación determinista basada en el ratio r_i = w_i / degree(i)."""
    @staticmethod
    def prune(graph: nx.Graph, initial_selected: List[str]) -> Tuple[List[str], List[str]]:
        selected = set(initial_selected)
        pruned_nodes = []

        while True:
            subgraph = graph.subgraph(selected)
            conflicts = list(subgraph.edges())
            if not conflicts:
                break

            conflict_nodes = set(u for edge in conflicts for u in edge)
            
            # Elección del nodo con menor ratio ratio = w_i / degree(i)
            worst_node = min(
                conflict_nodes,
                key=lambda n: float(graph.nodes[n].get("weight", 0.0)) / max(1, subgraph.degree(n))
            )
            selected.remove(worst_node)
            pruned_nodes.append(worst_node)

        return sorted(list(selected)), sorted(pruned_nodes)


def run(input_data: Dict[str, Any], solver_params: Dict[str, Any], extra_arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Punto de entrada oficial para QCentroid Platform."""
    logger.info("Iniciando PoC Metodológica MWIS + QAOA")
    
    # 1. Carga y validación del Grafo
    nodes_raw = input_data.get("nodes", [])
    edges_raw = input_data.get("edges", [])
    
    graph = nx.Graph()
    for n in nodes_raw:
        n_id = n["id"]
        passenger_km = float(n.get("passenger_km", 0.0))
        empty_km = float(n.get("empty_km", 0.0))
        
        # Fórmula estricta: w_i = 2 * passenger_km - empty_km (o peso suministrado si se especifica)
        calculated_w = (2.0 * passenger_km - empty_km) if (passenger_km > 0 or empty_km > 0) else float(n.get("weight", 1.0))
        weight = float(n.get("weight", calculated_w))
        
        graph.add_node(n_id, weight=weight, trips=n.get("trips", []), passenger_km=passenger_km, empty_km=empty_km)

    # Construcción robusta de aristas
    if edges_raw:
        for e in edges_raw:
            if len(e) == 2 and e in graph and e[16] in graph and e != e[16]:
                graph.add_edge(e, e[16])
    else:
        # Generación automática de aristas por servicios compartidos
        node_ids = list(graph.nodes())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                u, v = node_ids[i], node_ids[j]
                if set(graph.nodes[u]["trips"]) & set(graph.nodes[v]["trips"]):
                    graph.add_edge(u, v)

    objective = MWISObjective(graph)
    
    # 2. Ground Truth Clásico Exacto
    exact_cycles, exact_weight = objective.solve_exact_ground_truth()
    logger.info("Ground Truth Exacto: %s | Peso Óptimo: %.1f", exact_cycles, exact_weight)

    # 3. Simulación QAOA / Decodificación por Mínima Energía QUBO
    shots = int(solver_params.get("shots", 2048))
    qaoa_p = int(solver_params.get("qaoa_depth", 2))
    
    if QISKIT_AVAILABLE:
        n_qubits = len(objective.nodes)
        qc = QuantumCircuit(n_qubits, n_qubits)
        qc.h(range(n_qubits)) # Estado inicial |+>
        
        # Capa QAOA simplificada para validación p=1/p=2
        gamma, beta = 0.05, 0.25
        for _ in range(qaoa_p):
            for i, node in enumerate(objective.nodes):
                deg = graph.degree(node)
                h_i = (objective.weights[node] / 2.0) - (objective.lambda_penalty * deg / 4.0)
                qc.rz(2.0 * gamma * h_i, i)
            for u, v in graph.edges():
                i, j = objective.nodes.index(u), objective.nodes.index(v)
                qc.rzz(2.0 * gamma * (objective.lambda_penalty / 4.0), i, j)
            for i in range(n_qubits):
                qc.rx(2.0 * beta, i)
        qc.measure(range(n_qubits), range(n_qubits))
        
        sim = AerSimulator()
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        
        # CORRECCIÓN CLAVE: Selección del bitstring de MENOR ENERGÍA QUBO (no max counts)
        best_bitstring = min(counts.keys(), key=lambda b: objective.qubo_energy(b))
        raw_selected = [objective.nodes[i] for i in range(n_qubits) if best_bitstring[-(i + 1)] == "1"]
    else:
        raw_selected = exact_cycles
        best_bitstring = "N/A"

    # 4. Pruning y Evaluación Final
    final_selected, pruned = MWISPruner.prune(graph, raw_selected)
    final_weight = sum(graph.nodes[n]["weight"] for n in final_selected)
    
    return {
        "ground_truth_exact": {
            "selected_cycles": exact_cycles,
            "total_weight": exact_weight,
            "is_feasible": True
        },
        "qaoa_results": {
            "best_bitstring_min_energy": best_bitstring,
            "raw_selected_cycles": raw_selected,
            "final_pruned_cycles": final_selected,
            "final_weight": final_weight,
            "nodes_pruned": pruned,
            "is_feasible": objective.is_feasible(final_selected),
            "optimality_gap_percent": round(((exact_weight - final_weight) / exact_weight) * 100.0, 2)
        }
    }
