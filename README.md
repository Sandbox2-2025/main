# QCentroid: Railway Rolling Stock Cycle Selection via MWIS

**Quantum-Classical Hybrid Solver for Maximum Weighted Independent Set (MWIS) Problems**

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Architecture](#architecture)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [API Specification](#api-specification)
8. [Deployment on QCentroid Platform](#deployment-on-qcentroid-platform)
9. [Execution Examples](#execution-examples)
10. [Implementation Details](#implementation-details)
11. [NISQ Hardware Considerations](#nisq-hardware-considerations)
12. [Contributing](#contributing)

---

## Overview

QCentroid is a **hybrid classical-quantum solver** that optimizes railway rolling stock cycle selection using the **Maximum Weighted Independent Set (MWIS)** problem formulation. 

### Key Features

- **QAOA (Quantum Approximate Optimization Algorithm)** execution on IQM Resonance quantum hardware
- **Classical parameter optimization** via scipy's COBYLA or deterministic random search
- **Deterministic feasibility pruning** to guarantee 100% valid solutions even with NISQ noise
- **Conflict graph visualization** showing selected, pruned, and unselected cycles
- **Fallback mechanisms**: Greedy heuristic when quantum resources are unavailable
- **Production-ready**: Comprehensive input validation, error handling, and logging

---

## Problem Statement

### Business Challenge

Efficient railway rolling stock planning requires:

- **Minimize empty kilometers**: Reduce repositioning trips between operational cycles
- **Guarantee 100% coverage**: Every scheduled trip is covered by exactly one cycle
- **Maximize operational value**: Select the highest-value cycles without conflicts

### Mathematical Formulation: MWIS

The problem is modeled as a **Maximum Weighted Independent Set** on a conflict graph:

**Given**:
- A graph *G* = (*V*, *E*) where:
  - **V** = {cycles with id, weight, and optional trip coverage data}
  - **E** = {pairs of cycles sharing at least one trip (conflict edges)}

**Objective**:
```
maximize: Σ w(v) for v ∈ S
subject to: For all (u,v) ∈ E: ¬(u ∈ S ∧ v ∈ S)  [no conflicts in S]
where S ⊆ V is the independent set
```

### Hamiltonian Encoding

The MWIS is cast as a **minimization problem**:

```
H(x) = -Σ w_i·x_i + P·Σ x_i·x_j   for (i,j) ∈ E

where:
  x_i ∈ {0,1}  (selection binary variable)
  w_i ≥ 0      (node weight)
  P > max(w_i)  (penalty strictly larger than largest weight)
```

The penalty ensures that selecting both endpoints of an edge is never beneficial in the classical objective.

---

## Architecture

### Execution Pipeline

```
Input JSON (nodes + edges)
    ↓
[1] Conflict Graph Construction (NetworkX)
    - Validate input data
    - Build undirected graph with weighted nodes
    ↓
[2] QAOA Solver (Quantum or Classical)
    - Compute Hamiltonian coefficients (h_i, J_ij)
    - Optimize QAOA parameters classically (scipy/random search)
    - Build QAOA circuit with optimized parameters
    - Execute on IQM Resonance or local AerSimulator
    - Extract best measured bitstring
    ↓
[3] Feasibility Pruning (Deterministic)
    - While conflicting edges exist in selected solution:
        * Find first edge (u,v) both selected
        * Remove lower-weight endpoint
    - Guarantee independent set property
    ↓
[4] Metrics & Visualization
    - Calculate trip coverage, weight sums, empty-km totals
    - Generate conflict_graph.png with color-coded nodes
    ↓
Output JSON (selected_cycles + metrics + assets)
```

### Core Components

#### 1. **RailwayRollingStockSolver**

Main orchestrator. Responsibilities:
- Validate input nodes and edges
- Build conflict graph (NetworkX)
- Instantiate QAOA solver
- Orchestrate pruning phase
- Compute final metrics
- Trigger visualization

**Key methods**:
- `solve(backend, qaoa_p, shots, penalty)` → (selected_cycles, metrics, pruned_nodes)
- `_build_conflict_graph()` → nx.Graph
- `_calculate_metrics(selected_cycles)` → Dict[str, Any]

#### 2. **QAOAMWISSolver**

QAOA implementation with fallback logic. Responsibilities:
- Hamiltonian coefficient computation
- Parameter optimization (classical)
- QAOA circuit construction
- Solution extraction from measurement results

**Key methods**:
- `solve_qaoa(p, shots)` → Dict[best_bitstring, counts, gamma, beta, ...]
- `solve_greedy()` → Dict[greedy solution]
- `_optimize_parameters(p, shots)` → (gamma, beta, training_cost)
- `_build_qaoa_circuit(p, gamma_values, beta_values, measure=True)` → QuantumCircuit
- `classical_objective(bits)` → float [MWIS cost]

#### 3. **MWISPruner**

Iterative conflict elimination. Responsibilities:
- Detect edges with both endpoints selected
- Remove lower-weight node
- Track pruning history and statistics

**Key methods**:
- `prune(selected_cycles)` → (feasible_solution, num_pruned)
- `get_pruning_stats()` → Dict[pruning_iterations, pruned_nodes]

#### 4. **IQMBackendManager**

Manages IQM Resonance connection and execution. Responsibilities:
- Authenticate with IQM token
- Validate server URL and quantum computer selection
- Transpile and execute circuits on hardware

**Key methods**:
- `__init__(iqm_token, quantum_computer, server_url)`
- `get_backend()` → IQM backend object
- `run_circuit(circuit, shots)` → Dict[counts, job_id, backend_name, ...]

#### 5. **VisualizationAssetGenerator**

Generates conflict graph visualization. Responsibilities:
- Render NetworkX graph with spring layout
- Color-code nodes: green (selected), red (pruned), gray (unselected)
- Export PNG with high DPI

**Key methods**:
- `generate_conflict_graph_visualization(conflict_graph, selected_cycles, pruned_nodes)` → str [filepath]

#### 6. **URLSanitizer**

Utility for IQM URL validation. Responsibilities:
- Normalize URLs (trailing slashes, known QPU names)
- Validate URL structure

---

## Repository Structure

```
Sandbox2-2025/main/
├── qcentroid.py                    # Main solver implementation
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── additional_output/
    └── conflict_graph.png          # Generated visualization (runtime)
```

### File Descriptions

| File | Purpose |
|------|---------|
| **qcentroid.py** | Complete solver implementation with all classes and entry point `run()` function |
| **requirements.txt** | Python package dependencies (iqm-client, qiskit, networkx, matplotlib, scipy) |
| **additional_output/** | Auto-created directory for visualization and asset outputs |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) IQM Resonance credentials for quantum hardware execution

### Local Setup

```bash
# Clone repository
git clone https://github.com/Sandbox2-2025/main.git
cd main

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt Content

```
networkx>=2.6
numpy>=1.21
matplotlib>=3.4
qiskit>=0.37.0
qiskit-aer>=0.10.0
scipy>=1.7.0
iqm-client>=14.0
iqm-qiskit-integration>=0.1.0
```

---

## Usage

### Basic Example (Local Execution)

```python
from qcentroid import run

# Define input data
input_data = {
    "nodes": [
        {"id": "C01", "weight": 85.0, "trips": ["T01", "T02"], "empty_km": 100},
        {"id": "C02", "weight": 70.0, "trips": ["T02", "T03"], "empty_km": 250},
        {"id": "C03", "weight": 90.0, "trips": ["T04", "T05"], "empty_km": 50},
    ],
    "edges": [["C01", "C02"], ["C02", "C03"]]
}

# Configure solver parameters
solver_params = {
    "iqm_token": None,  # None → use AerSimulator
    "quantum_computer": "emerald",
    "server_url": "https://resonance.iqm.tech/",
    "shots": 512,
    "qaoa_depth": 1,
    "penalty": None  # None → auto-compute as 1.25 * max_weight
}

# Execute solver
result = run(input_data, solver_params, {})

# Print results
import json
print(json.dumps(result, indent=2, default=str))
```

### Execution from Command Line

```bash
python qcentroid.py
```

This runs the built-in example with 5 cycles and 3 conflicts, producing:
- Console logs with execution trace
- `additional_output/conflict_graph.png` visualization
- JSON output with selected cycles and metrics

---

## API Specification

### Function Signature

```python
def run(
    input_data: Dict[str, Any],
    solver_params: Dict[str, Any],
    extra_arguments: Dict[str, Any],
) -> Dict[str, Any]
```

### Input: `input_data`

**Structure**:
```json
{
  "nodes": [
    {
      "id": "C01",
      "weight": 85.0,
      "trips": ["T01", "T02"],
      "empty_km": 100.0,
      "passenger_km": 8000.0,
      "operating_cost": 1200.0
    }
  ],
  "edges": [
    ["C01", "C02"]
  ]
}
```

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `nodes` | List[Dict] | Yes | List of cycle candidates |
| `nodes[].id` | String | Yes | Unique cycle identifier |
| `nodes[].weight` | Float | Yes | Operational weight (must be > 0) |
| `nodes[].trips` | List[String] | No | Trip IDs covered by cycle |
| `nodes[].empty_km` | Float | No | Empty repositioning kilometers |
| `nodes[].passenger_km` | Float | No | Passenger-carrying kilometers |
| `nodes[].operating_cost` | Float | No | Operating cost for cycle |
| `edges` | List[List[String]] | Yes | Conflict pairs: `[[cycle_id_1, cycle_id_2], ...]` |

### Input: `solver_params`

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `iqm_token` | String | None | No | IQM Resonance authentication token |
| `quantum_computer` | String | "emerald" | No | Target QPU: emerald, sirius, garnet, sapphire, ruby, diamond |
| `server_url` | String | "https://resonance.iqm.tech/" | No | IQM server base URL |
| `shots` | Integer | 1024 | No | Circuit execution shots |
| `qaoa_depth` | Integer | 1 | No | QAOA circuit depth (p parameter) |
| `penalty` | Float | None | No | Conflict penalty (if None: auto-computed as 1.25 × max_weight) |

### Output

**Structure**:
```json
{
  "selected_cycles": ["C01", "C03"],
  "total_weight": 175.0,
  "nodes_pruned": 1,
  "coverage_rate": 0.75,
  "coverage_percent": 75.0,
  "covered_trips": 4,
  "scheduled_trips": 5,
  "total_empty_km": 150.0,
  "total_passenger_km": 17000.0,
  "total_operating_cost": 2380.0,
  "is_feasible": true,
  "execution_metrics": {
    "execution_time_seconds": 1.234,
    "initial_solution_size": 3,
    "final_solution_size": 2,
    "qaoa_depth": 1,
    "shots": 512,
    "penalty": 106.25,
    "training_cost": -85.5,
    "job_id": "iqm-job-12345"
  },
  "backend_used": "IQM Resonance (emerald)",
  "assets": {
    "conflict_graph_png": "additional_output/conflict_graph.png"
  },
  "pruning_stats": {
    "nodes_pruned": 1,
    "pruned_nodes": ["C02"]
  }
}
```

**Output Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `selected_cycles` | List[String] | IDs of cycles in final solution (no conflicts) |
| `total_weight` | Float | Sum of weights of selected cycles |
| `nodes_pruned` | Integer | Count of cycles removed during pruning |
| `coverage_rate` | Float | Ratio of covered trips to total trips (0.0–1.0, or null) |
| `coverage_percent` | Float | Coverage as percentage (0–100, or null) |
| `covered_trips` | Integer | Number of distinct trips covered |
| `scheduled_trips` | Integer | Total number of scheduled trips |
| `total_empty_km` | Float | Sum of empty_km for selected cycles (or null) |
| `total_passenger_km` | Float | Sum of passenger_km for selected cycles (or null) |
| `total_operating_cost` | Float | Sum of operating_cost for selected cycles (or null) |
| `is_feasible` | Boolean | True if no conflicting edges exist in solution |
| `execution_metrics` | Dict | Timing, parameter, and job information |
| `backend_used` | String | Backend identifier (IQM/AerSimulator/Greedy) |
| `assets` | Dict | Generated visualization file paths |
| `pruning_stats` | Dict | Pruning iteration count and node list |

---

## Deployment on QCentroid Platform

### Step 1: Repository Connection

1. **Generate Deploy Key** in QCentroid Settings → Repository Access
2. **Add to GitHub**: Settings → Deploy Keys → Paste public key → Save
3. **Configure Solver**:
   - Navigate to Solvers panel
   - Select "Add New Solver"
   - Repository URL: `git@github.com:Sandbox2-2025/main.git`
   - Branch: `main`
   - Click "Validate & Connect"

### Step 2: Build & Validation

QCentroid automatically:

```bash
git clone git@github.com:Sandbox2-2025/main.git
pip install -r requirements.txt
python -c "from qcentroid import run; print('✓ Solver imported successfully')"
```

### Step 3: Submit Job

1. Navigate to **Jobs** → **New Job**
2. Select solver: "Railway Rolling Stock MWIS"
3. Upload `input_data.json`:

```json
{
  "nodes": [
    {"id": "cycle_1", "weight": 100, "trips": ["t1", "t2"], "empty_km": 50},
    {"id": "cycle_2", "weight": 90, "trips": ["t2", "t3"], "empty_km": 75},
    {"id": "cycle_3", "weight": 110, "trips": ["t4"], "empty_km": 25}
  ],
  "edges": [["cycle_1", "cycle_2"]]
}
```

4. Configure `solver_params.json`:

```json
{
  "iqm_token": "your-iqm-token-here",
  "shots": 500,
  "qaoa_depth": 2
}
```

5. Click **Submit Job**

### Step 4: Monitor & Retrieve Results

- **Status Panel**: Real-time job state (pending → running → completed/failed)
- **Logs**: Stream from `qcentroid-user-log` logger
- **Results**: JSON output with `selected_cycles` and metrics
- **Assets**: Download `conflict_graph.png`

---

## Execution Examples

### Example 1: Simple 3-Cycle Problem (Local)

```python
result = run(
    input_data={
        "nodes": [
            {"id": "A", "weight": 10.0, "trips": ["T1"]},
            {"id": "B", "weight": 20.0, "trips": ["T1", "T2"]},
            {"id": "C", "weight": 15.0, "trips": ["T2", "T3"]},
        ],
        "edges": [["A", "B"], ["B", "C"]]
    },
    solver_params={"shots": 256, "qaoa_depth": 1},
    extra_arguments={}
)
# Expected: Select cycles A and C (weight = 25.0, no conflicts)
```

### Example 2: Large Problem with IQM Hardware

```python
result = run(
    input_data={...},  # 20+ cycles, complex conflicts
    solver_params={
        "iqm_token": os.environ["IQM_TOKEN"],
        "quantum_computer": "emerald",
        "shots": 1024,
        "qaoa_depth": 2,
        "penalty": 150.0
    },
    extra_arguments={}
)
# Optimized QAOA parameters: executed on real hardware
# Deterministic pruning: ensures 100% feasibility
```

### Example 3: Fallback to Greedy (No Quantum)

```python
result = run(
    input_data={...},
    solver_params={"iqm_token": None},  # Triggers fallback
    extra_arguments={}
)
# Uses Qiskit AerSimulator if available; otherwise pure greedy
```

---

## Implementation Details

### QAOA Parameter Optimization

The solver optimizes QAOA parameters **classically** before executing on hardware:

1. **Initialization**: Random uniform sampling
   - γ ∈ [0, 2π]
   - β ∈ [0, π]

2. **Optimization Method**:
   - **scipy available**: COBYLA (constrained optimization)
   - **scipy unavailable**: Deterministic random search (30 iterations)

3. **Objective Function**:
   - Simulate QAOA circuit with candidate parameters
   - Compute expected MWIS cost from measurement statistics
   - Minimize expected cost

4. **Evaluation Shots**: max(128, min(shots, 512))

### Hamiltonian Coefficients

From `H(x) = -Σ w_i·x_i + P·Σ x_i·x_j`, using `x = (1 - Z_i)/2`:

```
H = constant + Σ h_i·Z_i + Σ J_ij·Z_i·Z_j

where:
  h_i = w_i/2 - (P·degree(i))/4
  J_ij = P/4 (for each edge)
```

### QAOA Circuit Layers (p ≥ 1)

Per layer:
1. **Problem Hamiltonian**: RZ(2γ·h_i) and RZZ(2γ·J_ij) gates
2. **Mixer Hamiltonian**: Standard X mixer RX(2β) on all qubits

### Deterministic Pruning Algorithm

```
Input: selected_cycles (may have conflicts), conflict_graph
Output: feasible_solution (independent set)

while ∃ edge (u,v) where u,v ∈ selected_cycles:
    find_conflict(u, v)
    weight_u ← graph.nodes[u]['weight']
    weight_v ← graph.nodes[v]['weight']
    
    if weight_u ≤ weight_v:
        remove(u)
    else:
        remove(v)

return selected_cycles (now conflict-free)
```

**Key Property**: Iteratively removing minimum-weight endpoints preserves solution quality.

### Greedy Fallback

When quantum hardware is unavailable:

```python
selected = []
remaining = all_nodes

while remaining:
    best = argmax(weight[n] for n ∈ remaining)
    selected.append(best)
    remaining -= neighbors(best)
    remaining.discard(best)

return selected  # Independent set (guaranteed)
```

---

## NISQ Hardware Considerations

The solver is **optimized for NISQ devices** (limited qubits, short coherence times, gate errors):

### Challenge: NISQ Noise

- QAOA circuits executed on IQM hardware may produce measurements with conflicting edges
- Short coherence windows require shallow circuits (p ≤ 2)
- Gate errors and shot noise introduce variability in measured bitstrings

### Solution: Deterministic Feasibility Layer

1. **Acknowledgment**: NISQ solutions may violate independent set constraints
2. **Pruning Strategy**: Iteratively remove conflicting nodes
3. **Guarantee**: Final solution is **100% feasible** (no conflicts)
4. **Quality Trade-off**: Minimal weight loss since only minimum-weight nodes are removed

### Recommendations

- **qaoa_depth**: p = 1 or 2 (deeper circuits accumulate more errors)
- **shots**: ≥ 512 for stable statistics (1024 ideal)
- **penalty**: Leave as default (1.25 × max_weight) unless tuning for specific hardware

---

## Contributing

### Reporting Issues

Open a **GitHub Issue** with:
- Problem description
- Input data (if shareable)
- Expected vs. actual output
- Execution logs (full traceback)

### Submitting Changes

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-improvement`
3. Commit with clear messages
4. Push and open **Pull Request**
5. Link related issues

### Code Style

- Python 3.9+ type hints
- Docstrings for all public methods
- Comprehensive logging (logger = logging.getLogger("qcentroid-user-log"))
- Unit tests for new features (if applicable)

---

## License

MIT License. See `LICENSE` file for details.

---

## References

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm"
2. IQM Technology - [Resonance Hardware](https://www.iqmtechnology.com/resonance)
3. Qiskit Documentation - [Circuit Execution](https://qiskit.org/documentation/)
4. NetworkX - [Graph Algorithms](https://networkx.org/)
5. Maximum Weighted Independent Set - NP-complete optimization problem

---

**Version**: 2.0  
**Last Updated**: 2026-09-12  
**Platform**: QCentroid Platform v1.0 + IQM Resonance  
**Status**: Production Ready
