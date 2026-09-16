 
**Quantum-Classical Hybrid Solver for Maximum Weighted Independent Set (MWIS) Problems**

 # Railway Rolling Stock Cycle Selection — QCentroid PoC

## Purpose

This repository is the **first integration PoC** for the railway rolling-stock
MWIS problem in QCentroid.

It is deliberately smaller than the final research implementation. Its goal is
to verify:

1. QCentroid accepts the dataset JSON.
2. QCentroid invokes the Python `run(input_data, solver_params, extra_arguments)` entrypoint.
3. The solver can validate the conflict graph.
4. The solver can build the MWIS/QUBO/Ising model.
5. A p=1 QAOA statevector simulation can execute inside the solver.
6. The solver returns JSON metrics.
7. Additional output assets are visible in the QCentroid job results.

QCentroid's current documentation specifies `qcentroid.py` as the solver
entrypoint and requires the `run()` signature used here. Dependencies can be
declared in `requirements.txt`, and files written under `additional_output`
are exposed as job-result assets.

## Important scope

This PoC does **not** yet connect to IQM or another external QPU.

The quantum part is a NumPy statevector implementation of **p=1 QAOA**.
This is intentional: the first test isolates the QCentroid integration from
hardware-provider configuration.

It also does not yet generate railway cycles from raw timetables. The uploaded
dataset already contains candidate cycles and their conflict graph.

## Files

- `qcentroid.py` — complete QCentroid solver.
- `requirements.txt` — NumPy dependency.
- `railway_rolling_stock_mwis_sample.json` — 12-cycle test instance.

## Input

Top level:

```json
{
  "nodes": [...],
  "edges": [...]
}
```

Each node represents a feasible candidate cycle.

The conflict graph is authoritative for the PoC.

## Mathematical model

MWIS:

```text
maximize sum_i w_i x_i
subject to x_i + x_j <= 1 for every conflict edge
x_i in {0,1}
```

QUBO:

```text
H(x) = -sum_i w_i x_i
       + lambda sum_(i,j in E) x_i x_j
```

Penalty:

```text
lambda = 4 * max_i(w_i)
```

Ising mapping:

```text
x_i = (1 - Z_i) / 2
```

The PoC then performs p=1 QAOA statevector simulation.

## Solver parameters

Optional `solver_params`:

- `seed` — integer, default `42`
- `shots` — integer, default `2000`
- `gamma_steps` — grid size, default `13`
- `beta_steps` — grid size, default `13`
- `gamma_max` — gamma search interval [-gamma_max, gamma_max], default `0.05`

For the first QCentroid test, the defaults are recommended.

## Expected result

The job should finish with:

- `status = completed`
- exact MWIS reference
- QAOA parameters
- raw QAOA solution
- pruning result
- final feasible solution
- coverage
- weight
- passenger km
- empty km
- execution time

The job should also expose:

- `result.json`
- `cycles.csv`
- `qaoa_counts.csv`
- `summary.md`

under the QCentroid job Assets section.

## Next phase

After this integration test succeeds, replace the NumPy statevector backend with
the QCentroid-supported quantum-provider route and add IQM execution. Do not
change the MWIS input contract or mathematical objective during that migration.
