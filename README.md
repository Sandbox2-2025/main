 1. **Title & Purpose**: Formalized title (**Quantum-Classical Hybrid Solver for Maximum Weighted Independent Set (MWIS) Problems**) and clear PoC integration goals for QCentroid, validating input JSON schema, entrypoint execution, circuit synthesis, and asset rendering.
2. **Technical Architecture**: Diagram outlining the 6-stage hybrid flow from JSON ingestion to graph formulation, QAOA circuit synthesis, IQM/Aer backend execution, degree-ratio pruning, and asset generation.
3. **Mathematical Formulation**: Complete derivation of cycle valuation (\\(w_i = 2 \cdot d_{\text{passenger}} - d_{\text{empty}}\\)), conflict penalty (\\(\lambda = 4 \cdot \max_i w_i\\)), Ising spin conversion (\\(h_i, J_{ij}\\)), QAOA unitaries (\\(RZ, RZZ, RX\\)), and deterministic pruning ratio (\\(r_i = w_i / \text{deg}(i)\\)).
4. **Repository Layout & Requirements**: Clear breakdown of required files (`qcentroid.py`, `requirements.txt`, `README.md`, `railway_rolling_stock_mwis_sample.json`, `additional_output/`).
5. **JSON Schemas & Interfaces**: Full input (`nodes`, `edges`) and output JSON contract examples, including `solver_params` (`iqm_token`, `quantum_computer`, `server_url`, `shots`, `qaoa_depth`).
6. **QCentroid Deployment Guide**: Step-by-step instructions for registering the solver with `No SDK` / `Classical CPU`, running *Pull & build*, launching Jobs, and inspecting execution logs and assets.
7. **Experimental Benchmarks**: Comparative performance table demonstrating \\(p=1\\) vs. \\(p=2\\) depth scaling on the 5-node DB ring corridor dataset.

 
