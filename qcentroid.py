"""Punto de entrada único de Q-Centroid para el benchmark ferroviario.

El mismo JSON se usa para ambos algoritmos. Se acepta ``algorithm`` en
solver_params o extra_arguments. Para compatibilidad con jobs existentes,
un token IQM sin ``algorithm`` selecciona IQM; CPU exige ``algorithm=cpu``.
"""

import logging

from solver_cpu import run as run_cpu_solver
from solver_iqm import run as run_iqm_solver


logger = logging.getLogger("qcentroid-user-log")


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    params = solver_params or {}
    extra = extra_arguments or {}
    configured = params.get("algorithm")
    provided = extra.get("algorithm")
    if configured and provided and configured != provided:
        raise ValueError("algorithm difiere entre solver_params y extra_arguments")
    algorithm = configured or provided
    if algorithm is None and (params.get("iqm_token") or extra.get("iqm_token")):
        algorithm = "iqm"
        logger.info("Seleccionando IQM por presencia de iqm_token")

    if algorithm == "cpu":
        logger.info("Iniciando solver CPU exacto")
        result = run_cpu_solver(input_data, params, extra)
    elif algorithm == "iqm":
        logger.info("Iniciando solver QAOA (backend IQM por defecto)")
        result = run_iqm_solver(input_data, params, extra)
    else:
        raise ValueError("Indica algorithm='cpu' o 'iqm', o facilita iqm_token para IQM")

    return {**result, "algorithm_requested": algorithm}
