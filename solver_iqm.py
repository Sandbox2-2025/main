"""Módulo IQM QAOA llamado por el qcentroid.py de la raíz.

Recibe el JSON completo en run(input_data, solver_params, extra_arguments).
IQM es el backend por defecto; Aer solo si se solicita use_iqm=False.
La validación y el postprocesado están incluidos aquí, sin módulos adicionales.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import json


def dataset_fingerprint(data):
    """Huella de la instancia completa compartida por los dos solvers."""
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def route_table(data):
    routes = {}
    for route in data["routes"]:
        a, b = route["from"], route["to"]
        if a == b or (a, b) in routes or (b, a) in routes:
            raise ValueError("Relación duplicada o inválida")
        routes[a, b] = routes[b, a] = (int(route["km"]), int(route["empty_hours"]))
    return routes


def move(routes, a, b):
    if a == b:
        return 0, 0
    if (a, b) not in routes:
        raise ValueError(f"Falta la relación {a} / {b}")
    return routes[a, b]


def evaluate_cycle(data, trip_ids):
    """Busca una inserción de mantenimiento y devuelve la de menos km vacíos.

    Modelo deliberado: salida a t=0 desde el origen del primer servicio,
    una parada de mantenimiento, transferencias directas, cierre a t<=48.
    """
    trips = {t["id"]: t for t in data["services"]}
    if not trip_ids or len(set(trip_ids)) != len(trip_ids):
        raise ValueError("Ciclo sin servicios o con servicios repetidos")
    if any(t not in trips for t in trip_ids):
        raise ValueError("Servicio desconocido en ciclo")
    seq = sorted((trips[t] for t in trip_ids), key=lambda t: (t["departure"], t["id"]))
    ids = [t["id"] for t in seq]
    if ids != list(trip_ids):
        raise ValueError("Los servicios del ciclo deben seguir el orden horario")
    routes = route_table(data)
    home = seq[0]["from"]
    station = data["maintenance_station"]
    stop = int(data["maintenance_hours"])
    horizon = int(data["horizon_hours"])
    limit = int(data["maximum_cycle_km"])
    best = None

    # -1: antes del primer servicio; i: tras el servicio i;
    # len(seq): al final (antes de cerrar el ciclo).
    for m in range(-1, len(seq) + 1):
        now, place, empty = 0, home, 0
        actions = []
        feasible = True
        for i, trip in enumerate(seq):
            if i == 0 and m == -1:
                km, h = move(routes, place, station)
                if km:
                    actions.append({"type": "empty", "from": place, "to": station,
                                    "start": now, "end": now + h, "km": km})
                now += h
                empty += km
                place = station
                actions.append({"type": "maintenance", "station": place,
                                "start": now, "end": now + stop})
                now += stop
            if place != trip["from"]:
                km, h = move(routes, place, trip["from"])
                actions.append({"type": "empty", "from": place, "to": trip["from"],
                                "start": now, "end": now + h, "km": km})
                now += h
                empty += km
                place = trip["from"]
            if now > trip["departure"]:
                feasible = False
                break
            now = trip["arrival"]
            place = trip["to"]
            actions.append({"type": "service", "id": trip["id"],
                            "from": trip["from"], "to": place,
                            "start": trip["departure"], "end": now, "km": trip["km"]})
            if m == i:
                if place != station:
                    km, h = move(routes, place, station)
                    actions.append({"type": "empty", "from": place, "to": station,
                                    "start": now, "end": now + h, "km": km})
                    now += h
                    empty += km
                    place = station
                actions.append({"type": "maintenance", "station": place,
                                "start": now, "end": now + stop})
                now += stop
        if not feasible:
            continue
        if m == len(seq):
            if place != station:
                km, h = move(routes, place, station)
                actions.append({"type": "empty", "from": place, "to": station,
                                "start": now, "end": now + h, "km": km})
                now += h
                empty += km
                place = station
            actions.append({"type": "maintenance", "station": place,
                            "start": now, "end": now + stop})
            now += stop
        if place != home:
            km, h = move(routes, place, home)
            actions.append({"type": "empty", "from": place, "to": home,
                            "start": now, "end": now + h, "km": km})
            now += h
            empty += km
        loaded = sum(t["km"] for t in seq)
        if now > horizon or loaded + empty > limit:
            continue
        result = {"trips": ids, "home": home, "passenger_km": loaded,
                  "empty_km": empty, "total_km": loaded + empty,
                  "return_hour": now, "maintenance_after_index": m,
                  "actions": actions}
        if best is None or (empty, now) < (best["empty_km"], best["return_hour"]):
            best = result
    return best


def validate_dataset(data):
    ids = [t["id"] for t in data["services"]]
    if len(ids) != 10 or len(set(ids)) != len(ids):
        raise ValueError("Se esperan diez servicios únicos")
    route_table(data)
    for t in data["services"]:
        if not (0 <= t["departure"] < t["arrival"] <= data["horizon_hours"]):
            raise ValueError(f"Horario inválido: {t['id']}")
    candidates = data["nodes"]
    if len(candidates) != 20 or len({c["id"] for c in candidates}) != 20:
        raise ValueError("Se esperan veinte ciclos únicos")
    for c in candidates:
        actual = evaluate_cycle(data, c["trips"])
        if actual is None:
            raise ValueError(f"Ciclo inviable: {c['id']}")
        for field in ("home", "passenger_km", "empty_km", "total_km", "return_hour",
                      "maintenance_after_index", "actions"):
            if c[field] != actual[field]:
                raise ValueError(f"{c['id']}: dato incoherente {field}")
        if c["weight"] != 2 * c["passenger_km"] - c["empty_km"]:
            raise ValueError(f"{c['id']}: peso incoherente")
    expected = conflict_edges(candidates)
    if data["edges"] != expected:
        raise ValueError("Aristas de conflicto incoherentes con los servicios")
    if not set(ids).issubset({t for c in candidates for t in c["trips"]}):
        raise ValueError("Algún servicio no tiene un ciclo candidato")
    return True


def conflict_edges(nodes):
    return [[a["id"], b["id"]] for a, b in combinations(nodes, 2)
            if set(a["trips"]) & set(b["trips"])]


def assess(data, selected_ids):
    by_id = {c["id"]: c for c in data["nodes"]}
    if len(selected_ids) != len(set(selected_ids)) or any(c not in by_id for c in selected_ids):
        raise ValueError("Ciclos seleccionados duplicados o desconocidos")
    selected = [by_id[c] for c in selected_ids]
    counts = Counter(t for c in selected for t in c["trips"])
    all_ids = {t["id"] for t in data["services"]}
    duplicates = sorted(t for t, n in counts.items() if n > 1)
    uncovered = sorted(all_ids - set(counts))
    return {"selected_cycles": sorted(selected_ids),
            "valid_no_overlap": not duplicates, "full_coverage": not duplicates and not uncovered,
            "covered_trips": len(all_ids) - len(uncovered),
            "coverage_percent": 100 * (len(all_ids) - len(uncovered)) / len(all_ids),
            "uncovered_trips": uncovered, "duplicated_trips": duplicates,
            "total_passenger_km": sum(c["passenger_km"] for c in selected),
            "total_empty_km": sum(c["empty_km"] for c in selected),
            "total_weight": sum(c["weight"] for c in selected),
            "number_of_trains": len(selected)}



import time


def _prune(raw_ids, weights, adjacency):
    selected = set(raw_ids)
    while True:
        degrees = {i: len(adjacency[i] & selected) for i in selected}
        bad = [i for i in selected if degrees[i]]
        if not bad:
            return sorted(selected)
        # Se conserva preferentemente peso/grado alto. Empate por ID.
        remove = min(bad, key=lambda i: (weights[i] / (degrees[i] + 1), i))
        selected.remove(remove)


def _energy(ids, weights, edges, penalty):
    chosen = set(ids)
    return -sum(weights[i] for i in chosen) + penalty * sum(a in chosen and b in chosen for a, b in edges)


def _decode(bitstring, ids):
    bits = bitstring.replace(" ", "")[::-1]
    if len(bits) != len(ids) or set(bits) - {"0", "1"}:
        raise ValueError("Bitstring incompatible con la lista ordenada de ciclos")
    return [ids[i] for i, bit in enumerate(bits) if bit == "1"]


def _circuit(ids, weights, edges, penalty, depth, gamma, beta):
    from qiskit import QuantumCircuit

    n = len(ids)
    position = {id_: i for i, id_ in enumerate(ids)}
    degree = Counter(a for a, _ in edges) + Counter(b for _, b in edges)
    scale = penalty  # Multiplicamos H por 1/scale para ángulos manejables.
    circuit = QuantumCircuit(n, n)
    for i in range(n):
        circuit.h(i)
    for _ in range(depth):
        for id_ in ids:
            a = (weights[id_] / 2 - penalty * degree[id_] / 4) / scale
            circuit.rz(2 * gamma * a, position[id_])
        for u, v in edges:
            circuit.rzz(gamma * penalty / (2 * scale), position[u], position[v])
        for i in range(n):
            circuit.rx(2 * beta, i)
    circuit.measure(range(n), range(n))
    return circuit


def _backend(extra):
    if not extra.get("use_iqm", True):
        from qiskit_aer import AerSimulator
        return AerSimulator(), "AerSimulator"
    token = extra.get("iqm_token")
    if not token:
        raise ValueError("Falta iqm_token. Para simulación explícita use_iqm=false.")
    from iqm.qiskit_iqm import IQMProvider
    from urllib.parse import urlsplit
    import os
    url = str(os.getenv("IQM_SERVER_URL", "https://resonance.iqm.tech/")).strip()
    if urlsplit(url).hostname in {"cocos.resonance.meetiqm.com", "resonance.meetiqm.com"}:
        url = "https://resonance.iqm.tech/"
    if urlsplit(url).scheme != "https":
        raise ValueError("IQM_SERVER_URL debe usar https")
    provider = IQMProvider(url.rstrip("/") + "/",
                           quantum_computer=str(extra.get("quantum_computer", "emerald")),
                           token=token)
    return provider.get_backend(), "IQM Resonance"


def evaluate_counts(data, counts):
    """Evaluación pura, utilizable para verificar el decodificador sin QPU."""
    validate_dataset(data)
    ids = [n["id"] for n in data["nodes"]]
    weights = {n["id"]: n["weight"] for n in data["nodes"]}
    edges = [tuple(e) for e in data["edges"]]
    adjacency = {id_: set() for id_ in ids}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    penalty = 4 * max(abs(w) for w in weights.values())
    candidates = []
    for bitstring, shots in counts.items():
        raw = _decode(bitstring, ids)
        repaired = _prune(raw, weights, adjacency)
        report = assess(data, repaired)
        candidates.append((report, bitstring, int(shots), _energy(raw, weights, edges, penalty), raw))
    if not candidates:
        raise ValueError("No hay muestras QAOA")
    best = max(candidates, key=lambda row: (
        row[0]["full_coverage"], row[0]["total_weight"],
        -row[0]["number_of_trains"], row[2]))
    raw_lowest = min(candidates, key=lambda row: (row[3], -row[2]))
    report, bits, frequency, energy, raw = best
    return {**report, "bitstring_selected": bits, "selected_bitstring_shots": frequency,
            "raw_selected_cycles": raw, "raw_qubo_energy": energy,
            "lowest_raw_energy_bitstring": raw_lowest[1],
            "unique_bitstrings": len(counts), "penalty": penalty,
            "sample_selection_rule": "full coverage first, then MWIS weight after conflict pruning"}


def run(input_data, solver_params=None, extra_arguments=None):
    """Resuelve con QAOA; acepta parámetros del job en cualquiera de los mapas."""
    params = solver_params or {}
    extra = dict(extra_arguments or {})
    for name in ("iqm_token", "quantum_computer", "use_iqm"):
        if name not in extra and name in params:
            extra[name] = params[name]
    validate_dataset(input_data)
    from qiskit import transpile

    t0 = time.perf_counter()
    depth = int(params.get("qaoa_depth", extra.get("qaoa_depth", 1)))
    shots = int(params.get("shots", extra.get("shots", 4096)))
    gamma = float(params.get("gamma", extra.get("gamma", 0.8)))
    beta = float(params.get("beta", extra.get("beta", 0.6)))
    if depth < 1 or shots < 1:
        raise ValueError("qaoa_depth y shots deben ser positivos")
    ids = [n["id"] for n in input_data["nodes"]]
    weights = {n["id"]: n["weight"] for n in input_data["nodes"]}
    penalty = 4 * max(abs(w) for w in weights.values())
    circuit = _circuit(ids, weights, [tuple(e) for e in input_data["edges"]],
                       penalty, depth, gamma, beta)
    backend, backend_used = _backend(extra)
    t_compile = time.perf_counter()
    circuit = transpile(circuit, backend=backend, optimization_level=2)
    compile_seconds = time.perf_counter() - t_compile
    t_job = time.perf_counter()
    job = backend.run(circuit, shots=shots)
    counts = job.result().get_counts()
    job_seconds = time.perf_counter() - t_job
    t_decode = time.perf_counter()
    report = evaluate_counts(input_data, counts)
    decode_seconds = time.perf_counter() - t_decode
    return {**report, "backend_used": backend_used, "objective": "MWIS: 2*passenger_km-empty_km",
            "case_id": input_data["case_id"], "dataset_sha256": dataset_fingerprint(input_data),
            "qaoa_depth": depth, "shots": shots, "gamma_normalized": gamma, "beta": beta,
            "transpile_seconds": compile_seconds, "backend_job_wall_seconds": job_seconds,
            "postprocessing_seconds": decode_seconds,
            "end_to_end_seconds": time.perf_counter() - t0,
            "execution_time_seconds": time.perf_counter() - t0,
            "job_id": job.job_id() if hasattr(job, "job_id") else None}
