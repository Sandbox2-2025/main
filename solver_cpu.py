"""Datos, generación y verificación independiente del caso ferroviario simulado.

Horas como enteros desde 00:00 del día 1; día 2 empieza en t=24.
La distancia `passenger_km` significa km recorridos con viajeros por el tren,
no pasajeros multiplicados por km.
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


def exact_cpu(data):
    """DP exacta sobre máscaras de servicios, maximiza MWIS y deshace empates
    por menos trenes. El tiempo indicado excluye generación y validación.
    """
    from time import perf_counter
    overall_start = perf_counter()
    validate_dataset(data)
    t0 = perf_counter()
    ids = [t["id"] for t in data["services"]]
    bits = {t: 1 << i for i, t in enumerate(ids)}
    states = {0: (0, ())}
    for c in data["nodes"]:
        cmask = sum(bits[t] for t in c["trips"])
        for mask, (weight, chosen) in list(states.items()):
            if mask & cmask:
                continue
            nxt = mask | cmask
            alternative = (weight + c["weight"], chosen + (c["id"],))
            previous = states.get(nxt)
            if previous is None or (alternative[0], -len(alternative[1])) > (previous[0], -len(previous[1])):
                states[nxt] = alternative
    best_mask, best = max(states.items(), key=lambda item: (item[1][0], -len(item[1][1])))
    full_mask = (1 << len(ids)) - 1
    full = states.get(full_mask)
    result = {"backend_used": "CPU exact DP", "objective": "MWIS: 2*passenger_km-empty_km",
              "case_id": data["case_id"], "dataset_sha256": dataset_fingerprint(data),
              "solver_time_seconds": perf_counter() - t0,
              "explored_trip_masks": len(states), **assess(data, list(best[1]))}
    if full:
        full_result = assess(data, list(full[1]))
        result["best_full_coverage"] = {
            "selected_cycles": full_result["selected_cycles"],
            "total_empty_km": full_result["total_empty_km"],
            "number_of_trains": full_result["number_of_trains"],
            "total_weight": full_result["total_weight"]}
    else:
        result["best_full_coverage"] = None
    result["execution_time_seconds"] = perf_counter() - overall_start
    return result


def run(input_data, solver_params=None, extra_arguments=None):
    """Punto de entrada de Q-Centroid. Todo el solver CPU está en este fichero."""
    del solver_params, extra_arguments
    return exact_cpu(input_data)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if len(sys.argv) != 2:
        raise SystemExit("Uso local: python qcentroid.py dataset_20_ciclos.json")
    dataset = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(json.dumps(run(dataset), ensure_ascii=False, indent=2))
