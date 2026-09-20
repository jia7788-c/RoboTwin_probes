"""Dependency-light Probe 1 metrics and task-cluster bootstrap."""
from __future__ import annotations
import random
from collections import defaultdict
from typing import Any, Iterable

COORDINATION = {"temporal_coordination", "spatial_coordination"}
def _ratio(n: int, d: int) -> float | None: return None if d == 0 else n / d

def summarize(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(episodes); ids = [(r.get("run_id"), r.get("episode_id")) for r in rows]
    if len(ids) != len(set(ids)): raise ValueError("duplicate episode ID within run")
    valid = [r for r in rows if not r.get("infrastructure_error")]
    success = sum(bool(r.get("success")) for r in valid); failure = len(valid) - success
    temporal = sum(r.get("primary_failure") == "temporal_coordination" for r in valid)
    spatial = sum(r.get("primary_failure") == "spatial_coordination" for r in valid)
    return {"N_attempted": len(rows), "N_valid": len(valid), "N_success": success, "N_failure": failure,
            "N_temporal": temporal, "N_spatial": spatial, "SR": _ratio(success, len(valid)),
            "CFR_all": _ratio(temporal + spatial, len(valid)), "CFR_fail": _ratio(temporal + spatial, failure)}

def task_macro(episodes: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes: groups[str(row["task_name"])].append(row)
    summaries = [summarize(value) for value in groups.values()]
    return {key: (sum(vals) / len(vals) if vals else None) for key in ("SR", "CFR_all", "CFR_fail") if (vals := [s[key] for s in summaries if s[key] is not None])}

def clustered_delta_cfr(episodes: Iterable[dict[str, Any]], coupling: dict[str, str], *, samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes: groups[str(row["task_name"])].append(row)
    strong = [t for t in groups if coupling.get(t) == "strong"]; weak = [t for t in groups if coupling.get(t) == "weak"]
    if not strong or not weak: return {"delta_cfr": None, "ci95": None, "bootstrap_seed": seed}
    def mean(names: list[str]) -> float: return sum(float(summarize(groups[t])["CFR_all"]) for t in names) / len(names)
    estimate = mean(strong) - mean(weak); rng = random.Random(seed); draws=[]
    for _ in range(samples): draws.append(mean([rng.choice(strong) for _ in strong]) - mean([rng.choice(weak) for _ in weak]))
    draws.sort(); lo=draws[int(.025*(samples-1))]; hi=draws[int(.975*(samples-1))]
    return {"delta_cfr": estimate, "ci95": [lo, hi], "bootstrap_seed": seed}
