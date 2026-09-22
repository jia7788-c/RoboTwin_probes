"""Dependency-light Probe 1 metrics and task-cluster bootstrap."""

from __future__ import annotations
import random
from collections import defaultdict
from typing import Any, Iterable

COORDINATION = {"temporal_coordination", "spatial_coordination"}


def _ratio(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def summarize(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(episodes)
    valid = [r for r in rows if not r.get("infrastructure_error")]
    ids = [(r.get("run_id"), r.get("episode_id")) for r in valid]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate valid episode ID within run")
    conditions = {(r.get("baseline"), r.get("training_seed")) for r in rows}
    if len(conditions) > 1:
        raise ValueError("mixed baseline/training_seed: summarize each condition separately")
    classified = all(
        r.get("primary_failure") in {
            "perception_grounding", "left_execution", "right_execution",
            "temporal_coordination", "spatial_coordination", "other_uncertain",
        }
        and r.get("analysis", {}).get("confidence") not in (None, "", "unreviewed")
        and bool(r.get("analysis", {}).get("rule_version"))
        and bool(r.get("analysis", {}).get("evidence"))
        for r in valid if not r.get("success")
    )
    success = sum(bool(r.get("success")) for r in valid)
    failure = len(valid) - success
    temporal = sum(r.get("primary_failure") == "temporal_coordination" for r in valid)
    spatial = sum(r.get("primary_failure") == "spatial_coordination" for r in valid)
    return {
        "N_attempted": len(rows),
        "N_valid": len(valid),
        "N_success": success,
        "N_failure": failure,
        "N_temporal": temporal if classified else None,
        "N_spatial": spatial if classified else None,
        "classification_complete": classified,
        "SR": _ratio(success, len(valid)),
        "CFR_all": _ratio(temporal + spatial, len(valid)) if classified else None,
        "CFR_fail": _ratio(temporal + spatial, failure) if classified else None,
    }


def task_macro(episodes: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    episodes = list(episodes)
    summarize(episodes)  # Reject mixed conditions and duplicate IDs before grouping.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        groups[str(row["task_name"])].append(row)
    summaries = [summarize(value) for value in groups.values()]
    result = {}
    for key in ("SR", "CFR_all", "CFR_fail"):
        values = [summary[key] for summary in summaries if summary[key] is not None]
        incomplete = key.startswith("CFR") and any(not s["classification_complete"] for s in summaries)
        result[key] = sum(values) / len(values) if values and not incomplete else None
    return result


def clustered_delta_cfr(
    episodes: Iterable[dict[str, Any]],
    coupling: dict[str, str],
    *,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Task-cluster CI within each baseline/training seed, never across them.

    Mixed input returns a conditions list with no pooled estimate. Homogeneous
    input retains the historical result shape. A condition missing either
    coupling stratum returns null rather than borrowing another condition.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    episodes = list(episodes)
    conditions: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        conditions[(row.get("baseline"), row.get("training_seed"))].append(row)
    if len(conditions) > 1:
        return {"conditions": [
            {"baseline": baseline, "training_seed": training_seed,
             **clustered_delta_cfr(rows, coupling, samples=samples, seed=seed)}
            for (baseline, training_seed), rows in conditions.items()
        ]}
    # Validate the entire condition before splitting it into task clusters.
    summarize(episodes)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        groups[str(row["task_name"])].append(row)
    if any(not summarize(rows)["classification_complete"] for rows in groups.values()):
        raise ValueError("classify all valid failures before computing Delta_CFR")
    strong = [
        t
        for t in groups
        if coupling.get(t) == "strong" and summarize(groups[t])["CFR_all"] is not None
    ]
    weak = [
        t
        for t in groups
        if coupling.get(t) == "weak" and summarize(groups[t])["CFR_all"] is not None
    ]
    if not strong or not weak:
        return {"delta_cfr": None, "ci95": None, "bootstrap_seed": seed}

    def mean(names: list[str]) -> float:
        return sum(float(summarize(groups[t])["CFR_all"]) for t in names) / len(names)

    estimate = mean(strong) - mean(weak)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(
            mean([rng.choice(strong) for _ in strong])
            - mean([rng.choice(weak) for _ in weak])
        )
    draws.sort()
    lo = draws[int(0.025 * (samples - 1))]
    hi = draws[int(0.975 * (samples - 1))]
    return {"delta_cfr": estimate, "ci95": [lo, hi], "bootstrap_seed": seed}
