"""Validate and assign mutually exclusive diagnostic failure labels."""
from __future__ import annotations
from typing import Any, Iterable

LABELS = {"perception_grounding", "left_execution", "right_execution", "temporal_coordination", "spatial_coordination", "other_uncertain"}

def classify_episode(*, success: bool, candidates: Iterable[dict[str, Any]], outcomes: Iterable[str] = ()) -> dict[str, Any]:
    evidence = sorted(list(candidates), key=lambda item: (item.get("time_s") is None, item.get("time_s", float("inf"))))
    unique_outcomes = list(dict.fromkeys(outcomes))
    if success: return {"primary_failure": None, "outcomes": unique_outcomes, "evidence": evidence}
    valid = [item for item in evidence if item.get("label") in LABELS and item.get("evidence")]
    if not valid: label = "other_uncertain"
    else:
        first_time = valid[0].get("time_s")
        earliest = [item for item in valid if item.get("time_s") == first_time]
        label = earliest[0]["label"] if len({item["label"] for item in earliest}) == 1 else "other_uncertain"
    return {"primary_failure": label, "outcomes": unique_outcomes, "evidence": evidence}
