"""Validate blinded human coupling annotations without generating labels."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

INDICATORS = (
    "c1_shared_object",
    "c2_temporal_dependency",
    "c3_spatial_dependency",
    "c4_simultaneous_activity",
)


def coupling_level(score: int) -> str:
    if score < 0 or score > 4:
        raise ValueError("coupling score must be between 0 and 4")
    if score <= 1:
        return "weak"
    if score == 2:
        return "medium"
    return "strong"


def read_annotations(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"task_name", "annotator_id", "evidence", *INDICATORS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"annotation CSV missing columns: {sorted(required)}")
        for line_number, raw in enumerate(reader, 2):
            if not raw["task_name"].strip() or not raw["annotator_id"].strip():
                raise ValueError(
                    f"task_name and annotator_id required at line {line_number}"
                )
            if not raw["evidence"].strip():
                raise ValueError(f"evidence required at line {line_number}")
            values = {}
            for indicator in INDICATORS:
                if raw[indicator] not in {"0", "1"}:
                    raise ValueError(f"{indicator} must be 0/1 at line {line_number}")
                values[indicator] = int(raw[indicator])
            rows.append(
                {
                    **raw,
                    **values,
                    "score": sum(values.values()),
                    "level": coupling_level(sum(values.values())),
                }
            )
    return rows


def annotation_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    annotators: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        annotators[row["task_name"]].add(row["annotator_id"])
    incomplete = sorted(task for task, people in annotators.items() if len(people) < 2)
    return {
        "task_count": len(annotators),
        "rows": len(rows),
        "tasks_with_fewer_than_two_annotators": incomplete,
        "ready_for_adjudication": bool(annotators) and not incomplete,
    }
