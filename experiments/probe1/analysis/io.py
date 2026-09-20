"""Validated readers for append-only Probe 1 output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from experiments.probe1.rollout.recorder import validate_episode_summary


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_episode_summaries(inputs: Iterable[Path]) -> list[dict[str, Any]]:
    summaries = []
    for root in inputs:
        paths = [root] if root.is_file() else sorted(root.glob("**/episode.json"))
        for path in paths:
            value = read_json(path)
            validate_episode_summary(value)
            summaries.append(value)
    return summaries
