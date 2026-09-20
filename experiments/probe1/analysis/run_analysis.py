"""Compute preregistered Probe 1 aggregate metrics from real episode summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_episode_summaries
from .statistics import summarize, task_macro


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Run directories or episode.json files"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    episodes = load_episode_summaries(args.inputs)
    result = {
        "analysis_version": "probe1.analysis.v1",
        "episode_micro": summarize(episodes),
        "task_macro": task_macro(episodes),
        "episode_count": len(episodes),
        "interpretation": "diagnostic association only; no causal claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
