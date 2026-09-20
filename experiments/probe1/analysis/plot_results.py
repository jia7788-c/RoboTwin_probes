"""Plot only previously computed Probe 1 results; never synthesizes values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    micro = payload.get("episode_micro", {})
    names = ["SR", "CFR_all", "CFR_fail"]
    if any(micro.get(name) is None for name in names):
        raise SystemExit("all SR/CFR metrics must be non-null before plotting")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(names, [micro[name] for name in names])
    axis.set_ylim(0, 1)
    axis.set_ylabel("rate")
    axis.set_title("Probe 1 episode-micro metrics")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
