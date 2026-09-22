"""Summarize each task/baseline/training seed without pooling conditions."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from .io import load_episode_summaries
from .statistics import summarize, task_macro


DEFAULT_FAILURE_THRESHOLDS = Path(__file__).resolve().parents[1] / "configs/failure_thresholds.yaml"
# Required keys follow probe1.docx Appendix A, not a guessed calibrated protocol.
REQUIRED_THRESHOLD_FIELDS = {
    "stable_grasp": ("min_contact_frames", "max_object_gripper_velocity_gap"),
    "release": ("min_opening", "min_no_contact_frames"),
    "handover": ("early_release_margin_steps",),
    "collision": ("min_force", "min_duration_steps"),
    "metadata": ("detector_version", "frozen_after_pilot"),
}


def validate_formal_thresholds(path: Path) -> dict:
    """Fail closed on absent/unfrozen/uncalibrated YAML, including nested fields."""
    import yaml  # Existing repository dependency; exploratory analysis stays lightweight.

    try:
        raw = path.read_bytes()
        config = yaml.safe_load(raw)
    except FileNotFoundError as exc:
        raise ValueError(f"formal failure thresholds file does not exist: {path}") from exc
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        raise ValueError(f"formal failure thresholds cannot be read: {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("formal failure thresholds must be a nonempty mapping")
    for section, fields in REQUIRED_THRESHOLD_FIELDS.items():
        values = config.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"formal failure thresholds missing section: {section}")
        for field in fields:
            if field not in values:
                raise ValueError(f"formal failure thresholds missing field: {section}.{field}")
    if config["metadata"]["frozen_after_pilot"] is not True:
        raise ValueError("formal failure thresholds require metadata.frozen_after_pilot: true")

    def check(value, name):
        if value is None or (isinstance(value, str) and
                             (not value.strip() or value.strip().upper() == "TO_BE_CALIBRATED")):
            raise ValueError(f"formal failure thresholds uncalibrated: {name}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"formal failure thresholds uncalibrated (nonfinite): {name}")
        if isinstance(value, (dict, list)):
            if not value:
                raise ValueError(f"formal failure thresholds uncalibrated (empty): {name}")
            children = value.items() if isinstance(value, dict) else enumerate(value)
            for key, child in children:
                check(child, f"{name}.{key}" if name else str(key))

    check(config, "")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "detector_version": config["metadata"]["detector_version"],
            "frozen_after_pilot": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal", action="store_true",
                        help="Require frozen protocol and reviewed failure classifications.")
    parser.add_argument("--failure-thresholds", type=Path, default=DEFAULT_FAILURE_THRESHOLDS,
                        help="Failure-threshold YAML checked by --formal (default: %(default)s).")
    args = parser.parse_args(argv)
    # Validate before any result directory/file is created, even for empty inputs.
    thresholds = validate_formal_thresholds(args.failure_thresholds) if args.formal else None
    episodes = load_episode_summaries(args.inputs)
    groups = defaultdict(list)
    for row in episodes:
        if not row.get("baseline"):
            raise ValueError("baseline missing; supply episode metadata or its run manifest")
        if args.formal and (row.get("training_seed") is None
                            or row.get("formal_analysis_allowed") is not True
                            or row.get("threshold_status") != "frozen"):
            raise ValueError("formal analysis requires training seed and a frozen, allowed protocol")
        groups[(row["baseline"], row.get("training_seed"))].append(row)
    conditions = []
    for (baseline, seed), rows in groups.items():
        micro = summarize(rows)
        if args.formal and not micro["classification_complete"]:
            raise ValueError("formal CFR requires reviewed failures with rule_version and evidence")
        tasks = defaultdict(list)
        for row in rows:
            tasks[row["task_name"]].append(row)
        conditions.append({
            "baseline": baseline, "training_seed": seed,
            "episode_micro": micro, "task_macro": task_macro(rows),
            "by_task": {task: summarize(values) for task, values in sorted(tasks.items())},
        })
    result = {"analysis_version": "probe1.analysis.v2", "conditions": conditions,
              "episode_count": len(episodes), "formal": args.formal,
              "interpretation": "diagnostic association only; no causal claim"}
    if thresholds is not None:
        result["failure_thresholds"] = thresholds
    # Backward compatibility only for one homogeneous condition.
    if len(conditions) == 1:
        result.update({key: conditions[0][key] for key in ("episode_micro", "task_macro")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
