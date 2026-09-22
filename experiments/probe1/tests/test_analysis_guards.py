"""Synthetic-only guard regressions; no model, GPU, or simulator fixtures."""
import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from experiments.probe1.analysis.run_analysis import main
from experiments.probe1.analysis.statistics import clustered_delta_cfr

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "experiments/probe1/configs/failure_thresholds.yaml"


def rows(task, outcomes, baseline="B0", training_seed=1):
    return [dict(run_id=f"synthetic-{task}-{baseline}-{training_seed}",
                 episode_id=i, task_name=task, eval_seed=i,
                 baseline=baseline, training_seed=training_seed,
                 success=not coordination,
                 termination_reason="timeout" if coordination else "success",
                 primary_failure="temporal_coordination" if coordination else None,
                 outcomes=["timeout"] if coordination else [],
                 analysis=dict(rule_version="synthetic.v1", confidence="reviewed",
                               evidence=["synthetic only"]))
            for i, coordination in enumerate(outcomes)]


def test_bootstrap_mixed_baselines_matches_hand_calculation():
    # Unequal episode counts distinguish task-macro (0.5) from micro (0.9).
    data = (rows("s0", [1] * 9) + rows("s1", [0]) + rows("w0", [0])
            + rows("s2", [0], "B1") + rows("w1", [1], "B1"))
    coupling = dict(s0="strong", s1="strong", w0="weak", s2="strong", w1="weak")
    got = clustered_delta_cfr(data, coupling, samples=100, seed=7)
    assert "delta_cfr" not in got and "ci95" not in got
    groups = {(r["baseline"], r["training_seed"]): r for r in got["conditions"]}
    assert groups[("B0", 1)]["delta_cfr"] == .5  # (1+0)/2 - 0
    assert groups[("B1", 1)]["delta_cfr"] == -1  # 0 - 1
    # Explicit task-cluster bootstrap reference, not episode resampling.
    rng = random.Random(7)
    draws = []
    for _ in range(100):
        draws.append((rng.choice([1., 0.]) + rng.choice([1., 0.])) / 2
                     - rng.choice([0.]))
    draws.sort()
    assert groups[("B0", 1)]["ci95"] == [draws[2], draws[96]]
    assert groups[("B1", 1)]["ci95"] == [-1, -1]
    assert got == clustered_delta_cfr(iter(data), coupling, samples=100, seed=7)


def test_bootstrap_training_seeds_remain_separate():
    data = (rows("s", [1], training_seed=1) + rows("w", [0], training_seed=1)
            + rows("s", [0], training_seed=2) + rows("w", [1], training_seed=2)
            + rows("s", [1], training_seed=3))
    got = clustered_delta_cfr(data, dict(s="strong", w="weak"), samples=20)
    assert "delta_cfr" not in got
    groups = {r["training_seed"]: r for r in got["conditions"]}
    assert len(groups) == 3
    assert groups[1]["delta_cfr"] == 1 and groups[1]["ci95"] == [1, 1]
    assert groups[2]["delta_cfr"] == -1 and groups[2]["ci95"] == [-1, -1]
    assert groups[3]["delta_cfr"] is None and groups[3]["ci95"] is None


def test_bootstrap_cannot_borrow_other_conditions_missing_stratum():
    got = clustered_delta_cfr(rows("s", [1]) + rows("w", [0], "B1", 2),
                              dict(s="strong", w="weak"), samples=20)
    assert len(got["conditions"]) == 2
    assert all(r["delta_cfr"] is None for r in got["conditions"])


def calibrated_synthetic_config():
    # Test constants only: NOT calibrated thresholds or a real pilot protocol.
    return dict(stable_grasp=dict(min_contact_frames=8, max_object_gripper_velocity_gap=.01),
                release=dict(min_opening=.8, min_no_contact_frames=5),
                handover=dict(early_release_margin_steps=2),
                collision=dict(min_force=1., min_duration_steps=2),
                metadata=dict(detector_version="synthetic.v1", frozen_after_pilot=True))


def formal_input(tmp_path):
    path = tmp_path / "episode.json"
    record = rows("s", [1])[0]
    record.update(formal_analysis_allowed=True, threshold_status="frozen")
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


@pytest.mark.parametrize("invalid", [None, "TO_BE_CALIBRATED"])
def test_formal_null_or_placeholder_cli_exits_without_output(tmp_path, invalid):
    config = calibrated_synthetic_config()
    config["stable_grasp"]["max_object_gripper_velocity_gap"] = invalid
    thresholds = tmp_path / "failure_thresholds.yaml"
    thresholds.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "result" / "formal.json"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "experiments.probe1.analysis.run_analysis",
         str(formal_input(tmp_path)), "--formal", "--failure-thresholds", str(thresholds),
         "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "uncalibrated" in result.stderr
    assert "stable_grasp.max_object_gripper_velocity_gap" in result.stderr
    assert not output.exists() and not output.parent.exists()


@pytest.mark.parametrize("invalid", [False, None, "true", 1])
def test_formal_requires_boolean_frozen_flag(tmp_path, invalid):
    config = calibrated_synthetic_config()
    config["metadata"]["frozen_after_pilot"] = invalid
    thresholds = tmp_path / "failure_thresholds.yaml"
    thresholds.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "formal.json"
    with pytest.raises(ValueError, match="frozen_after_pilot"):
        main([str(formal_input(tmp_path)), "--formal", "--failure-thresholds",
              str(thresholds), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("case", ["missing-file", "empty", "missing-field", "nested-null"])
def test_formal_missing_or_nested_thresholds_fail_closed(tmp_path, case):
    config = copy.deepcopy(calibrated_synthetic_config())
    if case == "empty":
        config = {}
    elif case == "missing-field":
        del config["collision"]["min_force"]
    elif case == "nested-null":
        config["extra_task"] = {"limits": [1, {"distance": None}]}
    thresholds = tmp_path / "failure_thresholds.yaml"
    if case != "missing-file":
        thresholds.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "formal.json"
    with pytest.raises(ValueError, match="formal"):
        main([str(formal_input(tmp_path)), "--formal", "--failure-thresholds",
              str(thresholds), "--output", str(output)])
    assert not output.exists()


def test_formal_valid_synthetic_config_records_hash(tmp_path):
    thresholds = tmp_path / "failure_thresholds.yaml"
    thresholds.write_text(yaml.safe_dump(calibrated_synthetic_config()), encoding="utf-8")
    output = tmp_path / "formal.json"
    assert main([str(formal_input(tmp_path)), "--formal", "--failure-thresholds",
                 str(thresholds), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["formal"] is True
    assert result["failure_thresholds"]["sha256"] == hashlib.sha256(thresholds.read_bytes()).hexdigest()
    assert result["failure_thresholds"]["path"] == str(thresholds.resolve())


def test_default_threshold_template_rejects_formal_not_exploratory(tmp_path):
    config = yaml.safe_load(TEMPLATE.read_text())
    assert config == {
        "stable_grasp": {"min_contact_frames": 8,
                         "max_object_gripper_velocity_gap": "TO_BE_CALIBRATED"},
        "release": {"min_opening": "TO_BE_CALIBRATED", "min_no_contact_frames": 5},
        "handover": {"early_release_margin_steps": "TO_BE_CALIBRATED"},
        "collision": {"min_force": "TO_BE_CALIBRATED",
                      "min_duration_steps": "TO_BE_CALIBRATED"},
        "metadata": {"detector_version": "v1", "frozen_after_pilot": False},
    }
    assert config["metadata"]["frozen_after_pilot"] is False
    source = formal_input(tmp_path)
    formal = tmp_path / "formal.json"
    with pytest.raises(ValueError, match="formal"):
        main([str(source), "--formal", "--output", str(formal)])
    assert not formal.exists()
    exploratory = tmp_path / "exploratory.json"
    assert main([str(source), "--output", str(exploratory)]) == 0
    assert json.loads(exploratory.read_text())["formal"] is False


@pytest.mark.parametrize("section,field", [
    (section, field) for section, values in calibrated_synthetic_config().items()
    for field in values
])
@pytest.mark.parametrize("invalid", [None, "TO_BE_CALIBRATED"])
def test_formal_every_field_names_failure_without_output(tmp_path, section, field, invalid):
    config = calibrated_synthetic_config()
    config[section][field] = invalid
    thresholds = tmp_path / "failure_thresholds.yaml"
    thresholds.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "result" / "formal.json"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "experiments.probe1.analysis.run_analysis",
         str(formal_input(tmp_path)), "--formal", "--failure-thresholds", str(thresholds),
         "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert f"{section}.{field}" in result.stderr
    assert "ValueError: formal failure thresholds" in result.stderr
    assert not output.exists() and not output.parent.exists()


@pytest.mark.parametrize("case", ["missing-file", "unfrozen"])
def test_formal_missing_file_or_unfrozen_cli_names_cause(tmp_path, case):
    thresholds = tmp_path / "failure_thresholds.yaml"
    if case == "unfrozen":
        config = calibrated_synthetic_config()
        config["metadata"]["frozen_after_pilot"] = False
        thresholds.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "result" / "formal.json"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "experiments.probe1.analysis.run_analysis",
         str(formal_input(tmp_path)), "--formal", "--failure-thresholds", str(thresholds),
         "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    if case == "missing-file":
        assert f"file does not exist: {thresholds}" in result.stderr
    else:
        assert "metadata.frozen_after_pilot: true" in result.stderr
    assert not output.exists() and not output.parent.exists()
