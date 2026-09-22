import json
from pathlib import Path
import pytest
from experiments.probe1.analysis.coupling import (
    annotation_coverage,
    coupling_level,
    read_annotations,
)
from experiments.probe1.analysis.event_detector import (
    Event,
    first_release,
    first_stable_grasp,
    handover_timing,
    seconds_to_samples,
    unexpected_drop,
)
from experiments.probe1.analysis.failure_analyzer import classify_episode
from experiments.probe1.analysis.io import load_episode_summaries, read_jsonl
from experiments.probe1.analysis.run_analysis import main as analysis_main
from experiments.probe1.analysis.statistics import clustered_delta_cfr, summarize
from experiments.probe1.rollout.hook import extract_task_state, limit_action_chunk
from experiments.probe1.rollout.recorder import (
    EpisodeRecorder,
    validate_episode_summary,
)


def row(i, success, failure=None, task="t", run="r", infra=None):
    return {
        "episode_id": i,
        "run_id": run,
        "task_name": task,
        "success": success,
        "primary_failure": failure,
        "infrastructure_error": infra,
        "analysis": {"rule_version": "synthetic.v1", "confidence": "reviewed", "evidence": ["synthetic"]},
    }


def test_failure_contract_and_outcome_deduplication():
    assert (
        classify_episode(success=True, candidates=[], outcomes=["collision"])[
            "primary_failure"
        ]
        is None
    )
    got = classify_episode(
        success=False,
        candidates=[
            {"label": "left_execution", "time_s": 1, "evidence": "x"},
            {"label": "temporal_coordination", "time_s": 1, "evidence": "y"},
        ],
        outcomes=["collision", "collision"],
    )
    assert got["primary_failure"] == "other_uncertain" and got["outcomes"] == [
        "collision"
    ]


def test_grasp_contact_chatter_and_sample_boundary():
    samples = [
        {"time_s": i / 10, "contact": c, "relative_motion_m": 0.001}
        for i, c in enumerate([1, 1, 0, 1, 1, 1])
    ]
    event = first_stable_grasp(
        samples, sample_hz=10, stable_seconds=0.3, max_relative_motion_m=0.01
    )
    assert (
        event.onset_s == 0.3
        and event.confirmed_s == 0.5
        and seconds_to_samples(0.21, 10) == 3
    )


def test_handover_early_and_never_grasped():
    release = Event("release", 1.0, 1.1, "rule", {})
    assert handover_timing(release, None, early_margin_s=0.2)["delta_t_s"] is None
    grasp = Event("grasp", 1.5, 1.6, "rule", {})
    assert (
        handover_timing(release, grasp, early_margin_s=0.2)["early_release_candidate"]
        is True
    )


def test_release_chatter_and_legal_placement_not_drop():
    samples = [
        {"time_s": i / 10, "contact": contact, "gripper_opening": opening}
        for i, (contact, opening) in enumerate([(0, 1), (1, 1), (0, 1), (0, 1)])
    ]
    release = first_release(
        samples, sample_hz=10, stable_seconds=0.2, open_threshold=0.8
    )
    assert release.onset_s == 0.2 and release.confirmed_s == 0.3
    placement = [
        {
            "time_s": 1,
            "object_height_m": 0.74,
            "supported": False,
            "expected_release": True,
        }
    ]
    assert (
        unexpected_drop(placement, support_height_m=0.74, height_margin_m=0.01) is None
    )


def test_metrics_edge_cases_duplicates_and_bootstrap_reproducible():
    assert summarize([])["SR"] is None
    all_fail = summarize([row(1, False, "temporal_coordination")])
    assert all_fail["SR"] == 0 and all_fail["CFR_fail"] == 1
    no_fail = summarize([row(1, True)])
    assert no_fail["CFR_fail"] is None
    with pytest.raises(ValueError):
        summarize([row(1, True), row(1, False)])
    rows = [row(1, False, "temporal_coordination", "s"), row(2, True, None, "w")]
    assert clustered_delta_cfr(
        rows, {"s": "strong", "w": "weak"}, samples=50, seed=4
    ) == clustered_delta_cfr(rows, {"s": "strong", "w": "weak"}, samples=50, seed=4)


def test_recorder_null_availability_and_no_overwrite(tmp_path: Path):
    rec = EpisodeRecorder(tmp_path, "run", "handover_block", 0, 7, 30, 1 / 250)
    rec.record_step(
        simulator_step=None,
        control_step=0,
        observation={"joint_action": {}, "endpose": {}},
        raw_action_chunk=[[1]],
        sent_action=[1],
        action_type="qpos",
        chunk_id=0,
        chunk_start_step=0,
        action_index_in_chunk=0,
        executed_length=1,
    )
    summary = rec.finish(success=False, termination_reason="timeout")
    step = json.loads((summary.parent / "steps.jsonl").read_text())
    assert step["contacts"] is None and not step["availability"]["contacts"]
    with pytest.raises(FileExistsError):
        EpisodeRecorder(tmp_path, "run", "handover_block", 0, 7, 30, 1 / 250)


def test_summary_schema_contract():
    valid = {
        "task_name": "t",
        "episode_id": 1,
        "run_id": "r",
        "eval_seed": 2,
        "success": True,
        "termination_reason": "success",
        "primary_failure": None,
        "outcomes": [],
    }
    validate_episode_summary(valid)
    with pytest.raises(ValueError):
        validate_episode_summary({**valid, "success": False})
    with pytest.raises(ValueError):
        validate_episode_summary({**valid, "outcomes": ["drop", "drop"]})
    infrastructure = {
        **valid,
        "success": False,
        "termination_reason": "infrastructure_error",
        "infrastructure_error": "renderer",
        "primary_failure": None,
    }
    validate_episode_summary(infrastructure)


def test_task_state_uses_observed_pose():
    class Pose:
        p = [1, 2, 3]
        q = [1, 0, 0, 0]

    class Actor:
        def get_pose(self):
            return Pose()

    class Env:
        box = Actor()
        target_box = Actor()

    state = extract_task_state(Env())
    assert state["box"]["position_m"] == [1, 2, 3]


def test_execution_length_is_explicit(monkeypatch):
    monkeypatch.delenv("PROBE1_EXECUTED_CHUNK_LENGTH", raising=False)
    assert limit_action_chunk([1, 2, 3]) == [1, 2, 3]
    monkeypatch.setenv("PROBE1_EXECUTED_CHUNK_LENGTH", "2")
    assert limit_action_chunk([1, 2, 3]) == [1, 2]
    with pytest.raises(ValueError):
        limit_action_chunk([1])


def test_offline_reader_and_analysis_cli(tmp_path: Path):
    rec = EpisodeRecorder(
        tmp_path, "run", "handover_block", 0, 7, 30, 1 / 250, baseline="B0", training_seed=11
    )
    rec.record_step(
        simulator_step=1,
        control_step=0,
        observation={},
        raw_action_chunk=[[1]],
        sent_action=[1],
        action_type="qpos",
        chunk_id=0,
        chunk_start_step=0,
        action_index_in_chunk=0,
        executed_length=1,
    )
    summary = rec.finish(success=True, termination_reason="success")
    assert len(read_jsonl(summary.parent / "steps.jsonl")) == 1
    loaded = load_episode_summaries([tmp_path])
    assert len(loaded) == 1
    assert loaded[0]["baseline"] == "B0" and loaded[0]["training_seed"] == 11
    assert loaded[0]["primary_failure"] is None
    assert loaded[0]["analysis"]["confidence"] == "unreviewed"
    output = tmp_path / "analysis.json"
    assert analysis_main([str(tmp_path), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["episode_micro"]["SR"] == 1
    assert result["conditions"][0]["baseline"] == "B0"
    assert result["conditions"][0]["training_seed"] == 11


def test_coupling_annotations_require_two_humans_and_binary_scores(tmp_path: Path):
    path = tmp_path / "annotations.csv"
    header = "task_name,annotator_id,c1_shared_object,c2_temporal_dependency,c3_spatial_dependency,c4_simultaneous_activity,evidence\n"
    path.write_text(
        header + "handover_block,a,1,1,1,1,task definition\n", encoding="utf-8"
    )
    rows = read_annotations(path)
    assert rows[0]["score"] == 4 and coupling_level(4) == "strong"
    assert not annotation_coverage(rows)["ready_for_adjudication"]
    path.write_text(header + "handover_block,a,2,1,1,1,bad\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_annotations(path)
