"""Append-only, JSONL Probe 1 rollout recorder.

The recorder deliberately stores unavailable simulator signals as ``null`` and
an explicit availability map.  It never substitutes zero for an observation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "probe1.episode.v1"
FAILURE_LABELS = {
    "perception_grounding",
    "left_execution",
    "right_execution",
    "temporal_coordination",
    "spatial_coordination",
    "other_uncertain",
}


def validate_episode_summary(payload: Mapping[str, Any]) -> None:
    required = {
        "task_name",
        "episode_id",
        "run_id",
        "eval_seed",
        "success",
        "termination_reason",
        "primary_failure",
        "outcomes",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"episode summary missing fields: {sorted(missing)}")
    failure = payload["primary_failure"]
    if bool(payload["success"]) and failure is not None:
        raise ValueError("successful episode must have primary_failure=null")
    if (
        not bool(payload["success"])
        and not payload.get("infrastructure_error")
        and failure not in FAILURE_LABELS
    ):
        raise ValueError("valid failed episode must have exactly one primary failure")
    if payload.get("infrastructure_error") and failure is not None:
        raise ValueError(
            "infrastructure attempts must not receive a policy failure label"
        )
    if len(payload["outcomes"]) != len(set(payload["outcomes"])):
        raise ValueError("outcomes must not contain duplicates")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Keep the recorder dependency-light; NumPy values from RoboTwin expose
    # these conversion methods, while ordinary values are handled below.
    if type(value).__module__.startswith("numpy"):
        return value.tolist() if hasattr(value, "tolist") else value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


@dataclass
class EpisodeRecorder:
    output_root: Path
    run_id: str
    task_name: str
    episode_id: int
    eval_seed: int
    control_frequency_hz: float
    physics_timestep_s: float | None
    prediction_horizon: int | None = None
    configured_executed_chunk_length: int | None = None
    video_path: str | None = None
    _stream: Any = field(init=False, repr=False)
    _steps: int = field(default=0, init=False)
    _episode_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive")
        self._episode_dir = (
            self.output_root
            / self.run_id
            / "episodes"
            / f"episode_{self.episode_id:06d}_seed_{self.eval_seed}"
        )
        self._episode_dir.mkdir(parents=True, exist_ok=False)
        self._stream = (self._episode_dir / "steps.jsonl").open("x", encoding="utf-8")

    @property
    def episode_dir(self) -> Path:
        return self._episode_dir

    def record_step(
        self,
        *,
        simulator_step: int | None,
        control_step: int,
        observation: Mapping[str, Any],
        raw_action_chunk: Any,
        sent_action: Any,
        action_type: str,
        chunk_id: int,
        chunk_start_step: int,
        action_index_in_chunk: int,
        executed_length: int,
        contacts: Any = None,
        object_states: Any = None,
        joint_velocities: Mapping[str, Any] | None = None,
        task_phase: str | None = None,
        task_phase_source: str | None = None,
        image_indices: Mapping[str, Any] | None = None,
    ) -> None:
        joint = observation.get("joint_action", {})
        endpose = observation.get("endpose", {})
        record = {
            "schema_version": SCHEMA_VERSION,
            "simulator_step": simulator_step,
            "control_step": control_step,
            "physics_time_s": (
                None
                if simulator_step is None or self.physics_timestep_s is None
                else simulator_step * self.physics_timestep_s
            ),
            "control_time_s": control_step / self.control_frequency_hz,
            # RoboTwin's ffmpeg path does not expose the exact frame counter.
            # Do not pretend control steps and encoded video frames are 1:1.
            "video_frame_index": None,
            "robot": {
                "left": {
                    "joint_position": joint.get("left_arm"),
                    "joint_velocity": (joint_velocities or {}).get("left"),
                    "ee_pose": endpose.get("left_endpose"),
                    "gripper_opening": joint.get(
                        "left_gripper", endpose.get("left_gripper")
                    ),
                },
                "right": {
                    "joint_position": joint.get("right_arm"),
                    "joint_velocity": (joint_velocities or {}).get("right"),
                    "ee_pose": endpose.get("right_endpose"),
                    "gripper_opening": joint.get(
                        "right_gripper", endpose.get("right_gripper")
                    ),
                },
            },
            "object_states": object_states,
            "contacts": contacts,
            "raw_action_chunk": raw_action_chunk,
            "returned_chunk_length": len(raw_action_chunk),
            "denormalized_action": sent_action,
            "sent_action": sent_action,
            "action_type": action_type,
            "chunk_id": chunk_id,
            "chunk_start_step": chunk_start_step,
            "action_index_in_chunk": action_index_in_chunk,
            "executed_length": executed_length,
            "image_indices": image_indices,
            "task_phase": task_phase,
            "task_phase_source": task_phase_source,
            "timing": {
                "robot_observation": "pre_action",
                "contacts_and_objects": "post_action",
            },
            "availability": {
                "joint_velocity": joint_velocities is not None,
                "contacts": contacts is not None,
                "object_states": object_states is not None,
                "simulator_step": simulator_step is not None,
                "image_indices": image_indices is not None,
                "video_frame_index": False,
            },
        }
        self._stream.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        self._stream.flush()
        self._steps += 1

    def finish(
        self,
        *,
        success: bool,
        termination_reason: str,
        infrastructure_error: str | None = None,
    ) -> Path:
        self._stream.close()
        primary_failure = None if success or infrastructure_error else "other_uncertain"
        summary = {
            "schema_version": SCHEMA_VERSION,
            "task_name": self.task_name,
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "eval_seed": self.eval_seed,
            "success": bool(success),
            "termination_reason": termination_reason,
            "infrastructure_error": infrastructure_error,
            "primary_failure": primary_failure,
            "outcomes": [],
            "analysis": {
                "rule_version": None,
                "confidence": "unreviewed",
                "events": [],
                "evidence": [],
            },
            "num_control_steps": self._steps,
            "prediction_horizon": self.prediction_horizon,
            "executed_chunk_length": self.configured_executed_chunk_length,
            "control_frequency_hz": self.control_frequency_hz,
            "physics_timestep_s": self.physics_timestep_s,
            "video_path": self.video_path,
            "replay_verified": False,
        }
        path = self._episode_dir / "episode.json"
        validate_episode_summary(summary)
        write_json_exclusive(path, summary)
        return path


def build_manifest(
    *,
    repo_root: Path,
    run_id: str,
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    baseline: str,
    training_seed: int | None,
    eval_seed: int,
    action_mapping: Mapping[str, Any],
    state_mapping: Mapping[str, Any],
    prediction_horizon: int | None,
    executed_chunk_length: int | None,
    control_frequency_hz: float,
    physics_timestep_s: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": "probe1.run.v1",
        "run_id": run_id,
        "repo_commit": git_revision(repo_root),
        "xpolicylab_commit": git_revision(repo_root / "XPolicyLab"),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "baseline": baseline,
        "training_seed": training_seed,
        "eval_seed": eval_seed,
        "python": platform.python_version(),
        "action_mapping": action_mapping,
        "state_mapping": state_mapping,
        "prediction_horizon": prediction_horizon,
        "executed_chunk_length": executed_chunk_length,
        "control_frequency_hz": control_frequency_hz,
        "physics_timestep_s": physics_timestep_s,
        "output_dir": str(output_dir),
        "replay_verified": False,
    }
