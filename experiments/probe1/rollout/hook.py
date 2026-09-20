"""Small optional hook used by RoboTwin's existing single-worker eval loop."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping

from .recorder import EpisodeRecorder


def limit_action_chunk(action_chunk: list[Any]) -> list[Any]:
    """Apply the explicitly configured K only for a Probe 1 run."""
    raw = os.environ.get("PROBE1_EXECUTED_CHUNK_LENGTH")
    if raw is None:
        return action_chunk
    length = int(raw)
    if length <= 0:
        raise ValueError("PROBE1_EXECUTED_CHUNK_LENGTH must be positive")
    if len(action_chunk) < length:
        raise ValueError(
            f"policy returned {len(action_chunk)} actions, fewer than configured K={length}"
        )
    return action_chunk[:length]


def _pose(entity: Any) -> dict[str, Any] | None:
    getter = getattr(entity, "get_pose", None)
    if not callable(getter):
        return None
    pose = getter()
    return {
        "position_m": getattr(pose, "p", None),
        "quaternion_wxyz": getattr(pose, "q", None),
    }


def extract_task_state(task_env: Any) -> dict[str, Any] | None:
    """Extract only known, actually observed M0 objects from the live task."""
    states = {}
    for name in ("box", "target_box"):
        value = _pose(getattr(task_env, name, None))
        if value is not None:
            states[name] = value
    return states or None


def extract_contacts(task_env: Any) -> list[dict[str, Any]] | None:
    """Serialize live SAPIEN contacts without inventing unavailable forces."""
    scene = getattr(task_env, "scene", None)
    getter = getattr(scene, "get_contacts", None)
    if not callable(getter):
        return None
    result = []
    for contact in getter():
        bodies = getattr(contact, "bodies", ())
        names = []
        for body in bodies:
            entity = getattr(body, "entity", body)
            names.append(getattr(entity, "name", None))
        points = []
        for point in getattr(contact, "points", ()):
            points.append(
                {
                    "position_m": getattr(point, "position", None),
                    "impulse_ns": getattr(point, "impulse", None),
                    "separation_m": getattr(point, "separation", None),
                }
            )
        result.append({"body_names": names, "points": points})
    return result


def extract_joint_velocities(task_env: Any) -> dict[str, Any] | None:
    robot = getattr(task_env, "robot", None)
    velocities = {}
    for side in ("left", "right"):
        entity = getattr(robot, f"{side}_entity", None)
        getter = getattr(entity, "get_qvel", None)
        if callable(getter):
            velocities[side] = getter()
    return velocities or None


def start_episode(
    *, task_name: str, episode_id: int, seed: int, frequency: float, task_env: Any
) -> EpisodeRecorder | None:
    root = os.environ.get("PROBE1_OUTPUT_DIR")
    run_id = os.environ.get("PROBE1_RUN_ID")
    if not root or not run_id:
        return None
    timestep = None
    scene = getattr(task_env, "scene", None)
    getter = getattr(scene, "get_timestep", None)
    if callable(getter):
        timestep = float(getter())
    horizon = os.environ.get("PROBE1_PREDICTION_HORIZON")
    executed = os.environ.get("PROBE1_EXECUTED_CHUNK_LENGTH")
    video_path = getattr(task_env, "eval_video_path", None)
    return EpisodeRecorder(
        Path(root).parent,
        run_id,
        task_name,
        episode_id,
        seed,
        frequency,
        timestep,
        int(horizon) if horizon else None,
        int(executed) if executed else None,
        str(video_path) if video_path else None,
    )


def record_action(
    recorder: EpisodeRecorder | None,
    *,
    task_env: Any,
    observation: Mapping[str, Any],
    raw_chunk: Any,
    sent_action: Any,
    action_type: str,
    chunk_id: int,
    chunk_start: int,
    action_index: int,
    executed_length: int,
) -> None:
    if recorder is None:
        return
    recorder.record_step(
        simulator_step=getattr(task_env, "step_count", None),
        control_step=getattr(task_env, "take_action_cnt", 0),
        observation=observation,
        raw_action_chunk=raw_chunk,
        sent_action=sent_action,
        action_type=action_type,
        chunk_id=chunk_id,
        chunk_start_step=chunk_start,
        action_index_in_chunk=action_index,
        executed_length=executed_length,
        contacts=extract_contacts(task_env),
        object_states=extract_task_state(task_env),
        joint_velocities=extract_joint_velocities(task_env),
    )


def finish_episode(
    recorder: EpisodeRecorder | None,
    *,
    success: bool,
    timed_out: bool,
    error: str | None,
) -> None:
    if recorder is None:
        return
    reason = (
        "infrastructure_error"
        if error
        else ("success" if success else ("timeout" if timed_out else "policy_failure"))
    )
    recorder.finish(
        success=success,
        termination_reason=reason,
        infrastructure_error=error,
    )
