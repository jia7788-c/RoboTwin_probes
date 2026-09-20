"""Small optional hook used by RoboTwin's existing single-worker eval loop."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping
from .recorder import EpisodeRecorder

def start_episode(*, task_name: str, episode_id: int, seed: int, frequency: float, task_env: Any) -> EpisodeRecorder | None:
    root=os.environ.get("PROBE1_OUTPUT_DIR"); run_id=os.environ.get("PROBE1_RUN_ID")
    if not root or not run_id: return None
    timestep=None
    scene=getattr(task_env,"scene",None)
    getter=getattr(scene,"get_timestep",None)
    if callable(getter): timestep=float(getter())
    horizon=os.environ.get("PROBE1_PREDICTION_HORIZON")
    video_path=getattr(task_env,"eval_video_path",None)
    return EpisodeRecorder(Path(root).parent,run_id,task_name,episode_id,seed,frequency,timestep,int(horizon) if horizon else None,str(video_path) if video_path else None)

def record_action(recorder: EpisodeRecorder | None, *, task_env: Any, observation: Mapping[str, Any], raw_chunk: Any, sent_action: Any, action_type: str, chunk_id: int, chunk_start: int, action_index: int, executed_length: int) -> None:
    if recorder is None: return
    recorder.record_step(simulator_step=getattr(task_env,"step_count",None),control_step=getattr(task_env,"take_action_cnt",0),observation=observation,raw_action_chunk=raw_chunk,sent_action=sent_action,action_type=action_type,chunk_id=chunk_id,chunk_start_step=chunk_start,action_index_in_chunk=action_index,executed_length=executed_length)

def finish_episode(recorder: EpisodeRecorder | None, *, success: bool, timed_out: bool, error: str | None) -> None:
    if recorder is None: return
    reason="infrastructure_error" if error else ("success" if success else ("timeout" if timed_out else "policy_failure"))
    recorder.finish(success=success,termination_reason=reason,infrastructure_error=error)
