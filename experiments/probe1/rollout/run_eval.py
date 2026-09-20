"""Validated thin launcher for the existing RoboTwin/XPolicyLab evaluator."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any
from .recorder import build_manifest, write_json_exclusive

ROOT = Path(__file__).resolve().parents[3]

def load_config(path: Path) -> dict[str, Any]:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"task_name", "task_config", "embodiment", "baseline", "policy_name", "action_type", "eval_seed", "episodes", "control_frequency_hz"}
    missing = required - set(data or {})
    if missing: raise ValueError(f"missing config fields: {sorted(missing)}")
    if data["embodiment"] != "aloha-agilex": raise ValueError("M0 requires the registry embodiment 'aloha-agilex'")
    if data["task_name"] != "handover_block": raise ValueError("M0 is frozen to the registry task 'handover_block'")
    if int(data["episodes"]) < 1: raise ValueError("episodes must be positive")
    return data

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"experiments/probe1/configs/m0_handover_block.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Verified task-SFT pi0.5 checkpoint")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-conda-env", required=True); parser.add_argument("--eval-conda-env", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args=parser.parse_args(argv); config=load_config(args.config)
    if not args.checkpoint.exists(): parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not (ROOT/"XPolicyLab/policy"/config["policy_name"]).is_dir(): parser.error(f"XPolicyLab adapter is unavailable: {config['policy_name']}; initialize the pinned submodule first")
    run_id=time.strftime("m0-%Y%m%dT%H%M%SZ", time.gmtime()); run_dir=args.output_root/run_id
    manifest=build_manifest(repo_root=ROOT, run_id=run_id, config_path=args.config, checkpoint=args.checkpoint, output_dir=run_dir,
      baseline=config["baseline"], training_seed=None, eval_seed=int(config["eval_seed"]), action_mapping={"order":"left_arm,left_gripper,right_arm,right_gripper","type":config["action_type"],"units":"adapter-defined; verify before rollout"},
      state_mapping={"order":"left then right","images":"RGB; head,left wrist,right wrist"}, prediction_horizon=config.get("prediction_horizon"), executed_chunk_length=config.get("executed_chunk_length"), control_frequency_hz=float(config["control_frequency_hz"]), physics_timestep_s=1/250)
    if not args.dry_run and (config.get("prediction_horizon") is None or config.get("executed_chunk_length") is None): parser.error("prediction_horizon and executed_chunk_length must be verified and set before rollout")
    run_dir.mkdir(parents=True, exist_ok=False); write_json_exclusive(run_dir/"manifest.json", manifest)
    command=["bash",str(ROOT/"scripts/eval_policy.sh"),"multitask","--config",str(ROOT/"experiments/probe1/configs/m0_eval.yaml"),"--policy-name",config["policy_name"],"--ckpt-name",str(args.checkpoint),"--env-cfg-type","arx_x5","--policy-conda-env",args.policy_conda_env,"--eval-env-conda-env",args.eval_conda_env,"--action-type",config["action_type"]]
    (run_dir/"command.json").write_text(json.dumps(command,indent=2)+"\n",encoding="utf-8")
    print(" ".join(command))
    if args.dry_run: return 0
    env=os.environ.copy(); env["PROBE1_OUTPUT_DIR"]=str(run_dir); env["PROBE1_RUN_ID"]=run_id
    env["PROBE1_PREDICTION_HORIZON"]=str(config["prediction_horizon"])
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
if __name__ == "__main__": raise SystemExit(main())
