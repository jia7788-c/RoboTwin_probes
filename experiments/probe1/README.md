# Probe 1 — bimanual coordination diagnostics

This directory implements the first, deliberately small M0 integration: the
unmodified joint π0.5 SFT policy on RoboTwin's registered `handover_block` task
and `aloha-agilex` embodiment. It does **not** implement B1/B2, train a model,
launch RL, choose the later task set, or claim experimental findings.

## Environment audit (2026-09-20)

| Check | Observed in this checkout |
| --- | --- |
| RoboTwin commit | `ae5fb15300eb03258afd6a564387c941ed9856db` |
| Branch | local branch `work` (requested upstream branch `probe1` is not present locally) |
| XPolicyLab | pinned submodule `fa431ecd893ee706883e64fe5fe1464ec8cd928d`, not initialized |
| Evaluation entry | `scripts/eval_policy.sh` → `scripts/eval_policy_xpolicylab.py` |
| Task / success | `envs/handover_block.py`; exact registry ID `handover_block` |
| Embodiment | `demo_clean.yml` declares exact ID `aloha-agilex`; XPolicyLab profile is `arx_x5` |
| Images | evaluator maps head/left wrist/right wrist to `cam_head`, `cam_left_wrist`, `cam_right_wrist`; arrays are already RGB |
| State/action | left arm + gripper followed by right arm + gripper; `joint` maps to RoboTwin `qpos` |
| Physics | Base task default is 1/250 s; manifest records the configured value and the runtime hook queries the scene |
| Runtime | host Python 3.14.4; no Conda command, NVIDIA tooling, GPU, RoboTwin runtime dependencies, checkpoint, or output disk visible |
| Checkpoint | `/home/data_ssd1/zyj/openpi_model` is absent; SFT provenance therefore could not be verified |
| Contacts/snapshots | generic observations expose neither contact forces nor complete simulator restore state; logged null/unavailable and `replay_verified=false` |

Consequently no real simulator smoke test was run here. Synthetic tests are
never written under the experiment output root and are not experimental data.

## M0 launch

Initialize the **pinned** XPolicyLab submodule using the repository's documented
procedure; do not silently update it. Confirm that its policy catalog contains
the configured π0.5 adapter name. Inspect the target checkpoint metadata to
establish that it is task SFT rather than a base model. Then copy the config or
edit it before the pilot:

1. Set `prediction_horizon` to the adapter/model value `H`.
2. Set `executed_chunk_length` to the evaluator value `K`.
3. Keep formal event thresholds null until an independent pilot calibrates and
   freezes them. Null values intentionally fail closed for formal analysis.
4. Verify action units, normalization, gripper direction, image sizes and
   channel order against the actual adapter and checkpoint.

Validate the command without starting a server or simulator:

```bash
python -m experiments.probe1.rollout.run_eval \
  --checkpoint /path/to/verified-task-sft-checkpoint \
  --output-root /path/to/probe1-output \
  --policy-conda-env openpi --eval-conda-env robotwin --dry-run
```

Remove `--dry-run` only after the two horizon fields and adapter contract are
verified. The launcher creates a unique run manifest before delegating to the
existing scheduler. `PROBE1_*` environment variables activate an optional hook
inside the ordinary single-worker loop; normal evaluations remain unchanged.
Each attempted episode gets append-only `steps.jsonl` and `episode.json` files.
Infrastructure errors remain distinguishable from valid timeout failures.

The current hook covers M0's single-worker path. Batch evaluation is explicitly
out of scope until its recorder lifecycle has an integration test. Object poses,
contacts/forces, velocities, phase labels and image indices remain null unless a
future task-specific simulator extractor supplies real values.

## Offline tests

```bash
python -m pytest experiments/probe1/tests -q
python -m experiments.probe1.rollout.run_eval --help
```

Tests cover the success/failure contract, ambiguous earliest evidence, outcome
deduplication, contact chatter, sampling-window conversion, early release,
never-grasped handling, empty/all/no-failure metrics, duplicate episode IDs,
deterministic clustered bootstrap, explicit missingness, JSONL readability and
append-only episode directories. Fixtures are synthetic.
