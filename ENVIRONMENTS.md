# Probe 1 Conda environments

Probe 1 uses two isolated environments because RoboTwin/SAPIEN and OpenPI have
different dependency stacks. The commands below install repository code only;
they do not download model weights, run training, or start evaluation.

> The environment names are examples. Pass the same names to
> `--eval-conda-env` and `--policy-conda-env` when launching M0.

## 1. Prerequisites and immutable paths

- Linux x86-64 with an NVIDIA driver compatible with the selected PyTorch/CUDA
  build. `nvidia-smi` must work on the machine that runs simulation/policy.
- Conda or Miniforge. Python 3.10 is used because the repository pins
  SAPIEN 3.0.0b1, SciPy 1.10.1, Torch 2.4.1 and NumPy 1.26.4.
- This RoboTwin checkout, including the **pinned** XPolicyLab submodule:

  ```bash
  cd /workspace/RoboTwin_probes
  git submodule sync -- XPolicyLab
  git submodule update --init XPolicyLab
  git submodule status XPolicyLab
  ```

  Do not use `git submodule update --remote`: it changes the tested framework
  version. The expected submodule commit in this checkout is
  `fa431ecd893ee706883e64fe5fe1464ec8cd928d`.
- An OpenPI source checkout compatible with the SFT checkpoint. A weight
  directory is not a source checkout and cannot be installed as a package.
- The intended SFT checkpoint. Before evaluation, record its provenance and
  verify that it is the target-task SFT model rather than base π0.5 weights.

The paths in the project instructions are deployment defaults, not hard-coded
by Probe 1:

```bash
export ROBOTWIN_ROOT=/workspace/RoboTwin_probes
export OPENPI_ROOT=/path/to/openpi-source
export PI05_CHECKPOINT=/home/data_ssd1/zyj/openpi_model
export PROBE1_OUTPUT_ROOT=/home/data_ssd1/zyj/RoboTwin_probes
```

## 2. RoboTwin simulator environment

Create an environment and install the repository's pinned requirements:

```bash
conda create -n robotwin-probe1 python=3.10 pip -y
conda activate robotwin-probe1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROBOTWIN_ROOT/scripts/requirements.txt"
python -m pip install -e "$ROBOTWIN_ROOT/XPolicyLab"
```

Run RoboTwin's existing native-library setup only after reviewing its local
SAPIEN/MPLib paths:

```bash
cd "$ROBOTWIN_ROOT"
bash scripts/_install.sh
python scripts/update_embodiment_config_path.py
```

Basic import checks:

```bash
python -c "import numpy, sapien, torch, yaml; print(numpy.__version__, sapien.__version__, torch.__version__)"
python -m compileall -q experiments/probe1 scripts/eval_policy_xpolicylab.py
python -m pytest experiments/probe1/tests -q
```

## 3. OpenPI / π0.5 policy environment

Use the dependency lock and installation command from the **actual OpenPI
checkout**. Do not duplicate guessed OpenPI/JAX versions in RoboTwin. For an
OpenPI checkout that supports editable pip installation, the minimal Conda
wrapper is:

```bash
conda create -n openpi-probe1 python=3.11 pip -y
conda activate openpi-probe1
cd "$OPENPI_ROOT"
python -m pip install --upgrade pip
python -m pip install -e .
```

If that checkout documents `uv sync`, run it inside this environment instead
of the editable-pip command. Use only one OpenPI lock/install workflow. Then
install or expose the π0.5 adapter shipped by the pinned XPolicyLab checkout;
select the adapter directory's exact name in
`experiments/probe1/configs/m0_handover_block.yaml`.

Policy-side checks must be adapted to the checked-out OpenPI API, but should at
least establish:

```bash
test -d "$PI05_CHECKPOINT"
python -c "import jax; print(jax.devices())"
```

Before a rollout, record and freeze:

- checkpoint identity/hash and whether it is single-task or multi-task SFT;
- observation image keys, RGB order and resolution;
- state dimensionality and left/right ordering;
- action ordering, units, absolute/incremental semantics, gripper polarity and
  normalization statistics;
- model prediction horizon `H` and evaluator execution length `K`.

## 4. Probe 1 preflight and M0

Activate either environment from a shell where `conda run` can see both named
environments, set the verified `policy_name`, `prediction_horizon` and
`executed_chunk_length` in the M0 config, then run scheduler validation:

```bash
cd "$ROBOTWIN_ROOT"
python -m experiments.probe1.rollout.run_eval \
  --checkpoint "$PI05_CHECKPOINT" \
  --output-root "$PROBE1_OUTPUT_ROOT" \
  --policy-conda-env openpi-probe1 \
  --eval-conda-env robotwin-probe1 \
  --dry-run
```

Remove `--dry-run` only after preflight succeeds. M0 is one episode by default;
increasing it to a costly run is a separate authorization decision.

## 5. Troubleshooting and reproducibility capture

Capture both environments alongside each run without modifying them:

```bash
conda list -n robotwin-probe1 --explicit > robotwin-explicit.txt
conda list -n openpi-probe1 --explicit > openpi-explicit.txt
git -C "$ROBOTWIN_ROOT" rev-parse HEAD
git -C "$ROBOTWIN_ROOT/XPolicyLab" rev-parse HEAD
```

- **No GPU / `nvidia-smi` missing:** offline tests and analysis can run, but do
  not report them as a simulator smoke test.
- **XPolicyLab adapter absent:** initialize the pinned submodule and use its
  real adapter name; do not create a guessed `policy/openpi` path.
- **Checkpoint loads but success is low:** first verify SFT provenance and the
  complete image/state/action contract. Base-model failure is not evidence of a
  coordination bottleneck.
- **Missing contacts/object fields:** retain `null` plus availability flags.
  Never replace missing measurements with zero.
