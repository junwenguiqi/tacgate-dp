# TacGate-DP

**Time–Modality Gated Tactile Conditioning for Diffusion Policies**

[中文](README.md) | **English** | [Method](docs/METHOD.md) | [Related work and contribution boundaries](docs/RELATED_WORK.md)

TacGate-DP learns how strongly historical tactile observations participate in
action generation. Left and right tactile CNN features are gated at each
observation time, fused with camera features and robot state, and injected into
the Diffusion Policy action U-Net through its global FiLM conditioning path.
The same observation-conditioned Router also gates the camera and state tokens.

This repository contains the encoders, two-stage training, checkpoints,
held-out evaluation, and a robot adapter interface. It is self-contained and
can be published as a standalone GitHub repository. The Python distribution is
named `tacgate-dp`; the import package remains `tactile_gate_policy`.

The current release has been checked with synthetic engineering fixtures.
Trained task weights, real robot evaluation results, and datasets are supplied
by the user rather than bundled with the repository.

## Architecture and tactile injection

![TacGate-DP architecture](figures/model_architecture.png)

[Vector PDF](figures/model_architecture.pdf) · [Editable SVG](figures/model_architecture.svg) · [LaTeX figure and caption](figures/model_architecture.tex)

The tactile conditioning path is:

```text
Left/right distributed tactile values
  -> independent 1D CNN encoders
  -> tactile observation tokens
  -> time–modality soft gates
  -> joint visual/tactile/state Transformer fusion
  -> global condition c
  -> FiLM in the DP action U-Net
  -> action-noise prediction and reverse DDPM sampling
```

The default configuration uses four observed time steps and five modalities,
with one 256-dimensional token per time/modality pair:

| Component | Default behavior |
|---|---|
| Global and wrist RGB encoders | Two independent DP-style ResNet18 backbones, without the classification head, followed by a 256-dimensional projection |
| Left and right tactile encoders | Two independent Conv1d CNNs over taxel indices, with widths 32/64/128, GroupNorm, GELU, global pooling, and token projection |
| Robot-state encoder | MLP from 22 raw proprioceptive values to a 256-dimensional token, after standardization |
| Observation tokens | `Z_obs: (B, 4, 5, 256)`, or 20 tokens per sample |
| Router | Learned time/modality embeddings, a two-layer four-head Transformer, and one independent Sigmoid gate per token |
| Soft gate | `Z_g = G * Z_obs + (1 - G) * mask`, with a learned per-modality mask token |
| Fusion | A separate two-layer four-head Transformer with its own time/modality embeddings; mean pooling over the 20 output tokens and projection to `c: (B, 256)` |
| Action diffusion | DP `ConditionalUnet1D`, conditioned through FiLM on the global condition and diffusion-step embedding; 100 reverse DDPM steps by default |
| Action horizon | Predict 8 actions and execute the first 4 before replanning |

The modality order is fixed:

```text
global_rgb, wrist_rgb, left_tactile, right_tactile, proprio
```

The Router reads the full observation history to produce `G: (B, 4, 5)`.
Its gate values blend the original encoder tokens with learned mask tokens;
all 20 token positions remain in the fusion sequence. Self-attention then
allows information exchange across both observation times and modalities.
The resulting condition `c` includes visual, tactile, and state information.
It is the input to the existing DP FiLM path, rather than a separate tactile
adapter inside the U-Net.

At inference, both `G` and `c` are computed once per predicted action chunk
and reused throughout reverse diffusion. The FiLM parameters additionally
depend on the diffusion step `k`. Only actions are diffused in this release.

The RGB pipeline follows DP's ResNet18, optional GroupNorm replacement,
random training crops, center evaluation crops, and ImageNet normalization.
The default is `vision_weights: null`, so encoders train from scratch without
downloading weights. Setting `vision_weights: IMAGENET1K_V1` explicitly uses
Torchvision pretrained weights. Checkpoint loading restores saved weights
without a fresh pretrained-weight download.

Each tactile input has 231 distributed values. By default it is interpreted
as **77 taxels × 3 interleaved channels**, and convolution runs along the taxel
index sequence. This assumes the source uses that ordering; index proximity
does not establish physical taxel proximity. Use `tactile_channels: 1` or adapt
the input ordering if the sensor layout differs. The three leading force
channels in the 234-value raw record are excluded.

See [the method description](docs/METHOD.md) for equations, training details,
and the precise scope of tactile injection.

## Installation

Use Python 3.9 or newer with compatible PyTorch and Torchvision versions.
Run the following commands from the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For CUDA training, install the appropriate PyTorch/Torchvision pair for your
system before installing the package. The package does not require sibling
research repositories or developer-specific absolute paths.

## Verify the local pipeline

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

The smoke script generates temporary PNG, Parquet, and Zarr fixtures, runs a
short Stage A job, resumes it, transfers to Stage B, performs reverse-diffusion
evaluation, and executes a Toy adapter rollout. It removes its temporary data
after completion. Tests use `unittest` and do not require `pytest`.

[VALIDATION.md](VALIDATION.md) records the environment and checks already run.
The recorded CPU run passed 23 tests, including encoder/Router gradients,
split isolation, numerical normalization, deterministic sampling, checkpoint
continuation, stage transfer, and the Toy execution loop. These checks verify
the implementation pipeline; they do not measure manipulation success.
Windows/Linux CPU GitHub Actions are configured. Remote CI, CUDA, and Zarr 3
are not covered by the recorded local validation.

## FR3/Paxini dataset format

```text
dataset_root/
├── data/episodes/episode_000000.parquet
├── images/...png
└── tactile/...zarr
```

Each episode Parquet file is the synchronization table. Required fields are:

| Field | Meaning |
|---|---|
| `frame_index`, `timestamp`, `sync.valid` | Frame index, timestamp in seconds, synchronization validity |
| `image.path`, `wrist_image.path` | Actual RGB image paths relative to `dataset_root` |
| `robot.ee_pose`, `robot.q`, `robot.dq`, `robot.gripper_pos` | 7 + 7 + 7 + 1 = 22 proprioceptive values |
| `action.sent_action8` | Absolute `x, y, z, qx, qy, qz, qw, gripper` action |
| `tactile.left/right.zarr_path`, `tactile.left/right.zarr_index` | Relative Zarr path and array row index for each tactile sensor |
| `tactile.left/right.force_raw_len` | Must be 3 |
| `tactile.left/right.distribution_raw_start` | Must be 3 |
| `tactile.left/right.distribution_raw_len` | Must be 231 |

Here `left/right` denotes separate `left` and `right` field names. Tactile
Zarr arrays have shape `(T, 234)`; the last 231 entries contain distributed
tactile values in `[0, 255]`. The dataset reads real image pixels and tactile
arrays, resolves paths within `dataset_root`, and rejects inconsistent numeric
values or timestamps. Windows-relative path separators are also supported.

Each default window contains four historical observations followed immediately
by eight future action labels. Windows do not cross episode or split boundaries,
and any window containing synchronization-invalid sampled observation or
action rows is filtered out. Multiple episodes are split by episode; a single
episode is split into disjoint contiguous time blocks. Proprioception and action
mean/std statistics are fitted from the training split only and saved in the
checkpoint. Tactile values are scaled by 255.

**Match the data and control intervals.** The supplied YAML files set
`frame_stride: 3` and `control_dt: 0.3`, matching approximately 10 Hz raw data
sampled every three frames. To retain a 10 Hz control interval, change both to
`frame_stride: 1` and `control_dt: 0.1`. Training validates the actual timestamp
interval against `control_dt`.

## Two-stage training

### Stage A: random-gate action pretraining

Stage A bypasses and freezes the Router. It samples binary Bernoulli gates
using a per-sample keep probability drawn uniformly from `[0.35, 0.95]`.
If a time row loses every modality, one modality is restored at random.
The mask tokens, fusion Transformer, and action U-Net are trained with the
action-noise prediction objective. Encoders also train by default, as the
default visual backbones have no pretrained weights.

```bash
python -m tactile_gate_policy.train --config configs/stage_a.yaml --dataset-root /path/to/dataset --output-dir runs/stage_a --device cuda:0
```

Use `--device cpu` for CPU execution.

### Stage B: learned time–modality gates

Stage B loads a compatible checkpoint and enables the Router. The supplied
configuration freezes the pretrained encoders while training the Router,
mask tokens, fusion Transformer, and action U-Net. Set
`training.freeze_encoders: false` to fine-tune the encoders jointly.

```bash
python -m tactile_gate_policy.train --config configs/stage_b.yaml --dataset-root /path/to/dataset --output-dir runs/stage_b --init-checkpoint runs/stage_a/best.pt --device cuda:0
```

The default Stage B objective is:

```text
L = action_noise_prediction_MSE
  + 0.1 * (mean(G) - 0.65)^2
  + 0.02 * mean(abs(G[:, 1:] - G[:, :-1]))
```

The gate-mean term controls the average retention strength; the smoothness
term compares adjacent historical times within each observation window.
The optional left/right tactile symmetry term has coefficient zero by default.
These are learning objectives, not guarantees of robustness or compute savings.
Training uses AdamW, warmup/cosine learning rates, and gradient clipping.

### Checkpoints, transfer, and continuation

Use `--init-checkpoint` to transfer between stages or start a new fine-tuning
run with new optimizer state. Both the policy configuration and dataset
root/split must match so saved normalization and validation isolation remain
valid. Stage B requires `--init-checkpoint` or `--resume`.

Use `--resume` to continue an interrupted job in the same stage:

```bash
python -m tactile_gate_policy.train --config configs/stage_b.yaml --dataset-root /path/to/dataset --output-dir runs/stage_b --resume runs/stage_b/last.pt --device cuda:0
```

Checkpoints include the policy and encoders, normalization buffers, optimizer,
learning-rate scheduler, split manifests, epoch/batch/global-step progress,
and RNG state. With fixed data and the same execution environment, resumed
training skips completed batches and restores the recorded state.

Resume requires matching policy, training, loss, and data configuration, with
exceptions for `device`, `cpu_threads`, and `max_steps`. In particular, changing
the epoch count is a fine-tuning operation rather than a strict resume.
`--max-steps N` stops a run at global step N; a later resume may increase N.

Each stage writes `last.pt`, `best.pt` selected by validation loss,
`config.yaml`, and `metrics.jsonl`. Existing training output requires explicit
resume or a new output directory.

## Held-out evaluation and inference

```bash
python -m tactile_gate_policy.evaluate --checkpoint runs/stage_b/best.pt --dataset-root /path/to/dataset --output-dir runs/evaluation --device cuda:0
```

Evaluation uses the held-out split recorded in the checkpoint. It writes
`metrics.json` and `predictions.npz`, including full predicted action chunks
and gates. Metrics include action MSE, position MAE, gripper MAE, quaternion
angular error in radians, and the mean time–modality gate matrix. Position and
gripper errors use the source dataset units. These measure demonstration-action
agreement, rather than closed-loop task success. The gate matrix is numerical
output; plotting it is a separate visualization step.

Programmatic prediction accepts a dictionary of tensors:

```python
from tactile_gate_policy.checkpoint import load_policy

policy, metadata = load_policy("runs/stage_b/best.pt", device="cpu")

# obs contains all five modality keys, on the policy device:
# global_rgb / wrist_rgb: (B, 4, 3, H, W), float in [0, 1] or uint8
# left_tactile / right_tactile: (B, 4, 231), float in [0, 255]
# proprio: (B, 4, 22), raw float values
# Numeric observation fields have not yet been normalized.
prediction = policy.predict_action(obs)
actions_to_execute = prediction["action"]      # (B, 4, 8)
complete_chunk = prediction["action_pred"]     # (B, 8, 8)
gates = prediction["gate"]                    # (B, 4, 5)
condition = prediction["condition"]           # (B, 256)
```

Actions are standardized during training. DDPM uses `clip_sample=False` to
preserve the standardized action range. Prediction denormalizes the action
chunk and normalizes each quaternion to unit length; a near-zero quaternion
falls back to the identity in `xyzw` order.

## Receding-horizon execution

The in-memory Toy adapter checks observation buffering, action prediction,
prefix execution, and history updates:

```bash
python -m tactile_gate_policy.rollout --checkpoint runs/stage_b/best.pt --adapter toy --max-steps 40
```

The rollout CLI requires a Stage B checkpoint with a trained Router.
For real FR3 hardware, provide a factory that implements the `RobotAdapter`
interface in `tactile_gate_policy.rollout`:

```python
class MyRobotAdapter:
    def observe(self):
        # Block until the next synchronized control-tick observation.
        # Return global_rgb, wrist_rgb, left_tactile, right_tactile, proprio.
        # Images may be HWC/CHW uint8 or float [0, 1].
        # Tactile: 231 distributed values, or 234 with three leading forces.
        # Proprio: 22 raw values in the training order.
        ...

    def execute(self, action8, control_dt):
        # Execute an absolute pose/gripper action in the training frame.
        # Implement hardware timing and constraints in this adapter.
        # Return True when the episode terminates.
        ...


def make_adapter(policy_config):
    return MyRobotAdapter()
```

```bash
python -m tactile_gate_policy.rollout --checkpoint runs/stage_b/best.pt --adapter my_robot_module:make_adapter --max-steps 40
```

The runner first collects a complete history, executes `n_action_steps` actions
per chunk, and receives a new observation after each executed action.
The deployment environment supplies the robot SDK, communication connection,
coordinate conventions, and low-level controller. A ready-to-use physical
FR3 adapter is not included.

## Publication figure

The architecture figure is provided as vector PDF/SVG and a high-resolution
PNG preview. Its defaults reflect `PolicyConfig`, and the LaTeX file includes
an English caption for a paper. To regenerate it:

```bash
python -m pip install -e ".[figures]"
python figures/draw_model_architecture.py
```

Use the vector PDF for LaTeX/arXiv submissions and the SVG when editing labels.
After changing the architecture, update the drawing and caption to match the
implementation.

## Contributions, attribution, and current scope

TacGate-DP's implementation brings together an observation-conditioned gate
for every historical time/modality pair, separate routing and fusion
Transformers, random-gate action pretraining followed by learned-gate
fine-tuning, and a reproducible FR3/Paxini data-to-action pipeline. Its tactile
emphasis is the selection and strength of historical tactile features along
the fused conditioning path.

The DP visual-encoder pipeline and action U-Net are reused/adapted from
Diffusion Policy. Soft gating, learned mask tokens, attention fusion, and
visuo-tactile diffusion policies have prior work; this release does not claim
those individual mechanisms as new. Assessing the proposed combination
requires task experiments and controlled ablations. See
[related work and contribution boundaries](docs/RELATED_WORK.md) for relevant
comparisons and the recommended evidence.

The implementation currently uses observation-conditioned gates. Action-
conditioned gates, gate-controlled observation diffusion noise, and joint
generation of visual/tactile/action sequences are research extensions rather
than current features.

The project uses the MIT license. Third-party provenance and the vendored DP
license are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Datasets, checkpoints, local run outputs, and temporary validation artifacts
are excluded from version control.
