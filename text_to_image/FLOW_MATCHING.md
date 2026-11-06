# Flow matching with stochastic endpoints

Text is endpoint `x0`; image is endpoint `x1`. With independent standard
Gaussian tensors, training uses

```text
x0_noisy = x0 + flow_text_sigma  * epsilon0
x1_noisy = x1 + flow_image_sigma * epsilon1
xt       = (1 - t) * x0_noisy + t * x1_noisy
target   = x1_noisy - x0_noisy
```

Both directions regress this forward-time velocity. Reverse integration uses
negative time increments. Cross-attention receives the **clean source** (`x0`
forward, `x1` reverse). Inference perturbs the source once and then integrates
an ODE; it does not add Brownian noise at every step. Repeated draws at a fixed
source can therefore produce different outputs. This implements the conditional
noisy-base idea in [Albergo et al., §4.2 and Eqs. 18–20](https://arxiv.org/abs/2310.03725)
with a linear interpolant; it is not a claim to reproduce their full experiments.

`--flow-text-sigma` and `--flow-image-sigma` independently default to **0**.
They are per-coordinate standard deviations in the scaled bridge space (after
text normalization/scaling and image VAE scaling), not pixel-space noise.
Zero sigmas consume no extra random numbers and preserve the original
`--sde flow_matching --force-unconditional` baseline, including its shared field
for forward and reverse inference. Old checkpoints load with both sigmas zero.
Checkpoint evaluation and training resume reject changed endpoint sigmas.
The token decoder keeps its original, separate clean-input augmentation.

For conditional flow, each enabled direction must have positive **source**
sigma. A positive image sigma alone trains I2T (`--no-forward`); a positive text
sigma alone trains T2I (`--no-reverse`). Set both sigmas positive to train two
conditional directions. A conditional field is not automatically the inverse
conditional generator. If a destination sigma is positive, the learned target
distribution is Gaussian-smoothed too; the implementation does not remove that
noise afterward. The two supplied single-direction runs keep the target clean.
Partial-trajectory editing currently rejects these new noisy flows because its
restoration protocol would need a separate definition.

## Matched launchers

Run from `text_to_image/`, in the existing training environment:

| Variant | Train | Held-out GPIC evaluation | Sigmas (text, image) |
|---|---|---|---|
| Original | `bash _train_flow_original.sh` | `bash _eval_flow_original.sh` | (0, 0) |
| T2I, text noise | `bash _train_flow_text_noise.sh` | `bash _eval_flow_text_noise.sh` | (0.01, 0) |
| I2T, image noise | `bash _train_flow_image_noise.sh` | `bash _eval_flow_image_noise.sh` | (0, 0.01) |

These wrappers share `_flow_common.sh`, based on the original flow recipe in
`_launch_ablation_sweep.sh`: DiTXA-L/2, 200k steps, batch 512, LR 1.5e-4, 5k warmup,
EMA 0.9995, epsilon 9.9e-4, and image REPA weight 0.5 at layer 8. Training-time
evaluation retains the original defaults (100 ODE steps, 1,024 samples).
The new conditional runs use the existing 0.3 conditioning-dropout setting;
the original forces dropout to 1. All train one direction per optimizer step,
so these runs keep the original per-step training budget. The original shared
field additionally supports reverse inference; the conditional runs each
support their trained direction. Model geometry and other hyperparameters match.

Set `DATA_ROOT`, `DINO_DIR`, `OUT_DIR`, `NPROC` (default 8), and optionally
`PREFIX_DIR` (default `/data`). Evaluation requires `CHECKPOINT` and defaults to
the test split. Flags after the script name override recipe defaults. Examples:

```bash
DRYRUN=1 bash _train_flow_text_noise.sh
DATA_ROOT=/data/train DINO_DIR=/data/dino/train OUT_DIR=/data/runs/text_noise \
  bash _train_flow_text_noise.sh
CHECKPOINT=/data/runs/text_noise/TIMESTAMP/checkpoints/step_0200000.pt \
  DATA_ROOT=/data/test DINO_DIR=/data/dino/test \
  bash _eval_flow_text_noise.sh --eval-coco-cider --eval-spice
```

Replace `TIMESTAMP` with the training run's timestamp directory.

Standalone comparison evaluation defaults to 500 Euler steps for every flow
variant (`EVAL_STEPS` overrides), 50k FID images, and 10k I2T captions, matching
the original held-out evaluation's sample counts. Set the existing SDE/ODE
evaluation to the **same step count**, CFG, seed, and sample IDs. At CFG 0 there
is one network evaluation per step; positive CFG uses two. Use CFG 0 for the
primary comparison with unconditional flow, which cannot use CFG. Historical
100-step flow results must be rerun to compare with a 500-step result.

This is a reasonable stochastic conditional-flow baseline, but sigma 0.01 alone
does not establish a strong result. For a paper, select sigma/CFG on validation
data, report training compute and inference evaluations, and inspect diversity
across repeated samples of the same source. Do not select hyperparameters on
the held-out COCO test set or promise reviewer acceptance from this addition.
COCO launch and metric conventions are in [COCO_EVALUATION.md](COCO_EVALUATION.md).
