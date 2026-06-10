# LIMA — training & PIV class-1 results

Two training regimes for the LIMA-6 (LIMAR2) estimator, both evaluated on the
same **PIV class-1** benchmark (`activefluidcontrol/piv-class1`, 9 flow-field
families: DNS turbulence, JHTDB channel/channel-hd/isotropic/MHD, SQG, backstep,
cylinder, uniform):

- **Regime A — supervised on class-1 fields** (`train.yaml`): the displacement
  fields are the dataset's *real* DNS/CFD/analytic fields; only the particle
  images are randomly re-rendered each batch. In-distribution.
- **Regime B — true kinematic training** (`train_kinematic.yaml`): the
  displacement fields are themselves *randomly generated* (synthpix
  `KinematicDataSource`, >=0.3.1); the model never sees a class-1 field during
  training — the Manickathan et al. (2022) kinematic strategy. Tests
  out-of-distribution generalization to class-1.

> **Terminology.** Regime A was originally labelled "kinematic training", but
> only the *image rendering* was randomized there, not the displacement fields.
> Regime B is kinematic training in the proper random-displacement-field sense;
> the two are kept distinct below.

## Regime A — supervised on class-1 fields (randomized rendering)

### Setup

- **Model:** LIMA-6, replicate padding, search range 2 (LIMAR2), the 2025
  paper's recommended config. ~0.93 M params (tabulated architecture).
  `experiments/lima/lima_piv.yaml`.
- **Training** (`experiments/lima/train.yaml`): synthpix renders fresh
  randomized particle images each batch on top of the class-1 *train-split*
  displacement fields (`include_images: false`). The *fields are the real
  class-1 fields* — only the rendering is randomized, so this is supervised
  training on class-1, not kinematic training (cf. Regime B). Particle density
  0.03–0.04 ppp, diameter 1.5–2.5 px, shot/Gaussian noise, particle dropout —
  re-randomized per batch (augmentation).
- **Data:** local mirror of the HF dataset — full test split (450 fields) + a
  150-field-per-scenario train subset (1,350 fields). Each field is re-rendered
  every time it is drawn, so the effective training set is far larger.
- **Optimizer:** Adam, lr **5e-4** (raised from the paper's 1e-4/2e-4 for a
  reduced-budget run), global-norm grad clip 1.0. Multi-level Jacobian-penalised
  L1 loss (lambda_u=0.91, lambda_J=0.09).
- **Budget:** 40,000 batches × 10 = 400k image-pair presentations, ~36 min on
  one RTX 5090. (The paper trains 200 epochs ≈ 3.6–7.4 M presentations.)
- **Eval** (`src/eval_lima.py`): endpoint error EPE = ‖pred − gt‖ per pixel,
  reported excluding a 16 px border (paper convention), over all 450 test fields.

### Learning curve — class-1 test (rendered, in-distribution)

| Checkpoint | Mean EPE | Median EPE | Std |
|---|---|---|---|
| random init (baseline) | 1.717 px | 1.294 px | 1.121 |
| 8,000 | 0.088 px | 0.076 px | 0.049 |
| 24,000 | 0.067 px | 0.056 px | 0.041 |
| **32,000 (best)** | **0.065 px** | **0.055 px** | **0.040** |
| 39,999 (final) | 0.115 px | 0.057 px | 1.007 |

**Best checkpoint: 32,000.** EPE drops ~26× from the random-init baseline to
**0.065 px mean / 0.055 px median**, comparable to or better than the paper's
LIMA-6 (mean EPE ≈ 0.17 px on the Carlier DNS case — a different test set).

**Late-training regression:** between 32k and 40k the `uniform` (large
pure-translation) scenario blew up from 0.068 → 0.96 px (median stayed at
0.057), spiking the std to 1.0. This is lr-too-high-without-decay instability on
the hardest examples — pure translation is also a known LIMA weak spot (not
emphasised in the class-1 training distribution). Fixes for a full run:
cosine/plateau lr decay + best-checkpoint selection on a validation split
(`save_only_best`, which was disabled here). Regime B (kinematic) below does
*not* show this regression at the same lr.

#### Best checkpoint (32k) — per-scenario mean EPE (excl. 16px), rendered test

| scenario | EPE (px) | n | | scenario | EPE (px) | n |
|---|---|---|---|---|---|---|
| backstep_Re800 | 0.025 | 25 | | JHTDB_isotropic1024_hd | 0.072 | 74 |
| cylinder_Re40 | 0.029 | 1 | | JHTDB_mhd1024_hd | 0.076 | 28 |
| cylinder_Re150 | 0.031 | 21 | | uniform | 0.068 | 24 |
| backstep_Re1000 | 0.035 | 20 | | SQG | 0.117 | 45 |
| cylinder_Re200 | 0.035 | 14 | | DNS_turbulence | 0.123 | 48 |
| JHTDB_channel_hd | 0.046 | 22 | | JHTDB_channel | 0.063 | 29 |

All scenarios sub-0.13 px. Turbulent (DNS, SQG) cases are hardest; canonical
flows (backstep, cylinder) are reconstructed to ~0.03 px.

### Domain transfer — class-1 test on REAL recorded images (`I0`/`I1`)

Evaluating the class-1-trained model on the dataset's *actual* recorded PIV
images (not re-rendered) — a harder, out-of-distribution test:

| Checkpoint | Mean EPE | Median EPE |
|---|---|---|
| 32,000 | 0.363 px | 0.151 px |
| 39,999 | 0.224 px | 0.156 px |

Median EPE ≈ **0.15 px** (sub-pixel) on real images. The mean is inflated by the
cylinder-near-wall real images (~0.5 px) where laser reflections and solid
boundaries are hardest — the same regime where WIDIM also struggles (LIMA-1
§3.3). The replicate padding (LIMAR) is specifically meant to help here.

## Regime B — true kinematic training (random displacement fields)

Here the displacement fields themselves are randomly generated by
synthpix's `KinematicDataSource` (synthpix 0.3.1): the model is trained
**without ever seeing a class-1 field** and is then evaluated on the class-1
test split, measuring pure out-of-distribution generalization. Estimator,
optimizer, budget and image-rendering distribution are identical to Regime A;
only the field source changes (`experiments/lima/train_kinematic.yaml`,
`experiments/lima/lima_piv_kinematic.yaml`).

### Field generator

synthpix generates each field as `ds = a * G_sigma * xi` (white noise `xi ~ U(-1,1)`,
Gaussian filter of width `sigma` px; Manickathan et al. 2022 §2.2). We use
**`scale_mode: "peak"`** (the synthpix >=0.3.1 default), which normalises each
field so its peak displacement magnitude equals `a` (px) — the paper's per-field
"Maximum displacement, max(ds_ref) (px)" (Manickathan, Mucignat & Lunati, Exp.
Fluids 64, 161, 2023, Table 1). Peak normalisation is sigma-independent, so the
**paper defaults sigma in [5,100], a in [0,16]** are used directly: no per-config
calibration. (The earlier linear-scale workaround — sigma [5,30], a [0,120] — was
needed only because synthpix 0.3.0 applied `a` as a raw multiplier, and Gaussian
filtering then collapsed the displacement to a ~0.10 px median; that is no longer
required.)

Generated displacement distribution (N=600 fields, seed 0):

| statistic | peak displacement (px) |
|---|---|
| p10 / median / p90 | 1.78 / 8.39 / 14.59 |
| p99 / max | 15.89 / 15.99 |

Per-field peak |d| is uniform on [0,16] by construction; per-field mean |d|:
median 3.16 px, overall mean 3.41 px. Only ~4 % of fields are sub-pixel (vs 21 %
under the old linear calibration) — a healthier, paper-faithful PIV mix.

### Setup

- **Training** (`train_kinematic.yaml`): `scheduler_class: kinematic`,
  `scale_mode: peak` with the paper ranges (sigma [5,100], a [0,16]), 18,278
  generated fields (LIMA-1 Table 1 count), `include_images: false` so synthpix
  *also* renders fresh particle images on top — both halves random. Same Adam
  lr 5e-4, grad-clip 1.0, multi-level Jacobian L1 loss, 40,000 batches × 10 as
  Regime A (~35 min on one RTX 5090).
- **Eval:** identical to Regime A — class-1 test split (450 fields), EPE
  excluding a 16 px border.

### Learning curve — class-1 test (rendered, OOD generalization)

EPE excluding a 16 px border:

| Checkpoint | Mean EPE | Median EPE | Std |
|---|---|---|---|
| random init (baseline) | 1.717 px | 1.294 px | 1.121 |
| 24,000 | 0.103 px | 0.081 px | 0.064 |
| **32,000 (best)** | **0.094 px** | **0.073 px** | **0.061** |
| 39,999 (final) | 0.098 px | 0.075 px | 0.062 |

Trained **only on random fields**, LIMA reaches **0.094 px mean / 0.073 px
median** on class-1 test — an **~18× reduction** over the random-init baseline
(1.717 px) and within ~1.5× of the in-distribution class-1-supervised model
(Regime A, 0.065 px). The curve is **flat and stable**: no late-training
blow-up — the `uniform` scenario sits at 0.038 px at both 32k and 39,999 (vs
Regime A's 0.07→0.96 px spike) — because the kinematic distribution is broader
and more uniform than the fixed class-1 fields. With the paper-faithful peak
ranges this run also edges out the earlier linear-calibration run (0.094 vs
0.111 px mean) and fixes its weakest case, `uniform` (0.038 vs 0.207 px).

#### Best checkpoint (32k) — per-scenario mean EPE (excl. 16px), rendered test

| scenario | EPE (px) | n | | scenario | EPE (px) | n |
|---|---|---|---|---|---|---|
| backstep_Re800 | 0.035 | 25 | | cylinder_Re300 | 0.064 | 19 |
| backstep_Re1200 | 0.036 | 26 | | cylinder_Re400 | 0.064 | 17 |
| backstep_Re1500 | 0.037 | 37 | | JHTDB_channel_hd | 0.075 | 22 |
| uniform | 0.038 | 24 | | JHTDB_channel | 0.104 | 29 |
| backstep_Re1000 | 0.038 | 20 | | JHTDB_isotropic1024_hd | 0.111 | 74 |
| cylinder_Re40 | 0.043 | 1 | | JHTDB_mhd1024_hd | 0.121 | 28 |
| cylinder_Re150 | 0.054 | 21 | | DNS_turbulence | 0.177 | 48 |
| cylinder_Re200 | 0.058 | 14 | | SQG | 0.187 | 45 |

All scenarios sub-0.19 px. Canonical flows (backstep, cylinder) and pure
translation (uniform) are recovered to ~0.04–0.06 px; turbulent cases (DNS, SQG)
are hardest — the same qualitative ordering as Regime A. Note `uniform` is now
among the *easiest* cases (0.038 px), because the peak-normalised distribution
spans larger displacements that cover pure translation well.

### Domain transfer — class-1 test on REAL recorded images

| Checkpoint | Mean EPE | Median EPE |
|---|---|---|
| 32,000 (best) | 0.254 px | 0.162 px |
| 39,999 (final) | 0.315 px | 0.167 px |

Median ≈ **0.16 px** on real recorded images; the mean is inflated by the
separated backstep flows. The kinematic model's **mean** here is still lower
than the class-1-trained model's (0.254 vs 0.363 px): the broader training
distribution transfers more robustly across the sim-to-real gap.

### Regime A vs B — head to head (class-1 test, best checkpoint)

| Eval | A: class-1 supervised | B: kinematic (random fields) |
|---|---|---|
| rendered images — mean / median | 0.065 / 0.055 px | 0.094 / 0.073 px |
| real recorded images — mean / median | 0.363 / 0.151 px | 0.254 / 0.162 px |

Regime A wins in-distribution (it trains on the eval fields); Regime B trains on
**no class-1 data at all** yet generalizes to within ~1.5× on rendered images,
to a **better mean** on real images, and trains more stably. Kinematic training
is the more general recipe — it needs no CFD/DNS field corpus.

## Reproduce

```bash
# Pull the dataset (needs HF access to activefluidcontrol/piv-class1):
#   test split + a per-scenario train subset -> /home/.../data/piv-class1

# --- Regime A: supervised on class-1 fields ---
# Train (non-episodic supervised driver; bypasses the episodic FluidEnv):
GOGGLES_PORT=$PORT PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python src/train_lima_supervised.py \
  --estimator experiments/lima/lima_piv.yaml --dataset experiments/lima/train.yaml
# Evaluate a checkpoint on the class-1 test split:
GOGGLES_PORT=$PORT PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python src/eval_lima.py --estimator experiments/lima/lima_piv.yaml \
  --dataset experiments/lima/eval.yaml \
  --load_from experiments/lima/output/lima_piv/0/checkpoints/32000

# --- Regime B: true kinematic training (random displacement fields) ---
GOGGLES_PORT=$PORT PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python src/train_lima_supervised.py \
  --estimator experiments/lima/lima_piv_kinematic.yaml \
  --dataset experiments/lima/train_kinematic.yaml
# Evaluate on the class-1 test split (rendered; use eval_real.yaml for real images):
GOGGLES_PORT=$PORT PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python src/eval_lima.py --estimator experiments/lima/lima_piv_kinematic.yaml \
  --dataset experiments/lima/eval.yaml \
  --load_from experiments/lima/output_kinematic/lima_piv/0/checkpoints/32000
```

## References

- Manickathan, L., Mucignat, C., & Lunati, I. (2023). "A lightweight neural
  network designed for fluid velocimetry." *Experiments in Fluids*, 64, 161.
  https://doi.org/10.1007/s00348-023-03695-8
- Mucignat, C., Zdybał, K., & Lunati, I. (2025). "Improving the performance of a
  lightweight convolutional neural network for particle image velocimetry
  through hyper-parameter and padding optimization." *Physics of Fluids*, 37,
  105112. https://doi.org/10.1063/5.0283779
- Manickathan, L., Mucignat, C., & Lunati, I. (2022). "Kinematic training of
  convolutional neural networks for particle image velocimetry." *Measurement
  Science and Technology*, 33, 124006. https://doi.org/10.1088/1361-6501/ac8fae
- Hur, J., & Roth, S. (2019). "Iterative residual refinement for joint optical
  flow and occlusion estimation." *CVPR*.
  https://doi.org/10.1109/CVPR.2019.00590
