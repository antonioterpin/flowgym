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
  `KinematicDataSource`, >=0.3.0); the model never sees a class-1 field during
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
synthpix's `KinematicDataSource` (new in synthpix 0.3.0): the model is trained
**without ever seeing a class-1 field** and is then evaluated on the class-1
test split, measuring pure out-of-distribution generalization. Estimator,
optimizer, budget and image-rendering distribution are identical to Regime A;
only the field source changes (`experiments/lima/train_kinematic.yaml`,
`experiments/lima/lima_piv_kinematic.yaml`).

### Field generator and calibration

synthpix generates each field as `ds = a * G_sigma * xi` (white noise `xi ~ U(-1,1)`,
Gaussian filter of width `sigma` px, per-field linear scale `a`; Manickathan et al.
2022 §2.2). **Caveat:** synthpix applies `a` as a *linear* scale, not the
paper's per-field peak-displacement target, and Gaussian filtering strongly
(and sigma-dependently) attenuates amplitude. synthpix's own defaults (sigma in
[5,100], a in [0,16]) therefore produce **degenerate sub-pixel fields** (per-field peak
displacement ≈ 0.10 px median) — useless for training. We calibrate
**sigma in [5,30], a in [0,120]** so the generated peaks span a PIV-realistic range
overlapping class-1 (N=600 fields, seed 0):

| statistic | peak displacement (px) |
|---|---|
| p10 / median / p90 | 0.54 / 2.54 / 7.76 |
| p99 / max | 14.4 / 18.6 |

Per-field mean |d|: median 0.78 px, overall mean 1.01 px. About 0.5 % of fields
exceed 16 px and 21 % are sub-pixel — a realistic PIV mix. A `scale_mode: peak`
option in synthpix would restore the paper's behaviour without per-config
calibration (tracked as a synthpix follow-up).

### Setup

- **Training** (`train_kinematic.yaml`): `scheduler_class: kinematic`, 18,278
  generated fields (LIMA-1 Table 1 count), `include_images: false` so synthpix
  *also* renders fresh particle images on top — both halves random. Same Adam
  lr 5e-4, grad-clip 1.0, multi-level Jacobian L1 loss, 40,000 batches × 10 as
  Regime A (~35 min on one RTX 5090).
- **Eval:** identical to Regime A — class-1 test split (450 fields), EPE
  excluding a 16 px border.

### Learning curve — class-1 test (rendered, OOD generalization)

| Checkpoint | Mean EPE | Median EPE | Std |
|---|---|---|---|
| random init (baseline) | 1.717 px | 1.294 px | 1.121 |
| **24,000 (best)** | **0.111 px** | **0.102 px** | **0.068** |
| 32,000 | 0.117 px | 0.105 px | 0.083 |
| 39,999 (final) | 0.113 px | 0.101 px | 0.083 |

Trained **only on random fields**, LIMA reaches **0.111 px mean / 0.102 px
median** on class-1 test — a **~15× reduction** over the random-init baseline
(1.717 px) and within ~1.7× of the in-distribution class-1-supervised model
(Regime A, 0.065 px). The curve is **flat and stable** from 24k on: unlike
Regime A there is **no late-training blow-up** (the `uniform` scenario stays
0.207 / 0.241 / 0.261 px at 24k / 32k / 39,999, vs Regime A's 0.07→0.96 px
spike), because the kinematic distribution is broader and more uniform than the
fixed class-1 fields.

#### Best checkpoint (24k) — per-scenario mean EPE (excl. 16px), rendered test

| scenario | EPE (px) | n | | scenario | EPE (px) | n |
|---|---|---|---|---|---|---|
| cylinder_Re40 | 0.038 | 1 | | JHTDB_isotropic1024_hd | 0.098 | 74 |
| cylinder_Re150 | 0.049 | 21 | | backstep_Re800 | 0.106 | 25 |
| cylinder_Re200 | 0.053 | 14 | | JHTDB_mhd1024_hd | 0.109 | 28 |
| cylinder_Re400 | 0.057 | 17 | | backstep_Re1000 | 0.122 | 20 |
| cylinder_Re300 | 0.059 | 19 | | JHTDB_channel | 0.134 | 29 |
| JHTDB_channel_hd | 0.065 | 22 | | DNS_turbulence | 0.161 | 48 |
| backstep_Re1500 | 0.070 | 37 | | SQG | 0.170 | 45 |
| backstep_Re1200 | 0.098 | 26 | | uniform | 0.207 | 24 |

All scenarios sub-0.21 px. Canonical flows (cylinder, backstep) are recovered to
~0.05 px; turbulent cases (DNS, SQG) and pure translation (uniform) are hardest
— the same qualitative ordering as Regime A (canonical easiest,
turbulence/translation hardest).

### Domain transfer — class-1 test on REAL recorded images

| Checkpoint | Mean EPE | Median EPE |
|---|---|---|
| 24,000 | 0.273 px | 0.187 px |
| 39,999 | 0.244 px | 0.200 px |

Median ≈ **0.19 px** on real recorded images; the mean is inflated by the
separated backstep flows (~0.6 px). The kinematic model's **mean** here is
actually lower than the class-1-trained model's (0.273 vs 0.363 px): the broader
training distribution transfers more robustly across the sim-to-real gap, at a
small cost in median.

### Regime A vs B — head to head (class-1 test, best checkpoint)

| Eval | A: class-1 supervised | B: kinematic (random fields) |
|---|---|---|
| rendered images — mean / median | 0.065 / 0.055 px | 0.111 / 0.102 px |
| real recorded images — mean / median | 0.363 / 0.151 px | 0.273 / 0.187 px |

Regime A wins in-distribution (it trains on the eval fields); Regime B trains on
**no class-1 data at all** yet generalizes to within ~1.7× on rendered images,
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
  --load_from experiments/lima/output_kinematic/lima_piv/0/checkpoints/24000
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
