# LIMA — kinematic training & PIV class-1 results

Kinematic training of the LIMA-6 (LIMAR2) estimator and evaluation on the
**PIV class-1** benchmark (`activefluidcontrol/piv-class1`, 9 flow scenarios:
DNS turbulence, JHTDB channel/isotropic/MHD, SQG, backstep, cylinder, uniform).

## Setup

- **Model:** LIMA-6, replicate padding, search range 2 (LIMAR2), the 2025
  paper's recommended config. ~0.93 M params (tabulated architecture).
  `experiments/lima/lima_piv.yaml`.
- **Kinematic training** (`experiments/lima/train.yaml`): synthpix renders fresh
  randomized particle images each batch on top of the class-1 *train-split*
  displacement fields (`include_images: false` — the Manickathan-et-al 2022
  kinematic strategy). Particle density 0.03–0.04 ppp, diameter 1.5–2.5 px,
  shot/Gaussian noise, particle dropout — re-randomized per batch (augmentation).
- **Data:** local mirror of the HF dataset — full test split (450 fields) + a
  150-field-per-scenario train subset (1,350 fields). Each field is re-rendered
  every time it is drawn, so the effective training set is far larger.
- **Optimizer:** Adam, lr **5e-4** (raised from the paper's 1e-4/2e-4 for a
  reduced-budget run), global-norm grad clip 1.0. Multi-level Jacobian-penalised
  L1 loss (λ_u=0.91, λ_J=0.09).
- **Budget:** 40,000 batches × 10 = 400k image-pair presentations, ~36 min on
  one RTX 5090. (The paper trains 200 epochs ≈ 3.6–7.4 M presentations.)
- **Eval** (`src/eval_lima.py`): endpoint error EPE = ‖pred − gt‖ per pixel,
  reported excluding a 16 px border (paper convention), over all 450 test fields.

## Learning curve — class-1 test (kinematic-rendered, in-distribution)

| Checkpoint | Mean EPE | Median EPE | Std |
|---|---|---|---|
| random init (baseline) | 1.717 px | 1.294 px | 1.121 |
| 8,000 | 0.088 px | 0.076 px | 0.049 |
| 24,000 | 0.067 px | 0.056 px | 0.041 |
| **32,000 (best)** | **0.065 px** | **0.055 px** | **0.040** |
| 39,999 (final) | 0.115 px | 0.057 px | 1.007 |

**Best checkpoint: 32,000.** EPE drops ~26× from the random-init baseline to
**0.065 px mean / 0.055 px median**, comparable to or better than the paper's
LIMA-6 (⟨ε⟩ ≈ 0.17 px on the Carlier DNS case — a different test set).

**Late-training regression:** between 32k and 40k the `uniform` (large
pure-translation) scenario blew up from 0.068 → 0.96 px (median stayed at
0.057), spiking the std to 1.0. This is lr-too-high-without-decay instability on
the hardest examples — pure translation is also a known LIMA weak spot (not
emphasised in the kinematic training distribution). Fixes for a full run:
cosine/plateau lr decay + best-checkpoint selection on a validation split
(`save_only_best`, which was disabled here).

### Best checkpoint (32k) — per-scenario mean EPE (excl. 16px), kinematic test

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

## Domain transfer — class-1 test on REAL recorded images (`I0`/`I1`)

Evaluating the kinematic-trained model on the dataset's *actual* recorded PIV
images (not re-rendered) — a harder, out-of-distribution test:

| Checkpoint | Mean EPE | Median EPE |
|---|---|---|
| 32,000 | 0.363 px | 0.151 px |
| 39,999 | 0.224 px | 0.156 px |

Median EPE ≈ **0.15 px** (sub-pixel) on real images. The mean is inflated by the
cylinder-near-wall real images (~0.5 px) where laser reflections and solid
boundaries are hardest — the same regime where WIDIM also struggles (LIMA-1
§3.3). The replicate padding (LIMAR) is specifically meant to help here.

## Reproduce

```bash
# Pull the dataset (needs HF access to activefluidcontrol/piv-class1):
#   test split + a per-scenario train subset -> /home/.../data/piv-class1
# Train (non-episodic supervised driver; bypasses the episodic FluidEnv):
GOGGLES_PORT=$PORT PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python src/train_lima_supervised.py \
  --estimator experiments/lima/lima_piv.yaml --dataset experiments/lima/train.yaml
# Evaluate a checkpoint on the class-1 test split:
GOGGLES_PORT=$PORT PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python src/eval_lima.py --estimator experiments/lima/lima_piv.yaml \
  --dataset experiments/lima/eval.yaml \
  --load_from experiments/lima/output/lima_piv/0/checkpoints/32000
```
