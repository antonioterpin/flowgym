# Appendix C: RAFT32 fine-tuning

Appendix C fine-tunes the pretrained RAFT32-PIV once per flow class of
piv-class1 (cylinder, backstep, DNS_turbulence, SQG, jhtdb, uniform). The
six specialists form the library of `table_1`'s raft32 row, which loads the
paper's checkpoints from W&B (`experiments/art_of_piv/models/raft32/`). This
folder retrains them with the same recipe and scores them next to the
paper's checkpoints.

## Recipe

From the W&B runs `artofpiv/RAFT32_finetuning` (2026-05-17/18), which
differ only in their data:

- start from the pretrained RAFT32-PIV
  (`experiments/art_of_piv/models/raft32/pretrained.yaml`);
- Adam with global-norm clipping at 0.5; learning rate 0 for 100 steps,
  then a linear one-cycle over 32k steps peaking at 1e-5 (5e-7 → 1e-5 by
  step 2,100 → 5e-7 by step 27,300), then 5e-7;
- 50k steps of batch 25 from the class's train split, shuffled (one random
  32 px patch per image);
- every 5k steps, validate on 4 batches of 25 shuffled val images of the
  class (the same 100 each time) and keep the best checkpoint, which is
  then scored on the class's test split.

## Files

- `exp.yaml`: the six fine-tuning runs, then the paper's six specialists
  in eval mode; seed 0.
- `models/finetune.yaml`: the recipe. `models/<class>.yaml` extend it and
  set the class's dataset.
- `models/paper/<class>.yaml`: the paper's specialist of the class (the
  checkpoint `table_1` uses), scored on the same test split.
- `datasets/finetune.yaml`: the shared training and validation settings;
  `datasets/<class>.yaml` extend it and set the class's train, val and test
  folders.
- `datasets/test/base.yaml`: a test split in batches of 3;
  `datasets/test/<class>.yaml` extend it and set the class's folders and
  number of batches (every test frame).
- `results.json`: written by `run.py`; per model, the test AEPE.

## Run

```bash
uv run --extra cuda12 --extra wandb --extra hf python experiments/run.py --exp art_of_piv/appendix/C_raft_finetuning
```

A class takes about 1 h 25 min on an RTX 4090. To split the classes over
GPUs, run one launcher per GPU with `--models` (e.g. `--models
models/cylinder.yaml models/backstep.yaml`), each with its own
`CUDA_VISIBLE_DEVICES` and `GOGGLES_SOCKET`; the launcher that finishes
last writes `results.json` for all of them.

## Results

RTX 4090 (GPUs 2 and 3), October 2026.

Validation AEPE every 5k steps, reproduced / W&B:

| class | 5k | 10k | 20k | 30k | 40k | 50k |
|---|---|---|---|---|---|---|
| cylinder | 0.00507 / 0.00503 | 0.00484 / 0.00483 | 0.00471 / 0.00471 | 0.00467 / 0.00467 | 0.00466 / 0.00466 | 0.00466 / 0.00466 |
| backstep | 0.00341 / 0.00331 | 0.00326 / 0.00321 | 0.00312 / 0.00312 | 0.00308 / 0.00307 | 0.00307 / 0.00307 | 0.00307 / 0.00307 |
| DNS_turbulence | 0.02146 / 0.02144 | 0.02115 / 0.02113 | 0.02074 / 0.02074 | 0.02058 / 0.02058 | 0.02055 / 0.02054 | 0.02053 / 0.02053 |
| SQG | 0.01835 / 0.01835 | 0.01797 / 0.01798 | 0.01775 / 0.01776 | 0.01768 / 0.01768 | 0.01766 / 0.01767 | 0.01765 / 0.01765 |
| jhtdb | 0.01174 / 0.01174 | 0.01158 / 0.01157 | 0.01134 / 0.01134 | 0.01129 / 0.01129 | 0.01129 / 0.01129 | 0.01128 / 0.01128 |
| uniform | 0.00302 / 0.00302 | 0.00305 / 0.00293 | 0.00298 / 0.00290 | 0.00286 / 0.00285 | 0.00289 / 0.00287 | 0.00297 / 0.00295 |

Test AEPE of the best checkpoint, reproduced against the paper's
specialist:

| class | best step | reproduced | paper | diff |
|---|---|---|---|---|
| cylinder | 50k | 0.004428 | 0.004429 | −0.0 % |
| backstep | 45k | 0.003343 | 0.003337 | +0.2 % |
| DNS_turbulence | 45k | 0.021533 | 0.021529 | +0.0 % |
| SQG | 50k | 0.017798 | 0.017806 | −0.0 % |
| jhtdb | 50k | 0.010989 | 0.010992 | −0.0 % |
| uniform | 30k | 0.002786 | 0.002751 | +1.3 % |

From 15k steps on, the curves stay within 0.00001 px of W&B for every
class but uniform, whose curve is noisy in both runs (it peaks at 15k and
drifts up after 30k). On test, five specialists match the paper's within 0.2 %; uniform
(24 test frames) is 1.3 % worse.
