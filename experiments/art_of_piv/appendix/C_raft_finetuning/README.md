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

Set up the environment as in [the ART of PIV README](../../README.md#run), then:

```bash
uv run python experiments/run.py --exp art_of_piv/appendix/C_raft_finetuning
```

A class takes about 1 h 25 min on an RTX 4090. To split the classes over
GPUs, run one launcher per GPU with `--models` (e.g. `--models
models/cylinder.yaml models/backstep.yaml`), each with its own
`CUDA_VISIBLE_DEVICES` and `GOGGLES_SOCKET`; the launcher that finishes
last writes `results.json` for all of them.

## Results

RTX 4090 (GPUs 2 and 3), October 2026. Test AEPE of each class's best
checkpoint (`models/<class>.yaml`) and of the paper's specialist
(`models/paper/<class>.yaml`), generated from `results.json`:

<!-- BEGIN results.json -->
| model | mean_epe |
|---|---|
| `models/cylinder.yaml` | 0.00443 |
| `models/backstep.yaml` | 0.00334 |
| `models/DNS_turbulence.yaml` | 0.02153 |
| `models/SQG.yaml` | 0.01780 |
| `models/jhtdb.yaml` | 0.01099 |
| `models/uniform.yaml` | 0.00279 |
| `models/paper/cylinder.yaml` | 0.00443 |
| `models/paper/backstep.yaml` | 0.00334 |
| `models/paper/DNS_turbulence.yaml` | 0.02153 |
| `models/paper/SQG.yaml` | 0.01781 |
| `models/paper/jhtdb.yaml` | 0.01099 |
| `models/paper/uniform.yaml` | 0.00275 |
<!-- END results.json -->
