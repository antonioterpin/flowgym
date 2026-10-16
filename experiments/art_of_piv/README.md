# art_of_piv: the ART of PIV paper

Each folder reproduces one result of the paper and is named after it: main-text
results at the top level, appendix results in `appendix/`. Run one with

```bash
uv run --extra cuda12 --extra wandb --extra hf python experiments/run.py --exp art_of_piv/<folder>
```

| Folder | Paper |
|---|---|
| `table_1/` | Table 1 (main results; also the points of Figure 2) |
| `appendix/A.1_learning_signal/` | Appendix A.1, Table 3 and Figure 8 (learning signal vs library size) |
| `appendix/A.2_signal_parameters/` | Appendix A.2, Figure 9 (bandit exploration) and Table 4 (regression target) |
| `appendix/A.3_agent_architecture/` | Appendix A.3, Table 5 (larger router) |
| `appendix/B_library_selection/` | Appendix B (greedy vs exact library selection; a script, not `run.py`) |
| `appendix/C_raft_finetuning/` | Appendix C (the six RAFT32 specialists of the raft32 row) |

The names follow the paper's numbering at the time of writing; they are
renamed once when the paper is final.

## Shared files

Experiments share the files below. Paths inside configs start at the
repository root (`extends: experiments/art_of_piv/models/router/fast.yaml`);
only `exp.yaml`'s `models:` list is relative to the experiment folder.

- `datasets/`: the router's training run (`train.yaml`: train split, recipe,
  labels from the library's cached per-frame errors, validation on `val.yaml`
  and test on `test.yaml`) and the split files. `train_bs64.yaml` and
  `val_bs64.yaml` use batches of 64; `val_bs20.yaml` is the whole val split
  in batches of 20.
- `models/`: one file per estimator. `router/standard.yaml` and
  `router/fast.yaml` are the two ART of PIV routers; `router/regression.yaml`
  and `router/bandit.yaml` train the standard one with other learning
  signals. `dis/`, `openpiv/` and `raft32/` hold the library members, each
  extending its family's `base.yaml` and named after its knobs
  (`dis/ps9_st2_gd4_lv3.yaml`: patch size 9, stride 2, 4 gradient-descent
  iterations, 3 levels).
- `libraries/<row>.yaml`: a fixed library, the list of model files a router
  selects from.

An experiment folder holds its `exp.yaml`, its own model files (routers that
extend a shared router and set a library, best single estimators that extend a
shared model and run in eval mode) and the `results.json` written by `run.py`.

Data come from the private Hub datasets `activefluidcontrol/piv-class1`
(images and flow fields) and `activefluidcontrol/piv-class1-caches` (error
caches), both pinned to a revision. Log in with `hf auth login` first.
