# F.3_selection_agent_training: does photometric augmentation help?

Trains the [real_labels](../../../real_labels/README.md) router twice per seed,
with and without photometric augmentation, and scores the best-on-val
checkpoint of each on a separate recording (the test split). The two use
the same data, library, router and steps; only the augmentation differs.

## Data and models

The recordings behind this experiment are published, training-ready and with
their label caches, as `activefluidcontrol/art-of-piv-water-channel`; the
trained routers (and these ten checkpoints) are in
`activefluidcontrol/art-of-piv-models` under `water_channel/`. The caches match
the ids of `libraries/water_channel.yaml`, so a download can be trained on
directly. The region-selection router of the same appendix uses
`libraries/water_channel_roi.yaml` and the `roi/` folder of the dataset, with
the recipe of [real_labels](../../../real_labels/README.md).

## Run

1. Record a separate session (another day, or a few hours later, so e.g.
   illumination and seeding density differ) and add its pairs to
   `pairs.csv` with `split` set to `test`. Train and val pairs stay as they
   are: the checkpoint is still picked on val, never on test.
2. Label every pair 0 for DIS A (+150 rpm) or 1 for DIS B (-150 rpm), the
   order of `experiments/art_of_piv/libraries/water_channel.yaml`, and build
   with the deployed router's preprocessing:

   ```bash
   uv run python scripts/build_label_dataset.py \
       --labels my_dataset/pairs.csv \
       --library experiments/art_of_piv/libraries/water_channel.yaml \
       --out data/real_labels \
       --preprocess <FlowGames flow_estimators/art_of_piv_cylinder_rpm_photo.yaml>
   ```

   Then set `image_shape` in `experiments/real_labels/datasets/train.yaml`
   and `val.yaml` to the size it prints (this experiment reads both). It
   also prints the number of test pairs.
3. Run:

   ```bash
   uv run --extra cuda12 --extra wandb python experiments/run.py --exp art_of_piv/appendix/F.3_selection_agent_training
   ```

## Into the paper

The paper reads `results.json` from here (manifest namespace
`channelAugmentation` in the paper's `scripts/results_manifest.toml`,
macros `\channelAugmentationWithAccuracyPct` and the like). After a run:
commit `results.json` and this README, merge into `dev`, move the paper's
`code/flowgym-private` submodule to that commit, and run `make numbers`
there.

## Results

`mean_epe` is the router's error rate on the test split (lower is better;
1 − accuracy), mean ± std over the 5 seeds. The table fills in from
`results.json` after a run.

<!-- BEGIN results.json -->
Not run yet.
<!-- END results.json -->
