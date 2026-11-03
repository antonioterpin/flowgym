# real_labels: train ART of PIV on labelled real image pairs

Train an ART of PIV router on real recordings, where each image pair is
labelled with the library member (e.g. the DIS config) that works best on
it. No ground-truth flow is needed.

## Dataset format

One folder holding the frames and a `pairs.csv`, as in `example/`:

```text
my_dataset/
├── pairs.csv
└── frames/          # any layout; paths in pairs.csv are relative to it
    ├── pair0_1.png
    ├── pair0_2.png
    └── ...
```

```csv
frame1,frame2,label,split
frames/pair0_1.png,frames/pair0_2.png,0,train
frames/pair1_1.png,frames/pair1_2.png,1,val
```

- `frame1`, `frame2`: the two frames of a pair, as grayscale image files
  (8-bit PNG or TIFF). Every frame has the same size, at full
  resolution and without preprocessing.
- `label`: the index of the best member, counting from 0 in the order of
  `library.yaml`. Every pair needs a label.
- `split` (optional): `train` or `val`. Split by recording, not by frame,
  because consecutive frames of one recording are nearly identical.
  Without this column the last 20% of rows go to `val`. Pairs from a
  separate recording session can be marked `test`: they are never trained
  or selected on, and
  [appendix F.3](../art_of_piv/appendix/F.3_selection_agent_training/README.md) scores
  the router on them with and without photometric augmentation.

## Train

1. List the members in `library.yaml`, in the order the labels use.
2. Build the training data:

   ```bash
   uv run python scripts/build_label_dataset.py \
       --labels my_dataset/pairs.csv \
       --library experiments/real_labels/library.yaml \
       --out data/real_labels \
       --preprocess <deployed estimator yaml>
   ```

   `--preprocess` applies the preprocessing of the estimator config that
   will be deployed (its `preprocess:` list, whatever it currently is) to
   every frame, so the router trains on the frames it will see deployed.
   The photometric augmentation then acts on those preprocessed frames.
   Leave the flag out only if the frames are already preprocessed that way.

   This writes each pair as a `.mat` file with zero flow into
   `data/real_labels/{train,val}`, and writes one cache per member into
   `data/real_labels/cache`. A cache's `epe` column is not an endpoint
   error: it is 0 for the labelled member and 1 for the others, so the
   router learns the label. The script prints the frame size.
3. Set `image_shape` in `datasets/train.yaml` and `datasets/val.yaml` to that
   size (after `--preprocess`, e.g. its crop). To give the router a smaller
   copy of the frames than the DIS configs get, add `router_preprocess` to
   `models/router.yaml`, and the same block to the deployed config.
   `datasets/train.yaml` randomises the lighting of every training pair
   (`photometric_augmentation`: a gain and offset shared by both frames,
   plus a small per-frame gain); keep it on, the router needs it to transfer
   across recordings.
4. Run:

   ```bash
   uv run --extra cuda12 --extra wandb python experiments/run.py --exp real_labels
   ```

Validation covers every val pair exactly once (`num_batches: all`).
Training and validation run no sub-estimator, because every frame hits
the label cache. On validation, `mean_error` is the router's error rate on
the labels and `choices/accuracy` is its accuracy. The run keeps the
checkpoint with the lowest validation error rate.
