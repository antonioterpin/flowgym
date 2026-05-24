# Test suite layout

The suite is organized **functionality-first** and split by test kind so it is
clear what each file covers and how it maps back to `src/flowgym/`.

```
tests/
├── conftest.py          # collection hooks + shared-fixture registration
├── mocks/               # shared fixtures, grouped by domain
│   ├── flow_data.py     #   synthetic .mat / .npy flow data
│   ├── training.py      #   mock estimators / samplers / envs / states
│   └── caching.py       #   cache dirs + synthpix batches
├── unit/                # fast, isolated tests; mirror src/flowgym/
│   ├── common/          #   common/ (filters, median, preprocess, base, …)
│   ├── flow/            #   flow/ (dis, postprocess, consensus, …)
│   ├── density/         #   density/
│   ├── nn/              #   nn/ (blocks, cnn, mlp, raft, registry)
│   └── training/        #   training/ (builders, replay, caching, …)
└── integration/         # cross-module / end-to-end behavior
    ├── training/        #   full training + checkpointing pipelines
    └── caching/         #   eval/estimator caching end-to-end
```

## Conventions

- **Unit vs integration.** A test is *unit* if it exercises a single module
  (mocking its collaborators) and *integration* if it wires several modules
  together or drives an end-to-end pipeline. Put it under the matching tree.
- **Mirror the source.** Unit tests live under the `unit/<subpackage>/` path
  that matches the module under test in `src/flowgym/`. Add new files there so
  coverage stays easy to locate.
- **Docstrings.** Every test file has a module docstring naming the source
  area it covers and whether it is unit or integration. Every test function
  has a one-line docstring describing the *behavior* being validated, not the
  implementation detail.
- **Shared fixtures** belong in `tests/mocks/` (registered via
  `pytest_plugins` in `conftest.py`), not in individual test files or a
  growing `conftest.py`.

## Markers

- `run_explicitly` — only collected with `-m run_explicitly`.
- `slow` — deselect with `-m "not slow"`.

Tests behind the `other_methods` extra (OpenCV / OpenPIV / PyTorch) are
skipped automatically when those optional dependencies are absent (see
`pytest_ignore_collect` in `conftest.py`).
