"""Shared test mocks and fixtures.

Fixtures here are registered globally via ``pytest_plugins`` in
``tests/conftest.py`` and grouped by domain:

- :mod:`tests.mocks.flow_data` — synthetic ``.mat`` / ``.npy`` flow data.
- :mod:`tests.mocks.training` — mock estimators, samplers, environments,
  and trainable states for training tests.
- :mod:`tests.mocks.caching` — cache directories and synthpix batches for
  the caching tests.
"""
