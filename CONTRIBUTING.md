# Contributing

Thanks for contributing to GE Sim. This guide covers the development setup and
what we expect in a pull request.

## Development install

Clone with submodules, then install the dev extra:

```bash
git clone --recursive <repo-url> gesim
cd gesim
pip install -e ".[dev]"
```

This installs the client, `pytest`, and `ruff`. The world-model server extra
(`pip install -e ".[server]"`) and a GPU are only needed to run the `gesim_v2`
model; the test suite runs CPU-only against the `example` model.

## Running tests

```bash
pytest
```

## Linting and formatting

```bash
ruff check src tests examples
ruff format --check src tests examples
```

Run `ruff format src tests examples` (without `--check`) to apply formatting.

## Pull request expectations

- **Tests for new code.** Add or extend tests under `tests/` for any new
  behavior; keep them CPU-only so they run in CI.
- **English-only comments**, and only where the code cannot speak for itself.
- **No internal paths.** No absolute machine paths, usernames, internal
  hostnames, or references to internal services in code, comments, or docs.
- **Keep the wire protocol in sync.** The client encoder in
  `src/gesim/client/transport.py` and the server handlers in
  `src/gesim/server/app.py` implement two ends of the same binary protocol; any
  change to one must be mirrored in the other (and covered by the codec /
  end-to-end tests).
- **Lint and tests must pass** (`ruff check`, `ruff format --check`, `pytest`)
  before requesting review; CI runs all three.
