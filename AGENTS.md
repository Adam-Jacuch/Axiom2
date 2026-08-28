# Axiom contributor guide

## Project overview

Axiom is a strictly typed, named-axis eDSL built on JAX. Its public API is
exported from `src/axiom/__init__.py`; core tensor and axis behavior lives in
`src/axiom/core.py`, compilation/model wrappers in `src/axiom/compiler.py`,
and neural-network operations in `src/axiom/nn.py`.

## Repository layout

- `src/axiom/`: library implementation.
- `tests/`: pytest suite for core behavior, MCP utilities, and sharding.
- `examples/`: executable examples and tutorial material.
- `pyproject.toml`: package metadata and dependencies.
- `uv.lock`: locked development environment; update it when dependencies change.

## Development workflow

- Use Python 3.10+ and run project commands through `uv`.
- Install the development environment with `uv sync --group dev`.
- Run the standard suite with `uv run pytest`.
- Run a focused test while iterating with `uv run pytest tests/test_full.py -q`.
- Run the sharding test separately with `uv run pytest tests/test_sharding.py -q`.
  It sets `XLA_FLAGS` at module import to emulate four CPU devices, so do not
  import JAX before that environment setting when modifying this test.

## Implementation conventions

- Preserve the functional JAX model: avoid object-oriented layer wrappers and
  mutable parameter ownership outside `compiler_state`.
- Tensors carry named `Axis` topology. Prefer axis-targeted operations such as
  `x.d.proj()` over positional-shape logic, and verify topology after operations
  that rename, merge, split, contract, gather, or slice axes.
- Parameter creation must remain deterministic during a JAX trace. Use the
  existing `init` and `Tie` mechanisms instead of allocating ad hoc arrays.
- Keep public imports in `src/axiom/__init__.py` aligned with intended public
  API changes.
- Keep MCP tools defensive: return useful diagnostics for invalid user-provided
  code instead of propagating unhandled errors.

## Tests and changes

- Add or update focused pytest coverage for changed behavior, including both
  values and tensor topology where relevant.
- Avoid committing generated or local artifacts such as `.venv/`,
  `__pycache__/`, `.pytest_cache/`, IDE metadata, or `*.egg-info/`.
- Keep examples small and runnable; use them to demonstrate user-facing named
  axis idioms.
