## About

## Install
- depending on your plattform / accelerator, uncomment the appropriate line in `pyproject.toml` under `[tool.uv.sources]` (uv does not allow conditional sources with overlapping markers)
- run e.g. `uv sync --extra cpu --dev` (replace `cpu` with `xpu` or `cuda` depending on your accelerator and source index)
