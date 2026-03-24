# Project Rules

## Package Management

- ALWAYS use `uv` for installing, removing, or managing Python packages.
- NEVER use `pip`, `python -m pip`, or any other package manager.
- Before installing anything, check if it will conflict with existing dependencies (especially torch/CUDA versions).
- Example: `uv pip install onnxruntime-gpu`
- Likewise, use uv to run scripts!
