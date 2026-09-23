# Coding Guidelines

This is a research-oriented deep learning project.

Prefer simple and explicit implementations over defensive programming.

- Make the smallest change necessary for the requested experiment.
- Trust internal tensor shapes, dtypes, devices, and configuration contracts.
- Do not add redundant shape/dtype/device checks unless they protect a non-obvious mathematical invariant.
- Do not add broad `try/except` blocks around training, evaluation, or model forward passes.
- Do not silently skip failed batches or non-finite losses.
- Do not automatically sanitize NaN/Inf values unless mathematically justified.
- Do not add CPU/CUDA fallbacks or compatibility paths unless explicitly requested.
- Do not add speculative robustness, abstractions, or support for hypothetical future use cases.
- Let PyTorch errors propagate when they already clearly indicate a programming error.
- Follow the existing style of the surrounding code.

Optimize for ease of experimentation and inspection, not library-grade robustness.

Do not make unrelated robustness improvements while completing a scoped task.