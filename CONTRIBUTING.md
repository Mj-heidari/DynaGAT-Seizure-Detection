# Contributing

Open an issue describing the problem, exact command, commit, Python/PyTorch versions,
and a minimal example. Synthetic examples are preferred when they reproduce the issue.

## Development workflow

1. Create a focused branch from the current `main`.
2. Install PyTorch and `requirements.txt` in an isolated environment.
3. Make the change and update any affected commands or documentation.
4. Run `python -m compileall -q .` and `python -u run_selftest.py`.
5. Review `git diff` and open a pull request describing the change and validation.

Use lowercase `snake_case` for Python and helper-script filenames. Keep public
entry points named `run_<task>.py`, technical guides under `docs/`, and optional
shell helpers under `scripts/`. Keep standard GitHub metadata filenames uppercase.

Changes to preprocessing, model computation, metrics, splits, and alarm selection
need focused correctness checks and an explicit explanation of their effect on
previous results. Preserve prior experiments in separate output directories.
Do not commit raw EEG, caches, weights, credentials, or generated run outputs.

Use normal Git branches and pull requests to publish changes; avoid force-pushing
shared branches. The repository owner must select a code license before the project
can provide explicit open-source reuse terms.
