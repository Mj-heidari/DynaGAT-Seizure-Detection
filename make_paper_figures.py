"""Compatibility entry point. Prefer ``python run_figures.py``."""
from run_figures import main

if __name__ == "__main__":
    raise SystemExit(main())
