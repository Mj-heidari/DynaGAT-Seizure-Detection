# Repository cleanup: migration notes

| Previous path | Current path | Action |
| --- | --- | --- |
| `docs/DESIGN_NOTES.md` | `docs/design_notes.md` | Update bookmarks |
| `make_paper_figures.py` | `run_figures.py` | Old command still forwards to the new entry point |
| `run_all.ps1` | `scripts/run_pipeline.ps1` | Use the new path; baseline and ablations are opt-in switches |
| `setup_vscode.ps1` | `scripts/setup_vscode.ps1` | Run from the new location |
| `push_to_github.ps1` | Standard Git workflow in `CONTRIBUTING.md` | Publishing helper removed |

Set `CHBMIT_BIDS_ROOT` to your data location. The default is now
`data/BIDS_CHB-MIT` inside the repository rather than a personal Windows path.
The PowerShell runner honors that setting. Existing model imports, scientific
hyperparameters, output filenames, cache version, and experiment signature
calculation are unchanged by this organizational cleanup.
