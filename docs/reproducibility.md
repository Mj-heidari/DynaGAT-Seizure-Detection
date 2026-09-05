# Setup and reproducibility

## Environment

Use Python 3.10+ in an isolated environment and install PyTorch separately from
`requirements.txt`. CPU is sufficient for synthetic checks; the full health check
expects CUDA. There is no dependency on `torch-geometric`. Requirements specify
minimum package versions, not a frozen environment. Save the environment used for
the manuscript with `python -m pip freeze > paper_results/requirements-lock.txt`
after the output directory exists.

Run commands from the repository root. Environment variables must be set before
starting Python; a `.env` file is not loaded automatically.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHBMIT_BIDS_ROOT` | `data/BIDS_CHB-MIT` under the repository | Raw BIDS data |
| `DYNAGAT_CACHE_DIR` | `data_cache_v4` under the repository | Preprocessed caches |
| `DYNAGAT_RESULTS_DIR` | `results` under the repository | Training outputs |

Prefer absolute paths. Use a separate results directory for a distinct experiment.

## Data contract

The loader expects BIDS subject directories (`sub-*`) containing EDF recordings
and corresponding events TSV files. For `*_eeg.edf`, the annotation filename is
`*_events.tsv`; otherwise it uses the EDF stem plus `_events.tsv`.
Events use `onset` and `duration` in seconds. If descriptive columns exist,
seizure rows must have a seizure label recognized by `dataset/io_edf.py`.
Missing event files currently return an empty seizure list, so verify annotation
coverage before treating a recording as seizure-free.

The BIDS conversion is not included. Document the source dataset version,
conversion procedure, included recordings, exclusions, and annotation coverage
alongside the paper. Validate the handling of linked subject identities in your
conversion and folds; nominal file or directory counts alone do not establish
patient independence.

## Execution

```bash
python -u run_selftest.py
python -u -m dataset.preprocess --max-subjects 2
python -u -m dataset.preprocess
python -u run_healthcheck.py
python -u run_lopo.py
python -u -m baselines.classical
python -u run_lopo.py --all-ablations
python -u run_export.py
python -u run_figures.py
```

The two-subject preprocessing command checks data compatibility; LOPO training
requires at least eight subject caches. `run_lopo.py --folds 1` limits training
to the development fold. Runtime depends on hardware and cohort size.

On Windows, the main workflow is also available as:

```powershell
.\scripts\run_pipeline.ps1
.\scripts\run_pipeline.ps1 -IncludeBaseline -IncludeAblations
.\scripts\setup_vscode.ps1
```

Set `CHBMIT_BIDS_ROOT` before launching VS Code so debug sessions inherit it.
Select the Python interpreter for your environment in VS Code.

## Resume limitations

Completed folds are skipped when their summary signature matches
`config.experiment_signature()`. That signature currently includes only selected
configuration values: it does not hash source code or every runtime override.
Do not reuse an output directory after changing code, CLI epochs/batch size, or
settings omitted from the signature. Set `DYNAGAT_RESULTS_DIR` to a new directory
and preserve the old artifacts separately. Record the actual command and commit.

Use `DYNAGAT_CACHE_DIR` for an alternate cache location consistently across
preprocessing and training. The current `--cache-dir` option in the LOPO driver
is used for subject discovery but is not forwarded to the fold trainer.

## Reporting the experiment

Report the development-fold exclusion, the actual patient split, validation-only
operating-point selection, window-end timestamps, event matching tolerances,
false-alarm denominator, and recording warm-up rules from the exact code revision
used. Preserve `run_manifest.json`, per-fold outputs, and the full environment.
The validation FA/h cap is a selection constraint, not a guarantee on held-out data.

The figure generator skips missing inputs. Successful figure generation therefore
does not certify that every fold or ablation is complete. Check coverage against
the manuscript before release. See [release_checklist.md](release_checklist.md).
