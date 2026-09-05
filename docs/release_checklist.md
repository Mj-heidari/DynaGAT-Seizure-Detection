# Paper-release checklist

## Repository metadata

- Confirm the software author metadata in `CITATION.cff`; add contributors as appropriate.
- Add the final paper title, complete authors, year, venue, and DOI when available.
- Select and add a code license with the relevant rights holders. Keep dataset terms separate.
- Set a concise GitHub description and relevant topics such as `eeg`, `seizure-detection`,
  `graph-attention-networks`, `pytorch`, and `chb-mit`.

## Scientific record

- Confirm the code revision matches the experiments reported in the paper.
- Check completed folds, development exclusions, ablation coverage, and patient identities.
- Preserve actual commands, configuration, dataset/conversion provenance, seeds,
  dependency versions, GPU information, and output checksums.
- Resolve or explicitly account for the resume and cache-path limitations in
  [reproducibility.md](reproducibility.md).
- Run the synthetic self-test and the full data/GPU health check in the experiment environment.
- Verify that all manuscript tables and figures come from the archived run.

## Freeze and share

- Review tracked files and Git history for credentials, raw EEG, local files, and artifacts
  that should not be redistributed. `.gitignore` does not remove existing history.
- Create a version tag and GitHub release for the verified experimental revision.
- Archive that release in a persistent research archive if a software DOI is needed.
- Add the real version and DOI to `CITATION.cff`; do not use placeholder identifiers.
- Include the repository URL and the exact release/commit in the manuscript.

Repository URL: https://github.com/Mj-heidari/DynaGAT-Seizure-Detection

The cleanup itself is not a new experimental validation or a finalized paper release.
