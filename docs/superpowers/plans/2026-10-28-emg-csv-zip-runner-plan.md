# Implementation plan: CSV and ZIP runner

1. Replace the CSV-only rejection test with acceptance tests for `--zip`, ZIP discovery, pseudonymized rows, and unsafe archive rejection. Run the targeted tests and confirm expected failure before restoring ZIP support.
2. Restore the previously reviewed `src/analizar_zip_emg.py` and its archive validation tests. Route the existing single-click CLI to accept exactly one of `--csv` or `--zip`; make the GUI picker accept both formats.
3. Run synthetic CSV and ZIP end-to-end tests; assert the result manifests keep source names private and `model_loaded=false`, with fatigue/repetition predictions null. Run the full suite and compile check.
4. Update the launcher, README usage (without changing `Límites y uso`), F58/F59 status, presentation scripts, and the existing Canva demo-slide wording so the input contract matches. State that no predictive model is available for deployment.
5. Rebuild `manifest/SHA256SUMS` from staged Git blobs, verify all hashes and privacy exclusions, review the diff, commit with the no-reply author identity, push the existing PR branch, and verify the remote PR head.

## Commands and pass conditions

- `python -m unittest tests.test_analizar_csv_emg tests.test_analizar_zip_emg -v`: CSV/ZIP behavior and safety tests pass.
- `python -m unittest discover -s tests -q`: full suite has zero failures.
- `python -m compileall -q src tests`: exits 0.
- `git diff --check`: exits 0.
- Manifest verifier compares expected SHA-256 with Git blobs for every listed path, with no private data paths included.
- `git ls-remote origin refs/heads/f57-oneclick-zip-analysis` equals the pushed commit; GitHub PR #3 head resolves to that same commit.

## Review focus

- ZIP path traversal, absolute paths, Windows paths, duplicate names, symlink and encrypted members.
- Single CSV inside a ZIP, multiple CSVs, preliminary R0/R0_1, malformed schema, corrupt archive, and ZIP renamed `.csv`.
- No participant filename in public logs or output tables; model-loading and prediction fields remain explicitly false/null.
- README limitations section remains byte-identical to the current public branch.
