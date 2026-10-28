# Project structure and data boundary

This GitHub repository is the public, reproducible software/documentation subset of the EMG project. It is not the complete private research archive and does not contain the source recordings or participant-level outputs.

## Public repository

| Path | Contents |
|---|---|
| `src/` | EMG quality checks, filtering and feature processing, descriptive CSV/ZIP/RAR runners, dashboards, and separately scoped model-analysis code. |
| `tests/` | Software regression tests. Passing tests do not validate physiological fatigue. |
| `reports/` | Public summaries, methods/status notes, and permitted aggregate outputs. |
| `bibliografia/` | Public BibTeX references used by the project. |
| `datasets/README.md` | Instructions for authorized local data placement; no participant recordings are tracked. |
| `presentation/` | Ten-minute Spanish/English scripts and their local generator. |
| `manifest/SHA256SUMS` | SHA-256 checksums for the tracked package, excluding the manifest itself. |

Use the root README for installation and the current runner commands. The RAR runner maps `set01`–`set05` to `R1`–`R5` by filenames and reports activity candidates. It does not verify mechanical repetitions or predict validated quadriceps fatigue.

## Private local delivery

The full audit bundle is assembled separately as `EMG_Integrador_Biomecanica/` with `emg integrador biomecanica.zip` beside it. It includes the original `raw.rar`, the original 15-participant ZIP, participant-level dashboards/CSVs, the complete editable report, and historical fold metrics. These files are excluded from GitHub because source ZIP paths identify participants and the dashboards contain signal data.

The currently available editable report is F56 (2026-10-21). The F60 RAR run (2026-10-28) is stored separately and has not been integrated into that report. Its outputs are descriptive, and it did not load a model or issue repetition/fatigue predictions.

## Privacy and scientific scope

Keep the private bundle on access-controlled storage. Do not commit its source archives, individual CSVs, per-recording dashboards, participant-linked metrics, or local receipts to a public branch. The local spectral threshold screen is exploratory; a threshold state is not a clinical or independently validated fatigue label.
