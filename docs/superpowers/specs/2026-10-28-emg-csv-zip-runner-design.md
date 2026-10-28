# Design: EMG CSV and ZIP descriptive runner

## Intent

Let an operator run the existing local descriptive EMG workflow from either one formal R1–R5 CSV or a ZIP containing such CSV recordings. This responds to the latest request for new participants while keeping source signals and outputs local.

## Goals and non-goals

- Accept one uncompressed CSV or a ZIP of CSVs with the supported raw schema.
- Validate archive names, duplicate members, encryption, symlinks, CSV count/size, schema, and R1–R5 labels before processing. Never extract using archive-provided paths.
- Keep each recording's products separate and pseudonymize ZIP folder grouping where supported.
- Preserve the existing limitations: no deployed/verified model artifact, no mechanical repetition ground truth, and no independent physiological fatigue target.
- Do not publish participant files, individual outputs, or checkpoints.

Non-goals: training or selecting a model, predicting fatigue/repetitions, merging the open PR, or treating the temporal-third benchmark as fatigue inference.

## Approaches considered

1. Keep CSV-only and require manual extraction. Safest and simplest, but contradicts the latest explicit request to run ZIPs containing CSVs.
2. Restore the already reviewed F57 archive path and route CSV/ZIP through one picker. Recommended: it has archive path checks and per-file outputs; extend tests and update wording to the current model limitations.
3. Build an entirely new archive pipeline. Rejected because it duplicates a previously tested implementation and increases regression risk.

## Design choices

- Use option 2, with this turn's request taking precedence over the earlier CSV-only instruction.
- ZIP processing remains descriptive and local. Compatible R1–R5 members are processed; R0/R0_1 and unsupported members are listed as non-processable.
- The interface and README call the operation “analysis” or “processing,” clarify that no model loads, and provide exact CSV and ZIP commands.
- Keep the established README `Límites y uso` text unchanged.
- Update the existing open PR branch rather than creating a second competing PR; leave merge decision to the user/review workflow.

## Success checks

- Tests prove CSV and ZIP inputs work end to end and that traversal/duplicate/renamed-archive cases fail safely.
- Full suite, compile check, `git diff --check`, and canonical public manifest verification pass.
- Remote PR branch contains the tested commit; remote main is not force-updated.
