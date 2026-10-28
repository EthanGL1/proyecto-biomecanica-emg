# EMG Project Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Deliver one traceable local EMG project package with clearly labeled original archives, current documents/code/results, folder READMEs, and an auditable ZIP while keeping participant-level data out of public GitHub.

**Architecture:** Build a curated private delivery folder from verified source artifacts; leave the working project, historical runs, and source archives untouched. Update the existing public Git branch only with reproducible code and safe documentation, then verify local/remote commit state and inspect the ZIP manifest.

**Tech Stack:** PowerShell, Git, Python 3.12 virtual environment, ZIP/ZIP64, SHA-256 manifests, GitHub connector.

**Spec:** User request in the 2026-10-28 project continuation: organize article and full project; label exact original RAR and 15-participant ZIP; write folder guides; preserve traceability; create `emg integrador biomecanica.zip`; update GitHub.

## Global Constraints

- Preserve the source `raw.rar` and `datasets/OneDrive_1_10-10-2026.zip` byte-for-byte; record SHA-256 before and after copying.
- Do not put original signals, participant CSVs, individual results, or dashboards containing signals on public GitHub.
- Use `DOCUMENTOS PARA ENTREGAR/Grupo_4_Informe_Final.docx` and `.pdf` as the latest existing report pair (F56, 2026-10-21); do not imply F60 results have been integrated into the article.
- Use `.worktrees/F57_oneclick_analysis` as the current reproducible code source and the local 2026-10-28 RAR results directory as private results.
- Preserve historical project folders and Git history; identify superseded versions rather than deleting evidence.

## Review Focus

- Source copy integrity: hashes and byte sizes of both source archives before/after copy.
- Public/private boundary: inspect staged paths for raw data, participant-level records, embedded-signal HTML, and credentials before push.
- Version alignment: show current report date/version separately from F57 code and F60 run outputs.
- ZIP validity: list entries, verify required READMEs/manifests, and ensure output ZIP does not include itself or temporary environments.
- Git publication: verify commit ID and remote branch/default branch state after update.

---

### Task 1: Assemble private, organized audit package

**Files:** Create the curated `EMG_Integrador_Biomecanica/` structure, per-folder `README.md`, source hash manifest, and package index; copy verified report, code, references/management records, presentation assets, original archives, and private run outputs.

- [ ] Create folders for start-here, originals, article, code, results, bibliography/management, presentation, and history index.
- [ ] Copy `raw.rar`, the original 15-participant ZIP, latest DOCX/PDF, F57 sources excluding virtualenv/cache, and F60 run outputs; preserve originals.
- [ ] Write a README in each folder describing origin, contents, privacy, and how to use it.
- [ ] Record source versions, SHA-256, size, counts, and known scientific limits in manifests.

### Task 2: Produce and verify requested ZIP

**Files:** Create `emg integrador biomecanica.zip` beside the curated folder and a ZIP verification receipt.

- [ ] Create ZIP64 archive from the curated folder with no `.venv`, caches, or nested previous ZIPs.
- [ ] Reopen ZIP, enumerate contents, verify required files and source hashes, and confirm raw data is labeled private.

### Task 3: Update the public GitHub repository safely

**Files:** Update public F57 documentation/organization metadata only; retain the existing user-authored scientific limitations in `README.md`.

- [ ] Review current remote branches and outstanding PRs; select the existing F57 worktree.
- [ ] Add safe folder navigation and exact reproducible commands; do not copy private archives/results.
- [ ] Run existing project verification only if needed for the changed public files, inspect staged paths, commit, and push.
- [ ] Verify remote commit and ensure no private data entered the public repository.

### Task 4: Final cross-check

- [ ] Compare local manifest and ZIP contents against source inventory.
- [ ] Record component statuses and residual limits; distinguish report F56 from code F57/run F60.
