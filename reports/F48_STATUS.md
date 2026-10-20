# EMG study status — F48

**Cutoff:** 20 October 2026. This is an intermediate, data-safe status report. It does not replace the final LaTeX manuscript or claim project completion.

## Independent study branches

| Component | Verified evidence | Status and limit |
|---|---|---|
| Local quadriceps recordings | 140 original CSVs, 4,256,257 samples, 15 recording-derived groups; 138 formal R1–R5 files and two R0/R0_1 QC-only files. Stored fields are `Relative_Time_s`, `ADC_Value`, and `Voltage_mV`. | Complete for inventory and descriptive signal analysis. Stored modality (RAW/rectified/envelope), analog acquisition chain, physical voltage calibration, load, force, MVC, and mechanical repetition markers are unknown. |
| Exported ADC-to-voltage relation | Across every sample, the millivolt column equals `round(ADC_Value × 3300/4095, 2)`; 1,915,057 ADC values are negative. | The arithmetic export rule is verified; the underlying reference voltage, converter coding, bipolar offset, analog gain, and electrode-level calibration are not. The millivolt values are not asserted as calibrated physical measurements. |
| Local paired R1–R5 endpoint | 27 matched participant–leg pairs. Median changes: RMS +31.27% (95% interval 5.11 to 40.04), MDF +1.45 Hz (−4.49 to 2.22), MNF −2.06 Hz (−6.56 to 1.00). | Descriptive only. The MDF/MNF intervals include zero; no cohort-level classical spectral fatigue signature is demonstrated. No local fatigue classifier or physiological fatigue label exists. |
| Local events and activity mask | The F39 sensitivity output contains 7,846 windows and 2,102 candidate activations. Duration and occupancy settings retain weak and late activity for RMS where valid; unsupported spectra are marked non-evaluable. | Candidate activity is not a verified repetition. The visual plausibility audit did not have synchronized mechanical ground truth, so event recall, precision, and false-event rates are unavailable. |
| Signal conditioning and viewer | Candidate Butterworth HP20/HP40, Butterworth BP20–400 and 101-tap Hamming FIR were compared. Original and processed signals, FFT/PSD/STFT, RMS, MDF/MNF, masks and candidate events are shown in the existing F46 viewer and figures. | The filter response and changes are quantified, but no clean reference or verified acquisition chain permits a denoising winner or physical SNR claim. Notch was not applied because the predeclared line-prominence screen did not trigger. |
| Separate shoulder Borg model | 34 participants; 1,309 grouped out-of-fold windows; nested participant-wise LOSO macro-F1 0.6176 pooled and participant mean 0.5787 (SD 0.1344). The binary target is Borg RPE ≥17. | The score estimates perceived-exertion classification in that shoulder protocol. The final XGBoost checkpoint is not operationally verified: loading and prediction were blocked by the authorized local runtime. No fatigue probability from that checkpoint is claimed. |
| External compatibility | SWIFTIES has five channels, while the frozen Borg input contract requires six named channels/42 descriptors. | Incompatible; no external metric is reported and this is not a negative validation result. |
| Independent Cerqueira experiment | Official protocol, label archive and three classes (0/1/2) were inspected; labels cover 13 subjects and 156 trials. | Kept separate as upper-limb self-perceived fatigue. Raw EMG recovery, checksum/structure completion, irregular-time handling, task boundaries, and trial-level label synchronization remain pending. No training or evaluation result is claimed. |
| LaTeX manuscript | F48 was reorganized to match the required top-level headings and mandatory subsections. Structural validation reports 44 references, all cited; no missing figures or tables; all figure/table labels referenced; abstract 240 words; seven keywords; no submission checklist. | Source is implemented and structurally verified. PDF compilation is blocked because the required Tectonic bundle index is unavailable from the configured compiler service and no local TeX engine was found. Page count and visual PDF inspection remain pending. |

## Public-code verification

The public processing code was tested on a clean clone of the remote `main` baseline `19a024f6a8ed18da6b829f88725fcf267e417a93`. The baseline repository history inspected during this update uses the account's GitHub no-reply address. Before changes, the checkout had 47 passing `unittest` cases, `compileall` passed, and the existing help command loaded. The existing CLI was then run end-to-end in deterministic synthetic-demo mode with its HTML dashboard and data exports. It produced `segments.csv`, `mask.csv`, `features.csv`, `fft.csv`, `psd.csv`, `stft.csv`, `activity_candidates.csv`, `summary.json`, and `dashboard.html`, clearly labelled synthetic. This verifies software output and the existing viewer/export path, not physiological validity or cohort reproduction.

The repo's previous restricted F44 audit also records one real-recording smoke run outside the public checkout. No raw data or that recording's identity/hash is copied into this status file.

## Release limits

- The public processing CLI exports signal features, spectra, masks, STFT, candidate activity and a descriptive Gantt/viewer; it does not produce fatigue predictions or Borg scores.
- No Borg checkpoint, subject-level result, local signal, participant CSV, real-signal HTML, private trace, or per-window/per-event table is included.
- The historical F31 DOCX/PDF and reports remain labeled as historical context; they do not represent the F48 article.
- The LaTeX PDF, loadable Borg model, synchronized Cerqueira corpus/model, and compatible external fatigue validation are not complete.
- A public commit should be described as an **intermediate safe code/status update**, not a final scientific release, until these dependencies are resolved or explicitly closed as unavailable.
