# Group 4 — 10-Minute Scientific Presentation Script

**Language:** English
**Target duration:** planned at 10:00; 1,238 spoken words (approximately 124 words per minute). Assigned slide timings sum to 10:00; rehearse to confirm delivery pace.
**Deck:** existing Canva presentation “Defensa sEMG cuádriceps” (10 slides; existing layout retained).

Slide timings sum to 10:00. Each spoken section is word-for-word; visual cues are not spoken.

## Slide 1 — Quadriceps sEMG during unilateral knee extension

**Speaking time:** 00:30
**Visual reference:** Title and study question

**Complete speaking script:**

Good morning. We are Group 4, presenting an analysis of surface electromyography recordings associated with unilateral knee extension. Our focus is how the stored EMG channel behaves across formal sets and between sides. We distinguish the course guide’s intended protocol from what the files actually document. The central question is what these recordings support about muscle activity—and what they do not establish about fatigue.

## Slide 2 — Study aim and intended protocol

**Speaking time:** 00:40
**Visual reference:** Guide target and protocol statement

**Complete speaking script:**

The course guide frames the exercise as unilateral knee extension, with the rectus femoris as the intended target and a planned load of seventy percent of one-repetition maximum. This keeps the central question tied to the observed signal rather than assumed protocol details. Our aim here is narrower: describe signal-derived activity trends in the available files. The export does not verify electrode placement or the actual load used. That distinction matters because muscle identity, force, and workload affect how EMG amplitude and frequency should be interpreted.

## Slide 3 — What the recordings contain

**Speaking time:** 00:50
**Visual reference:** 15 IDs, 140 files, sample count, missing combinations

**Complete speaking script:**

The inventory contains recordings from fifteen anonymized participant IDs. There are one hundred forty CSV files: one hundred thirty-eight formal records labeled R1 through R5, plus two preliminary or noisy records, R0 and R0_1, retained for quality control. Together the files contain 4,256,257 sample points. Twelve expected participant, limb, and series combinations are absent. This limits complete within-subject coverage. Filenames encode side and record identifiers, but the export does not include synchronized force, video, or mechanical repetition markers. Therefore, detected activation events are not automatically counted as repetitions.

## Slide 4 — From CSV to EMG features

**Speaking time:** 00:55
**Visual reference:** Original-to-filtered processing diagram and filter labels

**Complete speaking script:**

The CSV fields are Relative_Time_s, ADC_Value, and Voltage_mV. Negative ADC values are present, and the export does not establish the device’s ADC resolution, analog chain, physical voltage calibration, or whether the stored channel is raw, rectified, or an envelope. The analysis preserved the original samples and treated discontinuities as separate segments. It compared high-pass cutoffs of twenty and forty hertz, a Butterworth band-pass from twenty to four hundred hertz, and a Hamming-window FIR band-pass. A notch was not applied because the line-interference screen did not meet the preset criterion. Filter comparison therefore concerns signal conditioning, not proof that one filter best represents muscle physiology.

## Slide 5 — RMS, MDF, MNF, and the spectrum

**Speaking time:** 00:50
**Visual reference:** Equations and conceptual signal/spectrum schematic

**Complete speaking script:**

RMS is the square root of the mean squared signal and summarizes amplitude; it is not a direct measure of force. The median frequency, or MDF, divides the power spectrum into equal lower- and upper-frequency power. Mean frequency, or MNF, is the power-weighted average frequency: the sum of frequency times power divided by total power. The Fourier transform describes frequency content, while the power spectral density estimates how signal power is distributed across frequency. During sustained, controlled contractions, MDF and MNF may decline as the spectrum shifts lower. RMS can rise, remain stable, or fall. FFT and PSD estimates are meaningful only when the segment is long enough and contains continuous muscle activity.

## Slide 6 — Activity masking and spectral validity

**Speaking time:** 01:00
**Visual reference:** Activity/rest mask and spectral analysis intervals

**Complete speaking script:**

Rest and low-activity windows can produce unstable or extreme spectral estimates, so the analysis separated activity detection from spectral measurement. The activity envelope identifies candidate active intervals; weak contractions were retained when their signal quality and continuous duration supported measurement. Window overlap is not the same parameter as active occupancy: one controls how often analysis windows advance, while the other describes how much of a window contains activity. MDF and MNF were calculated only on continuous, active, valid segments, never by joining across rest or gaps. Longer segments improve spectral stability but reduce the amount of usable data. We compared candidate durations instead of relying on zero-padding to imply more physical resolution. This changes coverage, so masked and unmasked results must be interpreted together.

## Slide 7 — Within-limb progression from R1 to R5

**Speaking time:** 01:00
**Visual reference:** RMS, MDF, MNF changes and 95% confidence intervals

**Complete speaking script:**

The R1-to-R5 comparison includes twenty-seven matched participant–limb records. Mean RMS changed by plus 8.57 percent, with a 95-percent confidence interval from minus 0.68 to plus 18.26 percent. MDF changed by minus 0.54 hertz, with an interval from minus 3.88 to plus 2.66 hertz. MNF changed by minus 3.31 hertz, with an interval from minus 6.88 to minus 0.28 hertz; this estimate was sensitive to the spectral estimator. These directions are not fully consistent across indicators, and the first two intervals include no change. Different activity durations, pauses, effort, and signal quality can also explain variation. We therefore describe possible channel trends, not confirmed fatigue. These are aggregate paired changes, not evidence that each participant followed the same trajectory.

## Slide 8 — Bilateral comparison

**Speaking time:** 01:00
**Visual reference:** Paired left–right records and confidence intervals

**Complete speaking script:**

For the bilateral analysis, only records with both sides available were paired. The inventory provides sixty-three paired left–right records per series. The confidence intervals for RMS, MDF, and MNF include zero, so these data do not demonstrate a consistent side difference. That does not prove the limbs are physiologically equivalent, and it does not identify a side with greater functional fatigue. Filename-based laterality helps organize the files, but electrode placement, movement phase, force, and mechanics were not independently synchronized. The appropriate conclusion is limited: no robust bilateral difference was detected in the available channel-level indicators under this analysis. Paired analysis reduces imbalance but cannot correct unmeasured differences in execution or electrode placement.

## Slide 9 — Machine-learning benchmark

**Speaking time:** 01:00
**Visual reference:** Grouped LOSO Macro-F1 and target definition

**Complete speaking script:**

The local machine-learning benchmark predicts recording position: whether a window belongs to the first, middle, or final third of a recording. It does not predict a fatigue label. Grouped leave-one-subject-out validation evaluates performance on participants excluded from training, and the reported temporal Macro-F1 is 0.716 with a fold-level spread of 0.146. A high score here means the model can distinguish temporal thirds under this labeling scheme. It does not show that it detects physiological fatigue. The separate Borg and Cerqueira tasks concern perceived effort or fatigue in upper-limb protocols, so their models cannot be transferred as validated quadriceps-fatigue detectors. Compatible external physiological validation has not been demonstrated. The reported score should not be read as detection accuracy for a clinical event or its onset.

## Slide 10 — Interpretation, limitations, and conclusions

**Speaking time:** 02:15
**Visual reference:** Conclusions and evidence needed

**Complete speaking script:**

Muscular fatigue can alter motor-unit behavior and the spectral distribution of surface EMG. Under sustained contractions with controlled force, a progressive decrease in MDF or MNF can be compatible with changes in muscle-fiber conduction velocity. But this expected pattern is conditional, not a universal diagnostic rule. Dynamic knee extension adds movement, changing joint angle, recruitment, pauses, and electrode motion. Those factors can shift amplitude and frequency even without a verified loss of force. RMS is especially dependent on recruitment and effort, so an increase is not required for fatigue, and a decrease does not rule it out.

Our recordings show a modest average increase in RMS and downward average changes in MDF and MNF from R1 to R5, but only the MNF confidence interval excludes zero, and that estimate is estimator-sensitive. The bilateral intervals include zero. The signals therefore support exploratory channel-level trends; they do not establish a fatigue onset, a fatigue level, or a loss of quadriceps force. The temporal classifier answers a different question and must remain labeled as a time-position benchmark.

There are also limits in the source data: electrode site, physical voltage calibration, the hardware sampling chain, actual load, rest periods, cadence, and synchronized repetitions are not confirmed by the CSV export. The guide’s rectus-femoris and seventy-percent one-repetition-maximum protocol is an intended design, not proof that each recording followed it.

A stronger follow-up should verify electrode placement, calibrate the acquisition chain, record MVC or force, and synchronize every repetition with movement and rest. It should collect an independent fatigue reference and evaluate thresholds on participants held out from model selection. Until then, we conclude that the project characterizes EMG channel behavior during the intended unilateral knee-extension task, while physiological fatigue detection remains unvalidated. Thank you.
