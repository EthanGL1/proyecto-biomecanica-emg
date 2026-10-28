# Group 4 - 10-Minute Scientific Presentation Script

**Language:** English
**Target duration:** exactly 10:00 of assigned slide time; rehearse with a timer to match the speaking pace.
**Deck:** existing Canva presentation, "Defensa sEMG cuádriceps" (10 slides).
**Revision:** F57, 2026-10-28.
**Spoken-word target:** approximately 1,200 words; cues and the demonstration note are not spoken.

**Demonstration note (not spoken):** The F57 one-click launcher processes either a compatible ZIP or one formal R1–R5 CSV as a descriptive signal analysis. It loads no predictive model, does not verify mechanical repetitions, and does not predict physiological fatigue. The independent two-patient ZIP was not supplied with this run; therefore, this script reports no results for those patients. Use only an authorized input and a restricted local output folder.

Assigned slide timings sum to 10:00. The final timing depends on rehearsal pace; do not omit scientific limitations to meet the clock.

## Slide 1 - Quadriceps sEMG during unilateral knee extension (00:30)

**Visual cue:** Title, group, and study question.

**Script:**

Good morning. We are Group 4. We analyzed surface electromyography files associated with unilateral knee extension. Our focus is the signal processing and the changes in the stored channel across formal series and sides. We separate the protocol described in the course guide from the acquisition details that the CSV files cannot verify. The results are descriptive, not a validated fatigue diagnosis.

## Slide 2 - Study aim and intended protocol (00:40)

**Visual cue:** The guide's intended exercise and target muscle.

**Script:**

The course guide specifies unilateral knee extension, with the rectus femoris as the intended target and a planned load of seventy percent of one-repetition maximum. Those are protocol intentions, not measurements confirmed in each file. Our aim was to characterize the available signal and test whether its trends were compatible with established EMG analysis. The CSV export does not verify electrode placement, actual load, cadence, rest periods, or force. Those omissions limit physiological interpretation, even when signal calculations are reproducible.

## Slide 3 - What the recordings contain (00:50)

**Visual cue:** Participant-derived groups, formal/QC files, sample count, and missing combinations.

**Script:**

The inventory contains fifteen recording-derived participant groups and one hundred forty CSV files. One hundred thirty-eight are formal series R1 through R5; two files, R0 and R0_1, are preliminary or quality-control records. The files contain 4,256,257 sample points, with twelve expected participant, side, and series combinations absent. The CSV columns are Relative_Time_s, ADC_Value, and Voltage_mV. Negative ADC values occur. The last column is an arithmetic transformation, not a verified physical voltage calibration. No synchronized force, video, or mechanical repetition markers are included, so activity peaks cannot be called verified repetitions.

## Slide 4 - From CSV to EMG features (00:55)

**Visual cue:** The comparison of the stored signal and filter branches.

**Script:**

We preserved the original samples and treated timestamp gaps greater than one point five milliseconds as separate runs, without interpolation across them. The comparison included fourth-order Butterworth high-pass filters at twenty and forty hertz, a fourth-order twenty-to-four-hundred-hertz band-pass, and a one-hundred-one-tap Hamming-window FIR band-pass over the same band. Forward-backward filtering gives zero phase in valid runs, with edge effects still requiring care. A notch filter was not applied because the prespecified line-interference screen did not meet its ten-decibel criterion. With no clean reference channel, we cannot claim one filter recovered the true muscle signal better than the others.

## Slide 5 - RMS, MDF, MNF, and the spectrum (00:50)

**Visual cue:** Feature equations and the labeled conceptual signal/spectrum diagram.

**Script:**

For centered samples x, RMS is the square root of the mean of x squared; here it is expressed in ADC counts and is an amplitude proxy, not force. A Hann-windowed Fourier transform yields a one-sided power spectrum, with PSD expressed in squared ADC counts per hertz. MDF is the frequency where cumulative power reaches half the total power in the analyzed band. MNF is the power-weighted mean frequency: the sum of frequency times power divided by total power. In sustained contractions at approximately controlled force, MDF or MNF may decrease. RMS may increase, remain stable, or decrease. These patterns are context-dependent and are not universal fatigue thresholds.

## Slide 6 - Activity and spectral validity (01:00)

**Visual cue:** The activity/rest distinction and the continuous-segment rule.

**Script:**

The cohort's F39 analysis separated activity masking from spectral estimation. It retained weak contractions when signal quality and continuous duration were adequate, excluded rest from spectral metrics, and never joined activity across gaps. The occupancy threshold describes the active fraction within a window; it is different from the fifty-percent temporal overlap between successive windows. We compared continuous frames of zero point two five, zero point five, and one second at several occupancy settings. At fifty-percent occupancy, median spectral coverage declined from 27.6 to 19.1 to 2.7 percent as frame duration increased. Weak and late event retention also fell, and the one-second estimates were too sparse to estimate within-window dispersion. The F57 ZIP launcher is separate: it uses an exploratory Otsu threshold on fifty-millisecond RMS windows to list activity candidates. Otsu is not a fatigue detector, and the launcher does not load the F39 mask or a trained model.

## Slide 7 - Within-limb progression from R1 to R5 (01:00)

**Visual cue:** Cluster-bootstrap estimates and metric-specific denominators.

**Script:**

Twenty-seven same-participant, same-limb R1-to-R5 pairs were available. Metrics had different evaluable denominators: active RMS in twenty-six pairs across fifteen participants, and active MDF and MNF in twenty pairs across twelve participants. Participant-cluster estimates, with four thousand bootstrap resamples, were plus 8.571 percent for RMS, with a ninety-five-percent interval from minus 0.681 to plus 18.264; minus 0.535 hertz for MDF, interval minus 3.877 to plus 2.663; and minus 3.306 hertz for MNF, interval minus 6.878 to minus 0.277. Only the primary MNF interval excludes zero. Its direction is sensitive to the spectral estimator and duration settings. The series were not matched for force, intensity, rest, or duration, so these are channel trends, not confirmed fatigue.

## Slide 8 - Bilateral comparison (01:00)

**Visual cue:** Metric-specific left-right pairs and uncertainty.

**Script:**

There were sixty-three same-participant left-right pairs available per formal series. The primary active RMS contrast was minus 0.342 percent, with a ninety-five-percent interval from minus 9.865 to plus 8.837, evaluable in sixty pairs across fourteen participants. MDF was plus 0.994 hertz, interval minus 3.969 to plus 5.618, and MNF was plus 1.104 hertz, interval minus 3.751 to plus 5.675; each spectral metric was evaluable in forty-nine pairs across twelve participants. All three intervals include zero. This means the analysis did not detect a consistent difference; it does not prove physiological equivalence between legs. Laterality is read from filenames, and biomechanics or electrode position were not independently measured.

## Slide 9 - Temporal proxy benchmark (01:00)

**Visual cue:** Grouped LOSO score and target definition.

**Script:**

The previously reported local machine-learning benchmark classifies a recording window as belonging to its first, middle, or final temporal third. It uses grouped leave-one-subject-out evaluation across fifteen participants and reports Macro-F1 of 0.716 plus or minus 0.146. This score measures temporal-position classification, not repetitions, fatigue, force loss, fatigue onset, or a fatigue level. The F57 one-click runner does not load this benchmark or any other classifier. Borg and Cerqueira models address separate perceived-exertion or self-report targets in upper-limb protocols; they are not validated quadriceps-fatigue models. External physiological validation for this local task has not been demonstrated.

## Slide 10 - Interpretation, limits, and local demonstration (02:15)

**Visual cue:** The conclusions, next measurements, and the launcher in the project folder.

**Script:**

Muscle fatigue can change motor-unit behavior and the frequency distribution of surface EMG. During a sustained contraction with controlled force, a progressive decrease in MDF or MNF may be compatible with changes in muscle-fiber conduction velocity. But dynamic knee extension changes joint angle, recruitment, effort, and electrode motion; pauses and changing activity can also alter these indicators. RMS is not expected to move in one direction under every protocol. A spectral trend alone does not establish loss of force or a physiological fatigue level.

The recordings show a modest mean rise in active RMS and downward mean changes in MDF and MNF from R1 to R5. Only the primary MNF confidence interval excludes zero, and that estimate is sensitive to analysis choices. The bilateral intervals include zero. The sampling chain, physical voltage calibration, electrode placement, actual load, rest and cadence, and mechanically synchronized repetitions remain unverified. The correct conclusion is that the project describes trends in the stored EMG-derived channel for a task intended to study quadriceps extension. It does not confirm rectus-femoris fatigue or a fatigue onset.

For the local demonstration, double-click `INICIAR_INFERENCIA.bat`. If this is the first run, it prepares the Python environment and runs its checks. Select an authorized ZIP containing compatible CSV files or one formal R1–R5 CSV, then select a restricted local output folder. A single CSV is analyzed as one recording, without a participant grouping; a ZIP is processed recording by recording, with pseudonyms only when its folder structure supports grouping. The tool creates a recording manifest, activity-event candidates, a results index, and per-recording dashboards; the browser opens the index when processing completes. The outputs are descriptive features and activity candidates. The launcher loads no model, verifies no repetition count, and emits no fatigue prediction.

The independent two-patient ZIP was not supplied with this run, so we cannot show patient-specific results or claim external validation. When an authorized compatible ZIP is available, the same descriptive workflow can be demonstrated without using it to tune the method. A defensible repetition model requires synchronized repetition annotations. A fatigue model requires an independent fatigue target and participant-held-out evaluation. The next study should verify electrode site and acquisition calibration, record force or MVC, synchronize movement and repetitions, and collect an independent fatigue reference. Thank you.
