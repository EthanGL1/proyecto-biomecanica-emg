import csv
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reproducir_emg_referencia import (  # noqa: E402
    analyze_arrays,
    analyze_cohort_arrays,
    main,
    periodogram,
    read_cohort_signal_csv,
    read_cohort_signal_fragments,
    read_signal_csv,
    split_contiguous_runs,
    write_synthetic_fixture,
)


class PublicEmgPipelineTests(unittest.TestCase):
    def _write_cohort_csv(self, path: Path, groups, *, sample_count: int = 2000) -> Path:
        """Write a test-only cohort fixture with synthetic identifiers and signals."""
        fs_hz = 1000.0
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("subject_id", "side", "series", "time_s", "adc_value"))
            for group_index, (subject_id, side, series) in enumerate(groups):
                times = np.arange(sample_count, dtype=float) / fs_hz
                signal = np.sin(2.0 * np.pi * (70.0 + group_index) * times)
                writer.writerows(
                    (subject_id, side, series, f"{time_s:.9f}", f"{value:.12f}")
                    for time_s, value in zip(times, signal)
                )
        return path

    def test_gap_segmentation_splits_without_inserting_samples(self):
        times = np.asarray([0.000, 0.001, 0.002, 0.003, 0.004, 0.010, 0.011, 0.012])

        self.assertEqual(split_contiguous_runs(times), [(0, 5), (5, 8)])

    def test_hann_periodogram_recovers_tone_power_and_frequency(self):
        fs_hz = 1000.0
        times = np.arange(2000) / fs_hz
        signal = 2.0 * np.sin(2 * np.pi * 80.0 * times)

        frequencies, amplitude, psd = periodogram(signal, fs_hz)

        self.assertEqual(frequencies[np.argmax(amplitude)], 80.0)
        self.assertAlmostEqual(float(np.sum(psd) * fs_hz / signal.size), 2.0, places=2)

    def test_filtering_uses_common_edge_mask_and_exports_window_metrics_and_stft(self):
        fs_hz = 1000.0
        times = np.arange(6000) / fs_hz
        signal = (
            20.0 * np.sin(2 * np.pi * 80.0 * times)
            + 5.0 * np.sin(2 * np.pi * 300.0 * times)
            + 20.0 * np.sin(2 * np.pi * 10.0 * times)
        )

        result = analyze_arrays({"R1": (times, signal)})

        self.assertEqual(result.summary["series"], ["R1"])
        self.assertEqual(result.summary["filter_branches"], ["control", "hp20", "hp40", "bp20_400", "fir_bp20_400"])
        self.assertEqual(result.segment_rows[0]["common_valid_samples"], 5900)
        self.assertEqual(len(result.mask_rows), 6000)
        self.assertEqual(sum(row["common_valid"] for row in result.mask_rows), 5900)
        self.assertEqual(len(result.feature_rows), 45)
        self.assertEqual(len(result.fft_rows), 45 * 501)
        self.assertEqual(len(result.psd_rows), 5 * 501)
        self.assertEqual(len(result.stft_rows), 45 * 501)
        self.assertFalse(result.mask_rows[0]["common_valid"])
        self.assertTrue(result.mask_rows[50]["common_valid"])
        self.assertFalse(result.mask_rows[-50]["common_valid"])
        feature = next(row for row in result.feature_rows if row["filter_branch"] == "control")
        self.assertAlmostEqual(feature["rms_adc"], (20.0**2 + 5.0**2 + 20.0**2) ** 0.5 / 2**0.5, places=1)
        self.assertTrue(75.0 <= feature["mdf_hz"] <= 85.0)
        self.assertTrue(80.0 <= feature["mnf_hz"] <= 300.0)
        hp20 = next(row for row in result.feature_rows if row["filter_branch"] == "hp20")
        self.assertLess(hp20["rms_adc"], feature["rms_adc"] * 0.8)

    def test_pipeline_keeps_gap_runs_separate_and_never_builds_cross_gap_windows(self):
        fs_hz = 1000.0
        times = np.concatenate((np.arange(3000) / fs_hz, np.arange(3000, 6000) / fs_hz + 0.010))
        signal = np.sin(2 * np.pi * 80.0 * times)

        result = analyze_arrays({"R1": (times, signal)})

        self.assertEqual(len(result.segment_rows), 2)
        self.assertEqual([row["common_valid_samples"] for row in result.segment_rows], [2900, 2900])
        self.assertEqual(len(result.feature_rows), 30)
        self.assertTrue(all(row["time_s"] < 3.0 for row in result.feature_rows if row["run_index"] == 1))
        self.assertTrue(all(row["time_s"] > 3.0 for row in result.feature_rows if row["run_index"] == 2))

    def test_irregular_sampling_run_is_reported_but_not_filtered_or_labeled(self):
        intervals = np.tile(np.asarray([0.0008, 0.0012]), 3000)
        times = np.concatenate(([0.0], np.cumsum(intervals)))
        signal = np.sin(2 * np.pi * 80.0 * times)

        result = analyze_arrays({"R1": (times, signal)})

        self.assertEqual(result.segment_rows[0]["status"], "irregular_sampling_interval_spread_over_5_percent")
        self.assertEqual(result.segment_rows[0]["common_valid_samples"], 0)
        self.assertEqual(result.feature_rows, [])
        self.assertFalse(any(row["common_valid"] for row in result.mask_rows))

    def test_sampling_rate_at_or_below_nyquist_requirement_is_skipped(self):
        fs_hz = 500.0
        times = np.arange(3000) / fs_hz
        signal = np.sin(2 * np.pi * 80.0 * times)

        result = analyze_arrays({"R1": (times, signal)})

        self.assertEqual(result.segment_rows[0]["status"], "sampling_frequency_must_exceed_800_hz")
        self.assertEqual(result.feature_rows, [])

    def test_clock_reset_is_rejected_instead_of_mislabeling_elapsed_time_thirds(self):
        first = np.arange(3000) / 1000.0
        times = np.concatenate((first, first))
        signal = np.sin(2 * np.pi * 80.0 * times)

        with self.assertRaisesRegex(ValueError, "timestamps must not go backwards"):
            analyze_arrays({"R1": (times, signal)})

    def test_fixture_and_cli_run_end_to_end_without_identity_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "demo"
            fixture_path = output_dir / "synthetic_emg.csv"
            output_dir.mkdir()
            write_synthetic_fixture(fixture_path)

            with fixture_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(set(reader.fieldnames or []), {"series", "time_s", "adc_value"})
                series = {row["series"] for row in reader}
            self.assertEqual(series, {"R1", "R2", "R3", "R4", "R5"})

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--input", str(fixture_path), "--output-dir", str(output_dir), "--overwrite"]), 0)

            for name in ("summary.json", "segments.csv", "mask.csv", "features.csv", "fft.csv", "psd.csv", "stft.csv"):
                self.assertTrue((output_dir / name).is_file(), name)
            with (output_dir / "features.csv").open(newline="", encoding="utf-8") as handle:
                feature_rows = list(csv.DictReader(handle))
            self.assertEqual({row["series"] for row in feature_rows}, {"R1", "R2", "R3", "R4", "R5"})
            self.assertEqual({row["time_position_proxy"] for row in feature_rows}, {"temprana", "transicion", "tardia"})
            self.assertFalse({"participant_id", "recording_id", "subject_id"} & set(feature_rows[0]))

    def test_documented_demo_mode_generates_fixture_and_results_in_a_new_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "demo"

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--demo", "--output-dir", str(output_dir)]), 0)

            self.assertTrue((output_dir / "synthetic_emg.csv").is_file())
            with (output_dir / "summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["series"], ["R1", "R2", "R3", "R4", "R5"])
            self.assertEqual(summary["runs_processed"], 10)
            self.assertIn("not a physiological EMG simulator", summary["synthetic_fixture"])
            self.assertIn("no study results are reproduced", summary["data_scope"])

    def test_csv_input_summary_is_not_labeled_as_a_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "caller_input.csv"
            output_dir = root / "processed"
            fs_hz = 1000.0
            times = np.arange(5000) / fs_hz
            signal = np.sin(2 * np.pi * 90.0 * times)
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("series", "time_s", "adc_value"))
                writer.writerows(("R1", t, x) for t, x in zip(times, signal))

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--input", str(source), "--output-dir", str(output_dir)]), 0)

            with (output_dir / "summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["input_source"], "csv_input")
            self.assertNotIn("synthetic_fixture", summary)
            self.assertIn("caller-supplied CSV", summary["data_scope"])
            self.assertEqual(summary["runs_processed"], 1)

    def test_csv_reader_rejects_identity_columns_and_non_r1_r5_series(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.csv"
            source.write_text("series,time_s,adc_value,participant_id\nR1,0,1,example\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columns"):
                read_signal_csv(source)

            source.write_text("series,time_s,adc_value\nR0,0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "R1-R5"):
                read_signal_csv(source)

    def test_cohort_reader_accepts_explicit_subject_side_series_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "cohort.csv"
            groups = [("synthetic_subject_01", side, series)
                      for side in ("left", "right")
                      for series in ("R1", "R2", "R3", "R4", "R5")]
            self._write_cohort_csv(source, groups, sample_count=3)

            recordings = read_cohort_signal_csv(source)

        self.assertEqual(set(recordings), set(groups))
        self.assertTrue(all(times.size == values.size == 3 for times, values in recordings.values()))

    def test_cohort_reader_accepts_required_columns_in_any_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "reordered_cohort.csv"
            source.write_text(
                "adc_value,series,side,time_s,subject_id\n"
                "1.0,R1,left,0.000,synthetic_subject_01\n"
                "2.0,R1,left,0.001,synthetic_subject_01\n"
                "3.0,R1,left,0.002,synthetic_subject_01\n",
                encoding="utf-8",
            )

            recordings = read_cohort_signal_csv(source)

        self.assertEqual(set(recordings), {("synthetic_subject_01", "left", "R1")})
        times, values = recordings[("synthetic_subject_01", "left", "R1")]
        self.assertEqual(times.tolist(), [0.0, 0.001, 0.002])
        self.assertEqual(values.tolist(), [1.0, 2.0, 3.0])

    def test_cohort_reader_rejects_missing_or_conflicting_group_fields_and_bad_times(self):
        invalid_csvs = (
            ("subject_id,side,series,time_s,adc_value\n,left,R1,0,1\n", "subject_id"),
            ("subject_id,side,series,time_s,adc_value\nsynthetic_subject_01,LEFT,R1,0,1\n", "side"),
            ("subject_id,side,series,time_s,adc_value\nsynthetic_subject_01,left,R0,0,1\n", "series"),
            ("subject_id,side,series,time_s,adc_value\nsynthetic_subject_01,left,R1,0,1\nsynthetic_subject_01,left,R1,0,2\n", "duplicate sample"),
            ("subject_id,side,series,time_s,adc_value\nsynthetic_subject_01,left,R1,0,1\nsynthetic_subject_01,left,R1,0.1,2\nsynthetic_subject_01,left,R1,0.05,3\n", "reset"),
            ("subject_id,side,series,time_s,adc_value\nsynthetic_subject_01,left,R1,0,1\nsynthetic_subject_01,left,R2,0,2\nsynthetic_subject_01,left,R1,1,3\n", "block"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "invalid.csv"
            source.write_text("side,series,time_s,adc_value\nleft,R1,0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columns"):
                read_cohort_signal_csv(source)
            for contents, expected in invalid_csvs:
                with self.subTest(expected=expected):
                    source.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected):
                        read_cohort_signal_csv(source)

    def test_cohort_analysis_rejects_groups_without_a_complete_valid_window(self):
        times = np.arange(1000, dtype=float) / 1000.0
        signal = np.sin(2.0 * np.pi * 80.0 * times)
        recordings = {("synthetic_subject_01", "left", "R1"): (times, signal)}

        with self.assertRaisesRegex(ValueError, "too short"):
            analyze_cohort_arrays(recordings)

    def test_cohort_cli_smoke_processes_synthetic_15_subject_fixture_without_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "synthetic_15_subject_cohort.csv"
            output_dir = root / "cohort_output"
            groups = [(f"synthetic_subject_{index:02d}", "left" if index % 2 else "right", f"R{(index % 5) + 1}")
                      for index in range(1, 16)]
            self._write_cohort_csv(source, groups)

            completed = subprocess.run(
                [sys.executable, str(ROOT / "src" / "reproducir_emg_referencia.py"),
                 "--cohort-input", str(source), "--output-dir", str(output_dir)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["subject_count"], 15)
            self.assertEqual(summary["recording_group_count"], 15)
            self.assertFalse(summary["cohort_comparisons_computed"])
            self.assertEqual(summary["fatigue_target"], "none; elapsed-time thirds are a position proxy only")
            rows = []
            for filename in ("segments.csv", "mask.csv", "features.csv", "fft.csv", "psd.csv", "stft.csv"):
                with (output_dir / filename).open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    self.assertTrue({"subject_id", "side", "series"}.issubset(reader.fieldnames or []))
                    file_rows = list(reader)
                    self.assertTrue(file_rows, filename)
                    self.assertEqual({row["subject_id"] for row in file_rows}, {group[0] for group in groups})
                    self.assertEqual({row["side"] for row in file_rows}, {"left", "right"})
                    if filename == "features.csv":
                        rows = file_rows
            self.assertEqual({row["subject_id"] for row in rows}, {group[0] for group in groups})
            self.assertEqual({row["side"] for row in rows}, {"left", "right"})
            self.assertTrue(all(row["series"].startswith("R") for row in rows))

    def test_fragment_reader_reassembles_chronologically_and_preserves_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.csv"
            second = root / "second.csv"
            first.write_text(
                "subject_id,side,series,time_s,adc_value\n"
                "synthetic_subject_01,left,R1,0.000,1\n"
                "synthetic_subject_01,left,R1,0.001,2\n",
                encoding="utf-8",
            )
            second.write_text(
                "subject_id,side,series,time_s,adc_value\n"
                "synthetic_subject_01,left,R1,0.010,3\n"
                "synthetic_subject_01,left,R1,0.011,4\n",
                encoding="utf-8",
            )

            merged = read_cohort_signal_fragments([second, first])

        times, values = merged[("synthetic_subject_01", "left", "R1")]
        self.assertEqual(times.tolist(), [0.0, 0.001, 0.01, 0.011])
        self.assertEqual(values.tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(split_contiguous_runs(times), [(0, 2), (2, 4)])

    def test_fragment_reader_rejects_overlapping_or_duplicate_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.csv"
            second = root / "overlap.csv"
            first.write_text(
                "subject_id,side,series,time_s,adc_value\n"
                "synthetic_subject_01,left,R1,0.000,1\n"
                "synthetic_subject_01,left,R1,0.001,2\n",
                encoding="utf-8",
            )
            second.write_text(
                "subject_id,side,series,time_s,adc_value\n"
                "synthetic_subject_01,left,R1,0.001,2\n"
                "synthetic_subject_01,left,R1,0.002,3\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "overlap or duplicate timestamp"):
                read_cohort_signal_fragments([first, second])

    def test_cli_fragment_reassembly_matches_whole_recording_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = root / "full.csv"
            first = root / "part_01.csv"
            second = root / "part_02.csv"
            fs_hz = 1000.0
            times = np.arange(6000, dtype=float) / fs_hz
            signal = np.sin(2.0 * np.pi * 83.0 * times) + 0.1 * np.sin(2.0 * np.pi * 7.0 * times)

            def write_rows(path, start, stop):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("subject_id", "side", "series", "time_s", "adc_value"))
                    writer.writerows(
                        ("synthetic_subject_01", "left", "R1", f"{times[i]:.9f}", f"{signal[i]:.12f}")
                        for i in range(start, stop)
                    )

            write_rows(full, 0, len(times))
            write_rows(first, 0, 3000)
            write_rows(second, 3000, len(times))
            full_output = root / "whole_output"
            fragment_output = root / "fragment_output"

            full_run = subprocess.run(
                [sys.executable, str(ROOT / "src" / "reproducir_emg_referencia.py"),
                 "--cohort-input", str(full), "--output-dir", str(full_output)],
                cwd=ROOT, check=False, capture_output=True, text=True,
            )
            fragment_run = subprocess.run(
                [sys.executable, str(ROOT / "src" / "reproducir_emg_referencia.py"),
                 "--cohort-fragments", str(second), str(first), "--output-dir", str(fragment_output)],
                cwd=ROOT, check=False, capture_output=True, text=True,
            )

            self.assertEqual(full_run.returncode, 0, full_run.stderr)
            self.assertEqual(fragment_run.returncode, 0, fragment_run.stderr)
            for filename in ("segments.csv", "mask.csv", "features.csv", "fft.csv", "psd.csv", "stft.csv"):
                self.assertEqual((fragment_output / filename).read_bytes(), (full_output / filename).read_bytes(), filename)
            summary = json.loads((fragment_output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["input_source"], "cohort_fragments")
            self.assertEqual(summary["fragment_file_count"], 2)
            self.assertEqual(summary["sample_count"], 6000)
            self.assertEqual(summary["runs_processed"], 1)

    def test_cli_refuses_output_inside_public_checkout(self):
        with self.assertRaisesRegex(ValueError, "outside the public source checkout"):
            main(["--demo", "--output-dir", str(ROOT / "forbidden-demo")])

    def test_cli_refuses_to_overwrite_its_input_as_a_generated_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "out"
            output_dir.mkdir()
            input_path = output_dir / "features.csv"
            input_path.write_text("series,time_s,adc_value\nR1,0,1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "input CSV path conflicts"):
                main(["--input", str(input_path), "--output-dir", str(output_dir), "--overwrite"])


if __name__ == "__main__":
    unittest.main()
