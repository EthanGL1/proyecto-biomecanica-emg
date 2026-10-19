import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reproducir_emg_referencia import (  # noqa: E402
    analyze_arrays,
    main,
    periodogram,
    read_signal_csv,
    split_contiguous_runs,
    write_synthetic_fixture,
)


class PublicEmgPipelineTests(unittest.TestCase):
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

    def test_csv_reader_rejects_identity_columns_and_non_r1_r5_series(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.csv"
            source.write_text("series,time_s,adc_value,participant_id\nR1,0,1,example\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columns"):
                read_signal_csv(source)

            source.write_text("series,time_s,adc_value\nR0,0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "R1-R5"):
                read_signal_csv(source)

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
