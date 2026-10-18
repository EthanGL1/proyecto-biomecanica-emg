import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analizar_filtros_matematicos import (  # noqa: E402
    design_filter_bank,
    frequency_response,
    run_audit,
    select_representative_record,
    summarize_filter_effects,
    summarize_paired_laterality,
    _Canvas,
)
from comparar_filtros_emg import split_contiguous_runs  # noqa: E402
from extraer_caracteristicas_proxy_temporal import _contiguous_pair_indices  # noqa: E402
from analizar_trayectorias_emg import iter_contiguous_windows  # noqa: E402
from release_paths import ReleasePathError  # noqa: E402


class FilterMathContractTests(unittest.TestCase):
    def test_filter_audit_refuses_receipts_or_outputs_inside_checkout(self):
        with self.assertRaises(ReleasePathError):
            run_audit(ROOT / "README.md", ROOT / "reports" / "forbidden")

    def test_raster_canvas_rotates_vertical_axis_labels(self):
        canvas = _Canvas(300, 300)

        canvas.text(150, 150, "Frequency (Hz)", 18, "middle", rotate=-90)

        difference = ImageChops.difference(canvas.image, Image.new("RGB", canvas.image.size, "white"))
        bounds = difference.getbbox()
        self.assertIsNotNone(bounds)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        self.assertLess(width, height)

    def test_representative_record_selection_uses_median_length_and_excludes_qc_only_files(self):
        records = [
            {"record_type": "series_R1_R5", "n_samples": 80, "n_runs": 1},
            {"record_type": "series_R1_R5", "n_samples": 100, "n_runs": 1},
            {"record_type": "series_R1_R5", "n_samples": 120, "n_runs": 1},
            {"record_type": "preliminary_R0", "n_samples": 100, "n_runs": 1},
        ]

        selected = select_representative_record(records)

        self.assertEqual(selected["n_samples"], 100)
        self.assertEqual(selected["record_type"], "series_R1_R5")

    def test_timestamp_roundoff_at_the_gap_threshold_stays_inside_the_same_run(self):
        times = np.asarray([22.2844, 22.2854, 22.2869, 22.2879])

        self.assertEqual(split_contiguous_runs(times, gap_s=0.0015), [(0, 4)])

    def test_a_gap_more_than_one_nanosecond_above_threshold_splits_the_run(self):
        times = np.asarray([0.0, 0.001, 0.002500002, 0.003500002])

        self.assertEqual(split_contiguous_runs(times, gap_s=0.0015), [(0, 2), (2, 4)])

    def test_feature_extraction_keeps_a_pair_at_the_gap_threshold_despite_roundoff(self):
        times = [22.2844, 22.2854, 22.2869, 22.2879]

        self.assertEqual(_contiguous_pair_indices(times, [0, 1, 2, 3]), [(0, 1), (1, 2), (2, 3)])

    def test_window_extraction_keeps_a_complete_window_at_the_gap_threshold(self):
        times = np.asarray([22.2844, 22.2854, 22.2869, 22.2879])
        signal = np.asarray([1.0, -1.0, 2.0, -2.0])

        windows = iter_contiguous_windows(times, signal, np.ones(4, dtype=bool), window_samples=4, gap_screen_s=0.0015)

        self.assertEqual(len(windows), 1)
        np.testing.assert_array_equal(windows[0]["signal"], signal)

    def test_forward_backward_butterworth_has_zero_phase_and_squared_magnitude_at_corner(self):
        bank = design_filter_bank(1000.0)
        hp20 = bank["hp20"]
        at_corner = frequency_response(hp20, np.asarray([20.0]), zero_phase=False)[0]
        zero_phase_at_corner = frequency_response(hp20, np.asarray([20.0]), zero_phase=True)[0]

        self.assertTrue(np.isclose(20 * np.log10(abs(at_corner)), -3.0103, atol=0.01))
        self.assertTrue(np.isclose(20 * np.log10(abs(zero_phase_at_corner)), -6.0206, atol=0.02))
        self.assertTrue(np.isclose(np.angle(zero_phase_at_corner), 0.0, atol=1e-12))


    def test_hamming_fir_is_symmetric_with_50_sample_group_delay_and_unity_center_gain(self):
        fir = design_filter_bank(1000.0)["fir_bp20_400"]
        coefficients = fir["coefficients"]

        self.assertEqual(len(coefficients), 101)
        self.assertTrue(np.allclose(coefficients, coefficients[::-1], atol=1e-12))
        self.assertEqual(fir["group_delay_samples"], 50)
        self.assertTrue(np.isclose(abs(frequency_response(fir, np.asarray([210.0]))[0]), 1.0, atol=1e-9))


    def test_effect_summary_uses_participant_medians_and_omits_subject_identifiers(self):
        records = []
        for participant, values in (("P001", [90.0, 80.0, 70.0]), ("P002", [0.0])):
            for value in values:
                records.append({
                    "participant_id": participant,
                    "record_type": "series_R1_R5",
                    "branches": {
                        "control": {"metrics_common": {"rms_adc": 100.0}},
                        "hp20": {"metrics_common": {"rms_adc": value}},
                    },
                })

        rows = summarize_filter_effects(records, branches=("hp20",), metrics=("rms_adc",))
        row = rows[0]

        self.assertEqual(row["n_participants"], 2)
        self.assertTrue(np.isclose(row["median_change_pct"], -60.0))
        self.assertTrue(np.isclose(row["q1_change_pct"], -80.0))
        self.assertTrue(np.isclose(row["q3_change_pct"], -40.0))
        self.assertNotIn("participant_id", row)
        self.assertNotIn("P001", repr(rows))
        self.assertNotIn("P002", repr(rows))


    def test_effect_summary_excludes_preliminary_records_from_primary_results(self):
        records = [
            {
                "participant_id": "P001",
                "record_type": "series_R1_R5",
                "branches": {
                    "control": {"metrics_common": {"rms_adc": 100.0}},
                    "hp20": {"metrics_common": {"rms_adc": 90.0}},
                },
            },
            {
                "participant_id": "P001",
                "record_type": "preliminary_R0",
                "branches": {
                    "control": {"metrics_common": {"rms_adc": 100.0}},
                    "hp20": {"metrics_common": {"rms_adc": 1.0}},
                },
            },
        ]

        rows = summarize_filter_effects(records, branches=("hp20",), metrics=("rms_adc",))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_participants"], 1)
        self.assertTrue(np.isclose(rows[0]["median_change_pct"], -10.0))

    def test_effect_summary_reports_left_right_and_joint_groups_by_participant(self):
        records = []
        for participant, leg, values in (
            ("P001", "left", [80.0, 60.0, 100.0]),
            ("P002", "left", [100.0]),
            ("P001", "right", [90.0]),
            ("P003", "right", [70.0]),
        ):
            for value in values:
                records.append({
                    "participant_id": participant,
                    "record_type": "series_R1_R5",
                    "leg": leg,
                    "branches": {
                        "control": {"metrics_common": {"rms_adc": 100.0}},
                        "hp20": {"metrics_common": {"rms_adc": value}},
                    },
                })

        rows = summarize_filter_effects(records, branches=("hp20",), metrics=("rms_adc",))

        indexed = {row["laterality"]: row for row in rows}
        self.assertEqual(set(indexed), {"all", "left", "right"})
        self.assertEqual(indexed["left"]["n_participants"], 2)
        self.assertEqual(indexed["left"]["recordings_n"], 4)
        # P001 contributes its within-person median (-20%), not four pooled rows.
        self.assertTrue(np.isclose(indexed["left"]["median_change_pct"], -10.0))
        self.assertEqual(indexed["right"]["n_participants"], 2)
        self.assertEqual(indexed["right"]["recordings_n"], 2)
        self.assertTrue(np.isclose(indexed["right"]["median_change_pct"], -20.0))
        self.assertEqual(indexed["all"]["n_participants"], 3)
        self.assertEqual(indexed["all"]["recordings_n"], 6)
        self.assertNotIn("P001", repr(rows))

        paired = summarize_paired_laterality(records, branches=("hp20",), metrics=("rms_adc",))
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["n_paired_participants"], 1)
        self.assertTrue(np.isclose(paired[0]["median_left_minus_right"], -10.0))
        self.assertNotIn("P001", repr(paired))


if __name__ == "__main__":
    unittest.main()
