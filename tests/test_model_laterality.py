import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analizar_benchmark_por_lateralidad import analyze_predictions, render_nested_laterality_heatmap, run_analysis  # noqa: E402
from release_paths import ReleasePathError  # noqa: E402


def _rows(participant, fold, side, *, series="R1", wrong=False):
    result = []
    labels = ("temprana", "transicion", "tardia")
    for index, label in enumerate(labels):
        predicted = 0 if wrong else index
        scores = [0.8, 0.1, 0.1] if wrong else [0.05, 0.05, 0.05]
        if not wrong:
            scores[index] = 0.9
        result.append({
            "outer_fold": str(fold),
            "held_out_pseudonym": participant,
            "recording_id": series,
            "leg": side,
            "set_index": str(index + 1),
            "true_proxy_label": label,
            "configuration_id": "cfg_a",
            "filter_branch": "hp20",
            "feature_set": "time",
            "model": "logistic",
            "selected_parameter": "0.1",
            "predicted_proxy_label": labels[predicted],
            "score_temprana": str(scores[0]),
            "score_transicion": str(scores[1]),
            "score_tardia": str(scores[2]),
        })
    return result


class ModelLateralityTests(unittest.TestCase):
    def test_public_cli_refuses_private_inputs_or_outputs_inside_checkout(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            with self.assertRaises(ReleasePathError):
                run_analysis(ROOT / "README.md", ROOT / "README.md", Path(temp_dir) / "out")

    def test_analyzes_every_configuration_and_adds_selected_pipeline_by_leg(self):
        predictions = (
            _rows("P001", 1, "left", series="R1")
            + _rows("P001", 1, "right", series="R1", wrong=True)
            + _rows("P001", 1, "left", series="R2")
            + _rows("P001", 1, "right", series="R2")
            + _rows("P002", 2, "right", series="R1")
        )
        selections = [
            {"outer_fold": "1", "selected_configuration_id": "cfg_a"},
            {"outer_fold": "2", "selected_configuration_id": "cfg_a"},
        ]

        summary, paired, by_series, paired_by_series, confusion, audit = analyze_predictions(
            predictions, selections, bootstrap_replicates=200, seed=8,
        )

        self.assertEqual(audit["participant_groups"], 2)
        self.assertEqual(audit["participants_by_side"], {"left": 1, "right": 2})
        self.assertEqual(audit["segments_by_side"], {"left": 6, "right": 9})
        self.assertEqual(audit["paired_participant_groups"], 1)
        self.assertEqual(audit["configuration_count_including_nested"], 2)
        indexed = {(row["configuration_id"], row["side"]): row for row in summary}
        self.assertEqual(indexed[("nested_selected_pipeline", "left")]["participants"], 1)
        self.assertEqual(indexed[("nested_selected_pipeline", "right")]["participants"], 2)
        self.assertAlmostEqual(indexed[("cfg_a", "left")]["macro_f1_mean_subject"], 1.0)
        self.assertEqual(len(paired), 18)
        series_index = {(row["configuration_id"], row["side"], row["series"]): row for row in by_series}
        self.assertEqual(series_index[("nested_selected_pipeline", "left", "R2")]["participants"], 1)
        self.assertAlmostEqual(series_index[("cfg_a", "right", "R1")]["macro_f1_mean_subject"], 7 / 12)
        self.assertTrue(any(row["series"] == "R2" and row["metric"] == "macro_f1" for row in paired_by_series))
        self.assertEqual(audit["participants_by_leg_series"]["left"]["R2"], 1)
        self.assertEqual(audit["paired_participants_by_series"], {"R1": 1, "R2": 1, "R3": 0, "R4": 0, "R5": 0})
        self.assertEqual(len(confusion), (2 * 2 * 9) + (2 * 2 * 2 * 9))
        self.assertNotIn("P001", repr(summary))
        self.assertNotIn("P002", repr(paired))
        self.assertNotIn("participant_id", summary[0])

    def test_nested_leg_series_heatmap_is_aggregate_and_covers_five_series(self):
        rows = []
        for side in ("left", "right"):
            for index in range(1, 6):
                rows.append({
                    "configuration_id": "nested_selected_pipeline",
                    "side": side,
                    "series": f"R{index}",
                    "participants": 14 if side == "left" else 15,
                    "macro_f1_mean_subject": 0.5 + index / 20,
                    "macro_roc_auc_ovr_mean_subject": 0.75 + index / 20,
                })
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "heatmap.svg"
            render_nested_laterality_heatmap(rows, output)
            svg = output.read_text(encoding="utf-8")
        self.assertIn("Rendimiento del clasificador por pierna y serie", svg)
        self.assertIn("R5", svg)
        self.assertIn("n=15", svg)
        self.assertNotIn("participant_id", svg)


if __name__ == "__main__":
    unittest.main()
