from pathlib import Path
import csv
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from reproducir_emg_referencia import analyze_cohort_arrays, write_result, _write_csv


class CsvSpreadsheetSafetyTests(unittest.TestCase):
    def test_writer_neutralizes_formula_prefixes_only_at_export(self):
        identifiers = (
            "=1+1",
            "+1+1",
            "-1+1",
            "@SUM(A1:A2)",
            "\t=1+1",
            "\r=1+1",
            " \t=1+1",
            "\ufeff=1+1",
        )
        rows = [{"subject_id": value} for value in identifiers]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.csv"
            _write_csv(path, ("subject_id",), rows)
            with path.open(newline="", encoding="utf-8") as handle:
                exported = [row["subject_id"] for row in csv.DictReader(handle)]
        self.assertEqual(exported, ["'" + value for value in identifiers])
        self.assertEqual([row["subject_id"] for row in rows], list(identifiers))

    def test_cohort_export_neutralizes_identifier_in_all_six_tables(self):
        subject_id = "=1+1"
        times = np.arange(3000, dtype=float) / 1000.0
        signal = np.sin(2.0 * np.pi * 85.0 * times)
        result = analyze_cohort_arrays({(subject_id, "left", "R1"): (times, signal)})
        self.assertTrue(result.feature_rows)
        self.assertEqual(result.feature_rows[0]["subject_id"], subject_id)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            output.mkdir()
            write_result(result, output)
            for name in ("segments.csv", "mask.csv", "features.csv", "fft.csv", "psd.csv", "stft.csv"):
                with self.subTest(name=name):
                    with (output / name).open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertTrue(rows)
                    self.assertTrue(all(row["subject_id"] == "'" + subject_id for row in rows))


if __name__ == "__main__":
    unittest.main()