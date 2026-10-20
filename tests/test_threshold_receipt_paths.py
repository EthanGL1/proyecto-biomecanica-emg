import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analizar_umbral_senal_emg import analyze_dataset
from release_paths import ReleasePathError


class ThresholdReceiptPathTests(unittest.TestCase):
    def test_analyzer_preflights_all_receipt_archives_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            private_root = base / "private_arrays"
            private_root.mkdir()
            (base / "pseudonymized_csv").mkdir()
            manifest_path = base / "manifest_pseudonymized.csv"
            keys = [
                (f"P{participant:02d}", f"R{series}", leg)
                for participant in range(1, 16)
                for series in range(1, 6)
                for leg in ("left", "right")
            ][:140]
            with manifest_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("participant_id", "recording_id", "leg", "relative_path", "sha256_uncompressed"),
                )
                writer.writeheader()
                for participant, recording, leg in keys:
                    writer.writerow({
                        "participant_id": participant,
                        "recording_id": recording,
                        "leg": leg,
                        "relative_path": f"{participant}/{recording}_{leg}.csv",
                        "sha256_uncompressed": "0" * 64,
                    })

            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            outside = base / "outside.npz"
            outside.write_bytes(b"outside")
            receipt_rows = []
            for index, (participant, recording, leg) in enumerate(keys):
                if index == len(keys) - 1:
                    archive_path = outside
                else:
                    archive_path = private_root / f"record_{index:03d}.npz"
                    archive_path.write_bytes(f"archive-{index}".encode())
                receipt_rows.append({
                    "participant_id": participant,
                    "recording_id": recording,
                    "leg": leg,
                    "output_path": str(archive_path),
                    "output_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                })
            receipt_path = private_root / "receipt_filter_sensitivity_v1.json"
            receipt_path.write_text(json.dumps({
                "analysis": "filter_sensitivity_v1",
                "manifest_sha256": manifest_sha256,
                "records_private": receipt_rows,
            }), encoding="utf-8")

            output_root = base / "analysis_output"
            with self.assertRaisesRegex(ReleasePathError, "output_path"):
                analyze_dataset(private_root, output_root, base / "aggregate_output")
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
