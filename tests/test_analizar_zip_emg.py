import sys
import tempfile
import unittest
import zipfile
import csv
import io
import math
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analizar_zip_emg import (
    ArchiveSafetyError,
    discover_recordings,
    discover_single_csv,
    default_output_dir,
    _render_index,
    main,
)


RAW_HEADER = b"Relative_Time_s,ADC_Value,Voltage_mV\n0.000,0,0.00\n"


class DiscoverRecordingsTests(unittest.TestCase):
    def make_archive(self, members):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "input.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        return path

    def test_assigns_pseudonyms_series_and_side_without_exposing_folder_names(self):
        archive = self.make_archive(
            {
                "patient-name-one/iz_R1.csv": RAW_HEADER,
                "patient-name-two/dr_R2.csv": RAW_HEADER,
            }
        )

        records = discover_recordings(archive)

        self.assertEqual(len(records), 2)
        self.assertEqual(
            [(r.participant_id, r.series, r.side, r.status) for r in records],
            [("P01", "R1", "left", "ready"), ("P02", "R2", "right", "ready")],
        )
        self.assertNotIn("patient-name-one", repr(records))
        self.assertNotIn("patient-name-two", repr(records))

    def test_keeps_preliminary_r0_records_out_of_formal_processing(self):
        archive = self.make_archive(
            {
                "participant/R0.csv": RAW_HEADER,
                "participant/R0_1.csv": RAW_HEADER,
            }
        )

        records = discover_recordings(archive)

        self.assertEqual([r.series for r in records], ["R0", "R0_1"])
        self.assertTrue(all(r.status == "preliminary_qc_only" for r in records))

    def test_marks_ambiguous_or_missing_series_for_review(self):
        archive = self.make_archive(
            {
                "participant/iz_R1_R2.csv": RAW_HEADER,
                "participant/recording.csv": RAW_HEADER,
            }
        )

        records = discover_recordings(archive)

        self.assertEqual([r.status for r in records], ["ambiguous_series", "series_missing"])
        self.assertTrue(all(r.series is None for r in records))

    def test_marks_non_raw_schema_as_unsupported(self):
        archive = self.make_archive({"participant/iz_R1.csv": b"time,value\n0,1\n"})

        records = discover_recordings(archive)

        self.assertEqual(records[0].status, "unsupported_schema")

    def test_rejects_zip_path_traversal_before_extraction(self):
        archive = self.make_archive({"../outside/iz_R1.csv": RAW_HEADER})

        with self.assertRaises(ArchiveSafetyError):
            discover_recordings(archive)

    def test_rejects_windows_absolute_and_duplicate_member_paths(self):
        absolute = self.make_archive({"C:\\Temp\\iz_R1.csv": RAW_HEADER})
        duplicate = self.make_archive({"participant/iz_R1.csv": RAW_HEADER})
        with zipfile.ZipFile(duplicate, "a") as archive_file:
            archive_file.writestr("participant/iz_R1.csv", RAW_HEADER)

        with self.assertRaises(ArchiveSafetyError):
            discover_recordings(absolute)
        with self.assertRaises(ArchiveSafetyError):
            discover_recordings(duplicate)


class DefaultOutputDirectoryTests(unittest.TestCase):
    def test_prefers_user_local_app_data_over_documents(self):
        with patch.dict("os.environ", {"LOCALAPPDATA": r"C:\Users\tester\AppData\Local"}):
            self.assertEqual(
                default_output_dir(),
                Path(r"C:\Users\tester\AppData\Local\EMG_Biomecanica\Resultados"),
            )


class SinglePageOverallReportTests(unittest.TestCase):
    def test_overall_html_shows_all_event_times_without_overlapping_lanes_or_record_links(self):
        with tempfile.TemporaryDirectory() as temp_name:
            run_dir = Path(temp_name)
            rows = [
                {
                    "recording_id": "REC0001", "participant_pseudonym": "P01", "side": "right",
                    "series": "R1", "status": "processed", "candidate_event_count": 2,
                    "output_folder": "recordings/REC0001",
                },
                {
                    "recording_id": "REC0002", "participant_pseudonym": "P02", "side": "left",
                    "series": "R5", "status": "processed", "candidate_event_count": 1,
                    "output_folder": "recordings/REC0002",
                },
            ]
            with (run_dir / "activity_event_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "recording_id", "participant_pseudonym", "side", "series", "event_candidate_index",
                    "start_time_s", "end_time_s", "duration_s", "threshold_rms_adc", "mean_rms_adc",
                    "peak_rms_adc", "quality", "interpretation",
                ])
                writer.writeheader()
                writer.writerows([
                    {"recording_id": "REC0001", "participant_pseudonym": "P01", "side": "right", "series": "R1",
                     "event_candidate_index": 1, "start_time_s": 0, "end_time_s": 2, "duration_s": 2,
                     "threshold_rms_adc": 4, "mean_rms_adc": 5, "peak_rms_adc": 7,
                     "quality": "common_mask_valid", "interpretation": "candidate_activity_only"},
                    {"recording_id": "REC0001", "participant_pseudonym": "P01", "side": "right", "series": "R1",
                     "event_candidate_index": 2, "start_time_s": 1, "end_time_s": 3, "duration_s": 2,
                     "threshold_rms_adc": 4, "mean_rms_adc": 6, "peak_rms_adc": 8,
                     "quality": "common_mask_valid", "interpretation": "candidate_activity_only"},
                    {"recording_id": "REC0002", "participant_pseudonym": "P02", "side": "left", "series": "R5",
                     "event_candidate_index": 1, "start_time_s": 2, "end_time_s": 3, "duration_s": 1,
                     "threshold_rms_adc": 4, "mean_rms_adc": 5, "peak_rms_adc": 7,
                     "quality": "common_mask_valid", "interpretation": "candidate_activity_only"},
                ])
            (run_dir / "recordings" / "REC0001").mkdir(parents=True)
            (run_dir / "recordings" / "REC0002").mkdir(parents=True)
            (run_dir / "recordings" / "REC0001" / "segments.csv").write_text(
                "series,run_index,start_time_s,end_time_s,fs_hz,interval_spread_ratio,sample_count,common_valid_samples,excluded_edge_samples,status\n"
                "R1,1,0,5,1000,0,5000,4900,100,processed\n", encoding="utf-8"
            )
            (run_dir / "recordings" / "REC0002" / "segments.csv").write_text(
                "series,run_index,start_time_s,end_time_s,fs_hz,interval_spread_ratio,sample_count,common_valid_samples,excluded_edge_samples,status\n"
                "R5,1,0,4,1000,0,4000,3900,100,processed\n", encoding="utf-8"
            )
            summary = {
                "csv_count": 2, "processed_count": 2, "skipped_count": 0, "candidate_event_count": 3,
                "model_loaded": False, "mechanical_repetition_prediction": None,
                "physiological_fatigue_prediction": None,
            }

            _render_index(run_dir, summary, rows)
            rendered = (run_dir / "index.html").read_text(encoding="utf-8")

            self.assertIn("name=\"overall-activity-timeline\"", rendered)
            self.assertIn("data-lane=\"0\"", rendered)
            self.assertIn("data-lane=\"1\"", rendered)
            self.assertIn("REC0002", rendered)
            self.assertIn("Fatigue prediction: unavailable", rendered)
            self.assertIn("0.000-2.000", rendered)
            self.assertNotIn("recordings/REC0001/dashboard.html", rendered)
            self.assertNotIn("<a href=", rendered)


class DiscoverSingleCsvTests(unittest.TestCase):
    def make_csv(self, name, content=RAW_HEADER):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / name
        path.write_bytes(content)
        return path

    def test_accepts_one_supported_csv_and_hides_its_original_filename(self):
        path = self.make_csv("participant-private-name_iz_R3.csv")

        record = discover_single_csv(path)

        self.assertEqual((record.recording_id, record.series, record.side, record.status),
                         ("REC0001", "R3", "left", "ready"))
        self.assertIsNone(record.participant_id)
        self.assertNotIn("participant-private-name", repr(record))

    def test_rejects_csv_without_a_supported_raw_schema(self):
        path = self.make_csv("dr_R2.csv", b"time,value\n0,1\n")

        with self.assertRaises(ArchiveSafetyError):
            discover_single_csv(path)

    def test_keeps_preliminary_csv_out_of_formal_processing(self):
        path = self.make_csv("R0_1.csv")

        with self.assertRaisesRegex(ArchiveSafetyError, "preliminar"):
            discover_single_csv(path)

    def test_processes_one_csv_end_to_end_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            csv_path = root / "private-folder" / "iz_R1.csv"
            csv_path.parent.mkdir()
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("Relative_Time_s", "ADC_Value", "Voltage_mV"))
                for index in range(5000):
                    time_s = index / 1000
                    adc = round(2048 + 180 * math.sin(2 * math.pi * 80 * time_s))
                    writer.writerow((f"{time_s:.6f}", adc, f"{adc * 3300 / 4095:.3f}"))

            output_base = root / "private-results"
            with redirect_stdout(io.StringIO()):
                exit_code = main(["--csv", str(csv_path), "--output-dir", str(output_base), "--no-open"])
            run_dir = next(output_base.glob("analisis_emg_*"))

            summary = __import__("json").loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            manifest = (run_dir / "run_manifest.csv").read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["processed_count"], 1)
            self.assertFalse(summary["model_loaded"])
            self.assertIsNone(summary["mechanical_repetition_prediction"])
            self.assertIsNone(summary["physiological_fatigue_prediction"])
            self.assertTrue((run_dir / "recordings" / "REC0001" / "dashboard.html").is_file())
            self.assertNotIn("private-folder", manifest)

    def test_uses_local_home_fallback_when_local_app_data_is_unset(self):
        with patch.dict("os.environ", {}, clear=True), patch("analizar_zip_emg.Path.home", return_value=Path("/home/tester")):
            self.assertEqual(
                default_output_dir(),
                Path("/home/tester/.local/share/EMG_Biomecanica/Resultados"),
            )


if __name__ == "__main__":
    unittest.main()
