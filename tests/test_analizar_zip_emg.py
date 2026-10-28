import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analizar_zip_emg import ArchiveSafetyError, discover_recordings, default_output_dir


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

    def test_uses_local_home_fallback_when_local_app_data_is_unset(self):
        with patch.dict("os.environ", {}, clear=True), patch("analizar_zip_emg.Path.home", return_value=Path("/home/tester")):
            self.assertEqual(
                default_output_dir(),
                Path("/home/tester/.local/share/EMG_Biomecanica/Resultados"),
            )


if __name__ == "__main__":
    unittest.main()
