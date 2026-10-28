import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ejecutar_demo_raw_rar import RawArchiveError, map_rar_members


class MapRarMembersTests(unittest.TestCase):
    def full_archive_names(self):
        return [
            f"{person}_set{series:02d}_{side}.csv"
            for person in ("P11", "P12")
            for series in range(1, 6)
            for side in ("dr", "iz")
        ]

    def test_maps_sets_to_r_series_and_pseudonymizes_people(self):
        mapping = map_rar_members(self.full_archive_names())

        self.assertEqual(len(mapping), 20)
        self.assertEqual(mapping["P11_set01_dr.csv"], "P01/dr_R1.csv")
        self.assertEqual(mapping["P11_set05_iz.csv"], "P01/iz_R5.csv")
        self.assertEqual(mapping["P12_set01_dr.csv"], "P02/dr_R1.csv")
        self.assertEqual(mapping["P12_set05_iz.csv"], "P02/iz_R5.csv")
        self.assertNotIn("P11", " ".join(mapping.values()))
        self.assertNotIn("P12", " ".join(mapping.values()))

    def test_rejects_incomplete_participant_series_laterality_grid(self):
        names = self.full_archive_names()
        names.remove("P12_set04_iz.csv")

        with self.assertRaisesRegex(RawArchiveError, "incompleta"):
            map_rar_members(names)

    def test_rejects_unsafe_or_unexpected_member_names(self):
        names = self.full_archive_names()
        names[0] = "../P11_set01_dr.csv"

        with self.assertRaises(RawArchiveError):
            map_rar_members(names)

    def test_rejects_case_insensitive_duplicates(self):
        names = self.full_archive_names()
        names.append("p11_SET01_DR.CSV")

        with self.assertRaisesRegex(RawArchiveError, "duplicado"):
            map_rar_members(names)


if __name__ == "__main__":
    unittest.main()
