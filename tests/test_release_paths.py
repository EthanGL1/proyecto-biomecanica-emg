from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import release_paths
from release_paths import ReleasePathError
import generar_proxy_temporal


class ReleasePathContainmentTests(unittest.TestCase):
    def _resolve_under_root(self):
        resolver = getattr(release_paths, "resolve_under_root", None)
        self.assertTrue(callable(resolver), "release_paths.resolve_under_root must enforce derived-path containment")
        return resolver

    def _validate_component(self):
        validator = getattr(release_paths, "validate_path_component", None)
        self.assertTrue(callable(validator), "release_paths.validate_path_component must reject path syntax in identifiers")
        return validator

    def test_resolver_accepts_a_legitimate_existing_relative_file(self):
        resolver = self._resolve_under_root()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "signals"
            target = root / "nested" / "record.csv"
            target.parent.mkdir(parents=True)
            target.write_text("ok", encoding="utf-8")
            self.assertEqual(
                resolver(root, "nested/record.csv", field_name="relative_path", must_exist=True),
                target.resolve(),
            )

    def test_resolver_rejects_parent_absolute_drive_and_unc_paths(self):
        resolver = self._resolve_under_root()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "signals"
            root.mkdir()
            candidates = (
                "../outside.csv",
                r"..\outside.csv",
                str(Path(temporary) / "outside.csv"),
                r"C:\outside.csv",
                r"C:outside.csv",
                r"\\server\share\outside.csv",
            )
            for candidate in candidates:
                with self.subTest(candidate=candidate):
                    with self.assertRaises(ReleasePathError):
                        resolver(root, candidate, field_name="relative_path", must_exist=False)

    def test_component_validator_accepts_pseudonymous_ids_and_rejects_path_syntax(self):
        validate = self._validate_component()
        for value in ("P01", "P_01", "R0_1", "left", "unknown"):
            with self.subTest(value=value):
                self.assertEqual(validate(value, field_name="participant_id"), value)
        for value in ("Sujeto-Ñ01", "Participante 01"):
            with self.subTest(value=value):
                self.assertEqual(validate(value, field_name="participant_id"), value)
        for value in ("", ".", "..", "../P01", r"..\P01", r"C:P01", "P/01", "CON"):
            with self.subTest(value=value):
                with self.assertRaises(ReleasePathError):
                    validate(value, field_name="participant_id")

    def test_resolver_rejects_a_symlink_that_escapes_the_root(self):
        resolver = self._resolve_under_root()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "signals"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            link = root / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                if os.name != "nt":
                    self.skipTest("symlink creation is unavailable in this environment")
                quote = lambda path: "'" + str(path).replace("'", "''") + "'"
                command = (
                    "New-Item -ItemType Junction -Path "
                    + quote(link)
                    + " -Target "
                    + quote(outside)
                    + " | Out-Null"
                )
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    self.skipTest(f"symlink or junction creation is unavailable: {result.stderr.strip()}")
            with self.assertRaises(ReleasePathError):
                resolver(root, "linked/new.csv", field_name="relative_path", must_exist=False)

    def test_proxy_label_generation_rejects_manifest_input_outside_signal_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "signals"
            root.mkdir()
            outside = base / "outside.csv"
            safe_inputs = []
            for participant in ("P01", "P02", "P03"):
                relative = "../outside.csv" if participant == "P01" else f"{participant}_R1_left.csv"
                target = outside if participant == "P01" else root / relative
                if target not in safe_inputs:
                    with target.open("w", newline="", encoding="utf-8") as handle:
                        handle.write("Relative_Time_s,ADC_Value,Voltage_mV\n")
                        for index in range(301):
                            handle.write(f"{index / 100:.2f},1,1\n")
                    safe_inputs.append(target)
            manifest = base / "manifest.csv"
            manifest.write_text(
                "participant_id,recording_id,leg,set_index,relative_path\n"
                "P01,R1,left,1,../outside.csv\n"
                "P02,R1,left,1,P02_R1_left.csv\n"
                "P03,R1,left,1,P03_R1_left.csv\n",
                encoding="utf-8",
            )
            output = base / "labels"
            with self.assertRaisesRegex(ReleasePathError, "relative_path"):
                generar_proxy_temporal.generate(manifest, root, output, seed=12, n_inner=2)
            self.assertFalse(output.exists(), "refused path traversal must not create label outputs")


if __name__ == "__main__":
    unittest.main()