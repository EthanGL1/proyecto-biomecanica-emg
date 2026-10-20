import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CERQUEIRA_SRC = ROOT / "src" / "cerqueira"
sys.path.insert(0, str(CERQUEIRA_SRC))

import auditar_cerqueira_emg_label_alignment as alignment  # noqa: E402
import entrenar_cerqueira_loso as model_tools  # noqa: E402
import extraer_caracteristicas_cerqueira as features  # noqa: E402
from inferir_cerqueira_trial import InferenceError, run_inference  # noqa: E402


CHANNELS = [
    "R BICEPS BRACHII: EMG 1 [V]",
    "R DELTOID ANTERIOR: EMG 2 [V]",
    "R DELTOID MEDIUS: EMG 6 [V]",
    "R DELTOID POSTERIOR: EMG 7 [V]",
]


class CerqueiraPublicPipelineTests(unittest.TestCase):
    def _write_synthetic_fold(self, directory: Path) -> tuple[Path, Path]:
        columns = features._feature_columns()
        rng = np.random.default_rng(20261020)
        raw = rng.normal(size=(90, len(columns)))
        labels = np.tile(np.array([0, 1, 2], dtype=np.int64), 30)
        transformer = model_tools.fit_standardizer(raw)
        standardized = model_tools.transform_standardizer(raw, transformer)
        estimator = model_tools.fit_gaussian_nb(
            standardized, labels, np.ones(labels.size, dtype=np.float64)
        )
        model_path = directory / "heldout_subject_01.npz"
        model_tools._save_fold_model(model_path, "gaussian_nb", estimator, transformer, None)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        manifest_path = directory / "fold_models_manifest.json"
        manifest = {
            "feature_columns_ordered": columns,
            "feature_extraction": {
                "sampling_hz_median": 1000.0,
                "sampling_rate_relative_tolerance": 0.005,
                "window_seconds": 1.0,
                "window_overlap_fraction": 0.0,
                "frequency_band_hz": [20.0, 400.0],
                "fir_taps": 251,
            },
            "fold_models": [
                {"model_file": model_path.name, "model_sha256": digest},
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return model_path, manifest_path

    def _write_synthetic_trial(self, path: Path, *, samples: int = 5000) -> Path:
        sampling_hz = 1000.0
        times = np.arange(samples, dtype=np.float64) / sampling_hz
        header = ["x [s]", "x [s].1", "x [s].2", "x [s].3", *CHANNELS]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for t in times:
                writer.writerow([
                    f"{t:.9f}", f"{t:.9f}", f"{t:.9f}", f"{t:.9f}",
                    f"{np.sin(2*np.pi*55*t):.12g}",
                    f"{0.8*np.sin(2*np.pi*85*t):.12g}",
                    f"{0.6*np.sin(2*np.pi*125*t):.12g}",
                    f"{0.7*np.sin(2*np.pi*175*t):.12g}",
                ])
        return path

    def _synthetic_nested_archives(self, root: Path) -> tuple[Path, Path, str]:
        emg_path = root / "synthetic_emg.zip"
        label_path = root / "synthetic_labels.zip"
        label_text = "time,label\n" + "".join(
            f"{index / 50:.2f},{min(index // 50, 2)}\n" for index in range(150)
        )
        trial_text = ""
        for subject in range(1, 5):
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer)
            writer.writerow(["x [s]", "x [s].1", "x [s].2", "x [s].3", *CHANNELS])
            for index in range(3000):
                time_s = index / 1000.0
                values = [
                    0.001 * np.sin(2 * np.pi * (45 + subject * 2 + channel * 25) * time_s)
                    + 0.0001 * np.cos(2 * np.pi * (11 + channel) * time_s + subject)
                    for channel in range(4)
                ]
                writer.writerow([f"{time_s:.3f}"] * 4 + [f"{value:.12g}" for value in values])
            trial_text = buffer.getvalue()
            emg_nested = io.BytesIO()
            label_nested = io.BytesIO()
            with zipfile.ZipFile(emg_nested, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(f"subject_{subject}/trial_1.csv", trial_text)
            with zipfile.ZipFile(label_nested, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(f"subject_{subject}/trial_1.csv", label_text)
            if subject == 1:
                outer_emg = zipfile.ZipFile(emg_path, "w", compression=zipfile.ZIP_DEFLATED)
                outer_labels = zipfile.ZipFile(label_path, "w", compression=zipfile.ZIP_DEFLATED)
            outer_emg.writestr(f"subject_{subject}.zip", emg_nested.getvalue())
            outer_labels.writestr(f"subject_{subject}.zip", label_nested.getvalue())
        outer_emg.close()
        outer_labels.close()
        return emg_path, label_path, trial_text

    def test_fir_contract_and_label_coverage_are_explicit(self):
        self.assertTrue(alignment._TIME_COLUMN_RE.fullmatch("x [s].3"))
        coefficients = features.design_fir_bandpass(1000.0, 20.0, 400.0, 251)
        np.testing.assert_allclose(coefficients, coefficients[::-1], rtol=0.0, atol=1e-14)
        times = np.arange(50, dtype=np.float64) / 50.0
        labels = np.array([0] * 40 + [1] * 10, dtype=np.int8)
        label, coverage, majority = features.assign_window_label(times, labels, 0.0, 1.0)
        self.assertEqual(label, 0)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(majority, 0.8)
        mixed, _, _ = features.assign_window_label(times, labels, 0.0, 1.0, min_class_fraction=0.81)
        self.assertIsNone(mixed)

    def test_audit_writer_rejects_collisions_and_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            same_path = root / "same.json"
            with self.assertRaises(alignment.AlignmentAuditError):
                alignment.write_report({"trials": []}, same_path, same_path)

            existing = root / "existing.json"
            existing.write_text("keep this receipt", encoding="utf-8")
            with self.assertRaises(alignment.AlignmentAuditError):
                alignment.write_report({"trials": []}, existing, root / "new.csv")
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep this receipt")

            source = root / "source.zip"
            source.write_bytes(b"immutable input")
            with self.assertRaises(alignment.AlignmentAuditError):
                alignment.write_report(
                    {"trials": []}, source, root / "trial.csv", protected_paths=[source]
                )
            self.assertEqual(source.read_bytes(), b"immutable input")

    def test_audit_csv_neutralizes_formula_like_archive_member_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            emg_path = root / "emg.zip"
            label_path = root / "labels.zip"
            member = '=HYPERLINK("https://example.invalid","x")/subject_1/trial_1.csv'
            signal_csv = (
                "x [s],x [s].1,x [s].2,x [s].3,c1,c2,c3,c4\n"
                "0,0,0,0,1,2,3,4\n"
                "0.01,0.01,0.01,0.01,2,3,4,5\n"
            )
            label_csv = "time,label\n0,0\n0.01,1\n"
            nested_emg = io.BytesIO()
            nested_labels = io.BytesIO()
            with zipfile.ZipFile(nested_emg, "w") as archive:
                archive.writestr(member, signal_csv)
            with zipfile.ZipFile(nested_labels, "w") as archive:
                archive.writestr(member, label_csv)
            with zipfile.ZipFile(emg_path, "w") as archive:
                archive.writestr("subject_1.zip", nested_emg.getvalue())
            with zipfile.ZipFile(label_path, "w") as archive:
                archive.writestr("subject_1.zip", nested_labels.getvalue())

            report = alignment.audit_alignment(
                emg_path,
                label_path,
                expected_emg_md5=None,
                expected_label_md5=None,
            )
            csv_path = root / "audit.csv"
            alignment.write_report(
                report,
                root / "audit.json",
                csv_path,
                protected_paths=[emg_path, label_path],
            )
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))

            self.assertEqual(len(report["trials"]), 1)
            self.assertEqual(row["signal_path_in_archive"], "'" + member)
            self.assertEqual(row["label_path_in_archive"], "'" + member)
            json_report = json.loads((root / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(json_report["trials"][0]["signal_path_in_archive"], member)

    def test_spreadsheet_safe_cell_covers_formula_prefixes_and_keeps_normal_paths(self):
        unsafe_values = [
            "=1+1",
            "+1+1",
            "-1+1",
            "@SUM(A1:A2)",
            " \t=1+1",
            "\ufeff=1+1",
            "\n=1+1",
        ]
        for value in unsafe_values:
            with self.subTest(value=value):
                self.assertEqual(alignment._spreadsheet_safe_cell(value), "'" + value)

        ordinary_path = "subject_1/trial_1.csv"
        self.assertEqual(alignment._spreadsheet_safe_cell(ordinary_path), ordinary_path)

    def test_feature_writer_rejects_same_resolved_output_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "features.csv"
            with self.assertRaises(features.FeatureExtractionError):
                features.write_feature_outputs([], {}, target, target.parent / "." / target.name)
            self.assertFalse(target.exists())

            source = Path(temporary) / "alignment.json"
            source.write_text("immutable input", encoding="utf-8")
            with self.assertRaises(features.FeatureExtractionError):
                features.write_feature_outputs([], {}, source, Path(temporary) / "features.json",
                                               protected_paths=[source])
            self.assertEqual(source.read_text(encoding="utf-8"), "immutable input")

    def test_all_private_pipeline_outputs_are_rejected_inside_public_checkout(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output_root = Path(temporary) / "private"
            with self.assertRaisesRegex(alignment.AlignmentAuditError, "public checkout"):
                alignment.write_report(
                    {"trials": []}, output_root / "alignment.json", output_root / "trials.csv"
                )
            with self.assertRaisesRegex(features.FeatureExtractionError, "public checkout"):
                features.write_feature_outputs(
                    [], {}, output_root / "features.csv", output_root / "summary.json"
                )
            with self.assertRaisesRegex(model_tools.CerqueiraModelError, "public checkout"):
                model_tools.evaluate_loso(
                    output_root / "missing.csv", output_root / "missing.json", output_root / "loso"
                )

            with tempfile.TemporaryDirectory() as external_temp:
                external_root = Path(external_temp)
                model_path, manifest_path = self._write_synthetic_fold(external_root)
                input_path = self._write_synthetic_trial(external_root / "synthetic_trial.csv")
                with self.assertRaisesRegex(InferenceError, "public checkout"):
                    run_inference(
                        [input_path], model_path, manifest_path,
                        output_root / "scores.csv", output_root / "summary.json",
                    )
            self.assertFalse(output_root.exists())

    def test_model_loader_checks_manifest_hash_and_real_csv_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, manifest_path = self._write_synthetic_fold(root)
            input_path = self._write_synthetic_trial(root / "synthetic_trial.csv")
            output_path = root / "scores.csv"
            summary_path = root / "summary.json"
            summary = run_inference(
                [input_path], model_path, manifest_path, output_path, summary_path
            )
            self.assertEqual(summary["samples_processed"], 5000)
            self.assertEqual(summary["complete_windows_scored"], 5)
            predicted_counts = summary["predicted_label_counts"]
            self.assertEqual(sum(predicted_counts.values()), 5)
            self.assertTrue(set(predicted_counts).issubset({"0", "1", "2"}))
            scores = np.genfromtxt(output_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            self.assertEqual(len(scores), 5)
            for name in ("p0", "p1", "p2"):
                self.assertIn(name, scores.dtype.names)
                self.assertTrue(np.isfinite(scores[name]).all())
            probabilities = np.column_stack([scores["p0"], scores["p1"], scores["p2"]])
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)

            model_path.write_bytes(model_path.read_bytes() + b"tamper")
            with self.assertRaises(model_tools.CerqueiraModelError):
                model_tools.load_fold_model_bundle(model_path, manifest_path)

    def test_inference_rejects_output_aliases_and_creates_both_parents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path, manifest_path = self._write_synthetic_fold(root)
            input_path = self._write_synthetic_trial(root / "synthetic_trial.csv")
            same_output = root / "same-output.json"
            with self.assertRaises(InferenceError):
                run_inference([input_path], model_path, manifest_path, same_output, same_output)
            with self.assertRaises(InferenceError):
                run_inference([input_path], model_path, manifest_path, input_path, root / "summary.json")

            nested_csv = root / "predictions" / "scores.csv"
            nested_json = root / "receipts" / "summary.json"
            run_inference([input_path], model_path, manifest_path, nested_csv, nested_json)
            self.assertTrue(nested_csv.is_file())
            self.assertTrue(nested_json.is_file())

    def test_synthetic_archives_run_alignment_features_grouped_loso_and_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            emg_path, label_path, trial_text = self._synthetic_nested_archives(root)
            report = alignment.audit_alignment(emg_path, label_path)
            self.assertEqual(report["summary"]["matched_subject_count"], 4)
            self.assertEqual(report["summary"]["matched_trial_count"], 4)
            self.assertTrue(report["summary"]["pairing_complete"])
            audit_json = root / "audit" / "alignment.json"
            audit_csv = root / "audit" / "alignment.csv"
            alignment.write_report(
                report, audit_json, audit_csv, protected_paths=[emg_path, label_path]
            )

            rows, feature_summary = features.extract_feature_table(
                emg_path, label_path, audit_json,
                trial_coverage_threshold=1.0, min_label_coverage=0.80,
                min_class_fraction=0.80, fir_taps=51,
            )
            self.assertEqual(feature_summary["matched_subject_count"], 4)
            self.assertEqual(feature_summary["window_count"], 12)
            self.assertEqual({row["label"] for row in rows}, {0, 1, 2})
            feature_csv = root / "features" / "features.csv"
            feature_json = root / "features" / "summary.json"
            features.write_feature_outputs(
                rows, feature_summary, feature_csv, feature_json,
                protected_paths=[emg_path, label_path, audit_json],
            )

            loso_dir = root / "loso"
            model_tools.evaluate_loso(
                feature_csv, feature_json, loso_dir, regularizations=(0.1,),
                inner_splits=2, max_iter=1000,
            )
            model_path = loso_dir / "fold_models" / "heldout_subject_04.npz"
            manifest_path = loso_dir / "fold_models_manifest.json"
            trial_path = root / "heldout_trial.csv"
            trial_path.write_text(trial_text, encoding="utf-8", newline="")
            result_csv = root / "inference" / "scores.csv"
            result_json = root / "inference" / "summary.json"
            inference_summary = run_inference(
                [trial_path], model_path, manifest_path, result_csv, result_json
            )
            self.assertEqual(inference_summary["complete_windows_scored"], 3)
            self.assertTrue(result_csv.is_file())
            self.assertTrue(result_json.is_file())

    def test_fragment_assembly_refuses_a_time_gap(self):
        names = list(CHANNELS)
        first_times = np.arange(100, dtype=np.float64) / 1000.0
        second_times = np.arange(100, 200, dtype=np.float64) / 1000.0
        zeros_first = np.zeros((100, 4), dtype=np.float64)
        zeros_second = np.zeros((100, 4), dtype=np.float64)
        assembled = __import__("inferir_cerqueira_trial").assemble_fragments([
            (first_times, zeros_first, names), (second_times, zeros_second, names),
        ])
        self.assertEqual(assembled[3]["fragment_count"], 2)
        gapped = second_times + 0.01
        with self.assertRaises(InferenceError):
            __import__("inferir_cerqueira_trial").assemble_fragments([
                (first_times, zeros_first, names), (gapped, zeros_second, names),
            ])


if __name__ == "__main__":
    unittest.main()
