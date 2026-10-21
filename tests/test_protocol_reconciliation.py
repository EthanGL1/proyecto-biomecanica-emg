import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProtocolReconciliationTests(unittest.TestCase):
    def test_public_report_distinguishes_intended_protocol_from_record_evidence(self):
        report = (ROOT / "reports" / "F53_PROTOCOL_RECONCILIATION.md").read_text(
            encoding="utf-8"
        ).lower()

        for required in (
            "15 participantes",
            "138 formales",
            "rectus femoris",
            "extensión unilateral de rodilla",
            "70 % de 1rm",
            "protocolo previsto",
            "no se verifican",
            "voltage_mv",
            "señal guardada",
        ):
            with self.subTest(required=required):
                self.assertIn(required, report)

    def test_readme_does_not_deny_the_project_specific_muscle_target(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        self.assertNotIn("no impacto de un grupo muscular específico", readme)
        self.assertIn("recto femoral (rectus femoris)", readme)
        self.assertIn("f53_protocol_reconciliation.md", readme)


if __name__ == "__main__":
    unittest.main()
