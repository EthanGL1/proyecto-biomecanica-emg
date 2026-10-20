import base64
import gzip
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emg_dashboard import derive_candidate_activity, render_dashboard_html  # noqa: E402


class EmgDashboardTests(unittest.TestCase):
    def _example_rows(self):
        traces = [
            {
                "series": "R1", "run_index": 1, "sample_index": index,
                "time_s": index / 1000.0, "raw_adc": float(index),
                "control_adc": float(index), "hp20_adc": float(index) / 2,
                "hp40_adc": float(index) / 3, "bp20_400_adc": float(index) / 4,
                "fir_bp20_400_adc": float(index) / 5, "common_valid": index > 1,
            }
            for index in range(4)
        ]
        features = [
            {
                "series": "R1", "run_index": 1, "window_index": 1,
                "time_s": 0.5, "filter_branch": "hp20", "rms_adc": 2.0,
                "mdf_hz": 80.0, "mnf_hz": 90.0,
            }
        ]
        stft = [
            {
                "series": "R1", "run_index": 1, "window_index": 1,
                "time_s": 0.5, "filter_branch": "hp20", "frequency_hz": 80.0,
                "psd_adc2_per_hz": 0.25,
            }
        ]
        segments = [
            {"series": "R1", "run_index": 1, "start_time_s": 0.0, "end_time_s": 1.0, "status": "processed"}
        ]
        return traces, features, stft, segments

    def test_dashboard_is_standalone_and_discloses_exploratory_status(self):
        traces, features, stft, segments = self._example_rows()
        fft = [{
            "series": "R1", "run_index": 1, "window_index": 1,
            "time_s": 0.5, "filter_branch": "hp20", "frequency_hz": 80.0,
            "amplitude_adc": 1.0,
        }]
        psd = [{
            "series": "R1", "run_index": 1, "filter_branch": "hp20",
            "frequency_hz": 80.0, "psd_adc2_per_hz": 0.25,
        }]
        html = render_dashboard_html(traces, features, stft, segments, fft, psd)
        self.assertIn("<canvas", html)
        self.assertIn("score de fatiga validado", html.lower())
        self.assertIn("RMS (ADC)", html)
        self.assertIn("Frecuencia (Hz)", html)
        self.assertIn("MDF", html)
        self.assertIn("MNF", html)
        self.assertIn("STFT", html)
        self.assertIn("calidad fisiológica", html.lower())
        self.assertIn("FFT de ventana", html)
        self.assertIn("PSD promedio", html)
        self.assertIn("Actividad candidata", html)
        self.assertIn("No equivale a reposo ni repetición", html)
        self.assertNotIn("<script src=", html.lower())
        match = re.search(r'const DATA_GZIP_B64="([A-Za-z0-9+/=]+)";', html)
        self.assertIsNotNone(match)
        payload = json.loads(gzip.decompress(base64.b64decode(match.group(1))).decode("utf-8"))
        self.assertEqual(payload["fft"][0]["frequency_hz"], 80.0)
        self.assertEqual(payload["psd"][0]["psd_adc2_per_hz"], 0.25)
        self.assertNotIn("subject_id", json.dumps(payload))

    def test_candidate_activity_uses_filtered_rms_and_splits_on_gaps(self):
        traces = []
        fs = 1000.0
        segments = [(0, 200, 0.1), (200, 400, 10.0), (500, 700, 0.1), (700, 900, 10.0)]
        for run, (start, end, amplitude) in enumerate(segments, start=1):
            times = np.arange(end - start) / fs + (start / fs)
            signal = amplitude * np.sin(2 * np.pi * 80.0 * times)
            for index, (time_s, value) in enumerate(zip(times, signal)):
                traces.append({
                    "series": "R1", "run_index": run, "sample_index": index,
                    "time_s": float(time_s), "raw_adc": float(value),
                    "control_adc": float(value), "hp20_adc": float(value),
                    "hp40_adc": float(value), "bp20_400_adc": float(value),
                    "fir_bp20_400_adc": float(value), "common_valid": True,
                })

        frames, events = derive_candidate_activity(traces)

        self.assertTrue(frames)
        self.assertEqual({row["status"] for row in frames}, {"candidate_activity", "below_threshold"})
        self.assertGreaterEqual(len(events), 2)
        self.assertTrue(all(row["quality"] == "common_mask_valid" for row in events))
        self.assertTrue(all(row["duration_s"] > 0 for row in events))
        self.assertTrue(all(row["run_index"] in {1, 2, 3, 4} for row in events))

    def test_dashboard_refuses_empty_or_identity_bearing_trace_payload(self):
        _, features, stft, segments = self._example_rows()
        with self.assertRaisesRegex(ValueError, "trace rows"):
            render_dashboard_html([], features, stft, segments)
        traces, _, _, _ = self._example_rows()
        traces[0]["participant_id"] = "P01"
        with self.assertRaisesRegex(ValueError, "identity fields"):
            render_dashboard_html(traces, features, stft, segments)


if __name__ == "__main__":
    unittest.main()
