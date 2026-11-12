"""Check source-time duration accounting independently of recognition models."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from window_summary import main, write_window_summary


class WindowSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frames = self.root / "video_frames.csv"
        self.events = self.root / "video_events.csv"
        self.output = self.root / "video_windows.csv"

    def fixture(self, duration, fps=1, labels=("smiling",), states=None, events=()):
        with self.frames.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame_index", "timestamp_s",
                *["raw_" + label for label in labels], *labels])
            writer.writeheader()
            for i in range(math.ceil(duration * fps)):
                row = {"frame_index": i, "timestamp_s": f"{i/fps:.6f}"}
                for label in labels:
                    row[label] = row["raw_" + label] = states(i, label) if states else "0"
                writer.writerow(row)
        with self.events.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["behaviour", "start_s", "end_s_exclusive",
                                                        "timing_type", "completion_s"])
            writer.writeheader()
            writer.writerows(events)

    def summarize(self, duration, fps=1, window=30):
        report = write_window_summary(self.frames, self.events, self.output, fps, duration, window)
        with self.output.open() as handle:
            rows = list(csv.DictReader(handle))
        # Each behaviour independently partitions a window into active/inactive/unknown.
        for row in rows:
            known = float(row["observed_seconds"])
            unknown = float(row["unknown_seconds"])
            self.assertAlmostEqual(known + unknown, float(row["window_duration_s"]), places=5)
            if known:
                self.assertAlmostEqual(float(row["active_seconds"]) + float(row["inactive_seconds"]), known, places=5)
                self.assertLessEqual(float(row["active_percent_observed"]), 100)
        return report, rows

    def test_boundary_splits_duration_without_double_counting_episode(self):
        self.fixture(65, events=[{"behaviour": "smiling", "start_s": 29, "end_s_exclusive": 32}])
        report, rows = self.summarize(65)
        self.assertEqual(report["window_count"], 3)
        self.assertEqual([float(r["active_seconds"]) for r in rows], [1, 2, 0])
        self.assertEqual([r["event_count"] for r in rows], ["1", "0", "0"])
        self.assertEqual([r["events_overlapping"] for r in rows], ["1", "1", "0"])
        self.assertEqual([float(r["window_duration_s"]) for r in rows], [30, 30, 5])

    def test_adjacent_and_overlapping_native_categories_are_one_episode(self):
        self.fixture(30, events=[{"behaviour": "smiling", "start_s": a, "end_s_exclusive": b}
                                for a, b in [(3, 5), (1, 3), (2, 4), (1, 3)]])
        _, rows = self.summarize(30)
        self.assertEqual(float(rows[0]["active_seconds"]), 4)
        self.assertEqual(rows[0]["event_count"], "1")

    def test_unknown_coverage_is_excluded_and_disabled_is_not_false(self):
        self.fixture(60, labels=("smiling", "holding_teaching_aid"),
                     states=lambda i, label: "" if label == "holding_teaching_aid" or 10 <= i < 40 else "1",
                     events=[{"behaviour": "smiling", "start_s": 0, "end_s_exclusive": 60}])
        _, rows = self.summarize(60)
        smile = [r for r in rows if r["behaviour"] == "smiling"]
        self.assertEqual([float(r["active_seconds"]) for r in smile], [10, 20])
        self.assertEqual([float(r["unknown_seconds"]) for r in smile], [20, 10])
        self.assertTrue(all(r["event_count"] == "1" for r in smile))
        disabled = [r for r in rows if r["behaviour"] == "holding_teaching_aid"]
        for row in disabled:
            self.assertEqual(row["active_seconds"], "")
            self.assertEqual(row["event_count"], "")
            self.assertEqual(float(row["unknown_seconds"]), 30)

    def test_completed_cycles_use_motion_span_and_count_on_completion(self):
        events = [{"behaviour": "head_nodding", "start_s": a, "end_s_exclusive": b,
                   "completion_s": b, "timing_type": "completed_cycle"}
                  for a, b in [(28, 30), (30, 32), (28, 30)]]
        self.fixture(60, labels=("head_nodding", "head_shaking"), events=events)
        _, rows = self.summarize(60)
        nods = [r for r in rows if r["behaviour"] == "head_nodding"]
        self.assertEqual([float(r["active_seconds"]) for r in nods], [2, 2])
        self.assertEqual([r["event_count"] for r in nods], ["0", "2"])
        self.assertTrue(all(r["count_basis"] == "cycle_completion" for r in rows))

    def test_cycle_crossing_window_is_counted_once(self):
        self.fixture(60, labels=("head_shaking",), events=[{"behaviour": "head_shaking",
            "start_s": 29, "end_s_exclusive": 31, "completion_s": 31, "timing_type": "completed_cycle"}])
        _, rows = self.summarize(60)
        self.assertEqual([float(r["active_seconds"]) for r in rows], [1, 1])
        self.assertEqual([r["event_count"] for r in rows], ["0", "1"])
        self.assertEqual([r["events_overlapping"] for r in rows], ["1", "1"])

    def test_future_completion_from_longer_native_context_is_excluded(self):
        self.fixture(30, labels=("head_nodding",), events=[{"behaviour": "head_nodding",
            "start_s": 29, "end_s_exclusive": 30, "completion_s": 31, "timing_type": "completed_cycle"}])
        _, rows = self.summarize(30)
        self.assertEqual(float(rows[0]["active_seconds"]), 0)
        self.assertEqual(rows[0]["event_count"], "0")

    def test_brief_native_blink_retains_subframe_duration(self):
        self.fixture(1, fps=10, labels=("blinking",), events=[{"behaviour": "blinking",
            "start_s": .12, "end_s_exclusive": .16, "timing_type": "hmad_interval"}])
        _, rows = self.summarize(1, fps=10)
        self.assertAlmostEqual(float(rows[0]["active_seconds"]), .04)

    def test_fractional_fps_selected_prefix_has_exactly_four_windows(self):
        fps = 30000 / 1001
        self.fixture(120, fps=fps, events=[{"behaviour": "smiling", "start_s": 0,
            "end_s_exclusive": math.ceil(120*fps)/fps}])
        report, rows = self.summarize(120, fps=fps)
        self.assertEqual(report["window_count"], 4)
        self.assertTrue(all(float(r["active_seconds"]) == 30 for r in rows))
        self.assertEqual(float(rows[-1]["window_end_s_exclusive"]), 120)

    def test_regenerate_from_metadata_without_changing_evidence(self):
        self.fixture(31)
        original = self.frames.read_bytes(), self.events.read_bytes()
        metadata = self.root / "video_metadata.json"
        metadata.write_text(json.dumps({"fps": 1, "processed_frames": 31, "duration_s": 31,
            "outputs": {"frames": str(self.frames), "events": str(self.events)}}))
        original_metadata = metadata.read_bytes()
        report = main(["--metadata", str(metadata), "--duration-seconds", "30"])
        self.assertEqual(report["window_count"], 1)
        self.assertEqual(original, (self.frames.read_bytes(), self.events.read_bytes()))
        self.assertEqual(metadata.read_bytes(), original_metadata)

    def test_rejects_invalid_window_length_and_incomplete_frames(self):
        self.fixture(30)
        for window in (0, -1, math.nan, math.inf):
            with self.subTest(window=window), self.assertRaises(ValueError):
                self.summarize(30, window=window)
        with self.assertRaises(ValueError):
            self.summarize(31)
        with self.assertRaises(ValueError):
            write_window_summary(self.frames, self.events, self.frames, 1, 30)

    def test_rejects_nonconsecutive_frame_times(self):
        self.fixture(30)
        self.frames.write_text(self.frames.read_text().replace("5,5.000000", "5,5.500000"))
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.summarize(30)


if __name__ == "__main__":
    unittest.main()
