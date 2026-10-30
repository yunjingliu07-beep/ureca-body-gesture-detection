import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from detect_gestures import GESTURES, analyze, classify_frame, direction, event_rows


def body():
    pose = np.full((33, 5), np.nan)
    for index, xy in {11: (0.65, 0.35), 12: (0.35, 0.35), 13: (0.8, 0.35),
                      14: (0.2, 0.35), 15: (0.95, 0.35), 16: (0.05, 0.35)}.items():
        pose[index] = [*xy, 0, 0.99, 0.99]
    return pose


def pointing_hand():
    hand = np.tile([0.4, 0.7, 0], (21, 1))
    hand[0] = [0.5, 0.8, 0]
    hand[5:9] = [[0.5, y, 0] for y in (0.6, 0.45, 0.32, 0.2)]
    for offset in (9, 13, 17):
        hand[offset:offset + 4] = [[0.6, 0.65, 0], [0.62, 0.5, 0],
                                 [0.63, 0.6, 0], [0.61, 0.72, 0]]
    return hand


class GestureTests(unittest.TestCase):
    def test_straight_arms_and_vertical_direction(self):
        states, directions = classify_frame(body(), True, 1920, 1080)
        self.assertIs(states["left_arm_extended"], True)
        self.assertIs(states["right_arm_extended"], True)
        self.assertIs(states["left_hand_raised"], False)
        self.assertEqual(directions["left_arm_extended"], "image_right")
        self.assertEqual(directions["right_arm_extended"], "image_left")
        self.assertEqual(direction(np.array([0, -10])), "image_up")
        self.assertEqual(direction(np.array([0, 10])), "image_down")

    def test_raised_hand_does_not_require_visible_elbow(self):
        pose = body()
        pose[15, :2] = [0.65, 0.1]
        pose[13] = np.nan
        states, _ = classify_frame(pose, True, 1920, 1080)
        self.assertIs(states["left_hand_raised"], True)
        self.assertIsNone(states["left_arm_extended"])
        self.assertIsNone(states["arms_crossed"])

    def test_crossed_arms_and_occlusion(self):
        pose = body()
        for index, xy in {11: (0.65, 0.3), 12: (0.35, 0.3), 13: (0.68, 0.6),
                          14: (0.32, 0.6), 15: (0.35, 0.55), 16: (0.65, 0.55)}.items():
            pose[index, :2] = xy
        states, _ = classify_frame(pose, True, 1920, 1080)
        self.assertIs(states["arms_crossed"], True)
        pose[16, 3] = 0.1
        states, _ = classify_frame(pose, True, 1920, 1080)
        self.assertIsNone(states["arms_crossed"])

    def test_finger_pointing_without_body_and_ambiguous_side(self):
        hands = np.stack([pointing_hand(), pointing_hand()])
        kwargs = dict(hands=hands, hand_detected=np.array([True, False]),
                      handedness=np.array(["Left", "Left"]), handedness_scores=np.array([0.95, 0.95]))
        states, directions = classify_frame(body(), False, 1920, 1080, **kwargs)
        self.assertIs(states["left_hand_pointing"], True)
        self.assertEqual(directions["left_hand_pointing"], "image_up")
        self.assertIsNone(states["left_hand_raised"])
        kwargs["hand_detected"][1] = True
        states, _ = classify_frame(body(), False, 1920, 1080, **kwargs)
        self.assertIsNone(states["left_hand_pointing"])

    def test_open_hand_is_not_pointing(self):
        hand = pointing_hand()
        for offset in (9, 13, 17):
            hand[offset:offset + 4] = [[0.6, y, 0] for y in (0.6, 0.45, 0.3, 0.15)]
        states, _ = classify_frame(body(), False, 1920, 1080,
                                  hands=np.stack([hand, hand]), hand_detected=[True, False],
                                  handedness=["Right", ""], handedness_scores=[0.95, np.nan])
        self.assertIs(states["right_hand_pointing"], False)
        self.assertIsNone(states["left_hand_pointing"])

    def test_unknown_breaks_events_and_duration_filters_short_episodes(self):
        states = [dict.fromkeys(GESTURES) for _ in range(6)]
        for index, value in enumerate([True, True, True, None, True, True]):
            states[index]["left_hand_raised"] = value
        events = event_rows(np.arange(6) / 10, states, [{} for _ in states], 0.1, 0.25)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["duration_s"], 0.3)
        self.assertAlmostEqual(events[0]["end_s_exclusive"], 0.3)
        self.assertEqual(events[0]["end_frame_exclusive"], 3)

    def test_saved_archive_exports_unknowns_and_event_measurements(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "session_arrays.npz"
            np.savez_compressed(path, normalized_landmarks=np.stack([body()] * 12),
                                pose_detected=np.ones(12, dtype=bool), timestamps_s=np.arange(12) / 30)
            (root / "session_metadata.json").write_text(json.dumps({"width": 1920, "height": 1080, "fps": 30}))
            summary = analyze(path, root / "results")
            self.assertEqual(summary["gestures"]["left_arm_extended"]["event_count"], 1)
            self.assertAlmostEqual(summary["gestures"]["left_arm_extended"]["event_duration_s"], 0.4)
            self.assertEqual(summary["gestures"]["left_hand_pointing"]["observable_duration_s"], 0)
            with Path(summary["outputs"]["frames"]).open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0]["left_arm_extended"], "1")
            self.assertEqual(rows[0]["left_hand_pointing"], "")


if __name__ == "__main__":
    unittest.main()
