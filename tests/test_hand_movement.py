"""Verify independent wrist motion, body compensation and missing-data resets."""

import csv
import io
import unittest

import numpy as np

from behaviour_common import TemporalGate
from body_gestures import BODY_LABELS, BodyGestureDetector, BodyRules
from main import EventWriter


def upper_body():
    pose = np.full((33, 5), np.nan)
    for index, xy in {11: (0.6, 0.3), 12: (0.4, 0.3),
                      15: (0.7, 0.6), 16: (0.3, 0.6)}.items():
        pose[index] = [*xy, 0, 0.99, 0.99]
    return pose


class HandMovementTests(unittest.TestCase):
    def test_left_right_and_both_work_without_hips_elbows_or_hand_model(self):
        for moving in (("left",), ("right",), ("left", "right")):
            with self.subTest(moving=moving):
                detector = BodyGestureDetector(1280, 720)
                for index in range(11):
                    pose = upper_body()
                    for side, wrist in (("left", 15), ("right", 16)):
                        if side in moving:
                            pose[wrist, 0] += index * 0.008
                    states, _ = detector.update(pose, None, index / 10)
                for side in ("left", "right"):
                    self.assertIs(states[f"{side}_hand_movement"], side in moving)
                    self.assertIn(f"{side}_hand_movement", BODY_LABELS)
                self.assertIsNone(states["stationary"])

    def test_small_jitter_and_common_translation_scaling_stay_negative(self):
        for mode in ("jitter", "translate_and_scale"):
            with self.subTest(mode=mode):
                detector = BodyGestureDetector(1280, 720)
                for index in range(11):
                    pose = upper_body()
                    if mode == "jitter":
                        pose[[15, 16], 0] += 0.001 * np.sin(index)
                    else:
                        pose[:, :2] = (pose[:, :2] - 0.5) * (1 + 0.01 * index) + 0.5
                        pose[:, 0] += index * 0.005
                    states, info = detector.update(pose, None, index / 10)
                for side in ("left", "right"):
                    self.assertFalse(states[f"{side}_hand_movement"])
                    self.assertLess(info[f"{side}_hand_movement_range_shoulder_widths"], 0.12)

    def test_unknown_wrist_resets_only_its_own_side_and_gap_resets_both(self):
        detector = BodyGestureDetector(1280, 720)
        for index in range(6):
            pose = upper_body()
            pose[[15, 16], 0] += index * 0.008
            states, _ = detector.update(pose, None, index / 10)
        self.assertTrue(states["left_hand_movement"])
        self.assertTrue(states["right_hand_movement"])
        pose[[15, 16], 0] += 0.008
        pose[15, 3] = 0.1
        states, _ = detector.update(pose, None, 0.6)
        self.assertIsNone(states["left_hand_movement"])
        self.assertTrue(states["right_hand_movement"])
        pose[15, 3] = 0.99
        states, _ = detector.update(pose, None, 0.7)
        self.assertIsNone(states["left_hand_movement"])
        self.assertTrue(states["right_hand_movement"])
        states, _ = detector.update(pose, None, 1.0)
        self.assertIsNone(states["left_hand_movement"])
        self.assertIsNone(states["right_hand_movement"])

    def test_missing_shoulders_reset_both_and_initial_window_is_unknown(self):
        detector = BodyGestureDetector(1280, 720)
        for index in range(6):
            pose = upper_body()
            pose[[15, 16], 0] += index * 0.008
            states, _ = detector.update(pose, None, index / 10)
            if index < 3:
                self.assertIsNone(states["left_hand_movement"])
        pose[12] = np.nan
        states, _ = detector.update(pose, None, 0.6)
        self.assertIsNone(states["left_hand_movement"])
        self.assertIsNone(states["right_hand_movement"])
        states, _ = detector.update(upper_body(), None, 0.7)
        self.assertIsNone(states["left_hand_movement"])
        self.assertIsNone(states["right_hand_movement"])

    def test_out_and_back_counts_and_stopping_clears_rolling_movement(self):
        detector = BodyGestureDetector(1000, 1000)
        offsets = [0, 0.015, 0.030, 0.015, 0]
        for index, offset in enumerate(offsets):
            pose = upper_body()
            pose[15, 0] += offset
            states, info = detector.update(pose, None, index / 10)
        self.assertTrue(states["left_hand_movement"])
        self.assertAlmostEqual(info["left_hand_movement_range_shoulder_widths"], 0.15)
        for index in range(5, 11):
            states, _ = detector.update(upper_body(), None, index / 10)
        self.assertFalse(states["left_hand_movement"])

    def test_confirmed_movement_exports_an_episode_with_current_event_writer(self):
        detector, gate = BodyGestureDetector(1000, 1000), TemporalGate(0.25)
        output = io.StringIO()
        events = EventWriter(csv.writer(output), 10)
        for index in range(16):
            pose = upper_body()
            pose[15, 0] += index * 0.008
            raw, _ = detector.update(pose, None, index / 10)
            states = gate.update(raw, index / 10)
            events.update(states, index)
        self.assertTrue(states["left_hand_movement"])
        events.close(16)
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        movements = [row for row in rows if row["behaviour"] == "left_hand_movement"]
        self.assertEqual(len(movements), 1)
        self.assertEqual(movements[0]["end_frame_exclusive"], "16")
        self.assertFalse(any(row["behaviour"] == "right_hand_movement" for row in rows))
        with self.assertRaises(ValueError):
            BodyRules(hand_motion_window_s=0.2, hand_motion_min_s=0.3)


if __name__ == "__main__":
    unittest.main()
