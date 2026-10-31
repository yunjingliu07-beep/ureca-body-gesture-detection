"""Check rear-camera facing defaults and face-relative-to-neck turning rules."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from behaviour_common import TemporalGate
from body_gestures import BodyGestureDetector, classify_head_turn
from facial_expressions import FacialExpressionExtractor
from main import load_settings


def speaker(face_x=0.5, yaw=0):
    pose = np.full((33, 5), np.nan)
    pose[:11] = [[x, 0.15, 0, 0.99, 0.99] for x in np.linspace(face_x - 0.02, face_x + 0.02, 11)]
    for index, xy in {11: (0.6, 0.3), 12: (0.4, 0.3), 23: (0.55, 0.6), 24: (0.45, 0.6)}.items():
        pose[index] = [*xy, 0, 0.99, 0.99]
    world = pose.copy()
    theta = np.radians(yaw)
    across = 0.15 * np.array([np.cos(theta), 0, np.sin(theta)])
    world[11, :3], world[12, :3] = across, -across
    world[23, :3], world[24, :3] = across + [0, 0.4, 0], -across + [0, 0.4, 0]
    return pose, world


class FacingTurningTests(unittest.TestCase):
    def test_rear_camera_defaults_camera_is_class_and_away_is_board(self):
        for yaw, facing_class, facing_board in ((0, True, False), (180, False, True), (90, False, False)):
            with self.subTest(yaw=yaw):
                states, _ = BodyGestureDetector(1000, 1000).update(*speaker(yaw=yaw), 0)
                self.assertIs(states["facing_class"], facing_class)
                self.assertIs(states["facing_board"], facing_board)
        _, _, scene, _ = load_settings()
        self.assertEqual(scene, {"board_yaw_deg": 180.0, "class_yaw_deg": 0.0})
        scene = {"board_yaw_deg": None, "class_yaw_deg": None}
        states, _ = BodyGestureDetector(1000, 1000, scene=scene).update(*speaker(), 0)
        self.assertIsNone(states["facing_class"])
        self.assertIsNone(states["facing_board"])

    def test_face_left_center_right_of_neck_and_deadband_boundaries(self):
        for x, left, right in ((0.45, True, False), (0.48, False, False),
                                (0.5, False, False), (0.52, False, False), (0.55, False, True)):
            with self.subTest(face_x=x):
                pose, _ = speaker(face_x=x)
                states, info = classify_head_turn(pose, 1920, 1080)
                self.assertIs(states["turning_left"], left)
                self.assertIs(states["turning_right"], right)
                self.assertIs(states["turning"], left or right)
                self.assertEqual(info["face_skeleton_source"], "pose_face")
                self.assertEqual(info["neck_xy_px"], [960, 324])

    def test_current_mesh_takes_precedence_and_bad_mesh_falls_back(self):
        pose, _ = speaker(face_x=0.45)
        mesh = np.tile([0.55, 0.15, 0], (478, 1))
        states, info = classify_head_turn(pose, 1000, 1000, face_landmarks=mesh)
        self.assertTrue(states["turning_right"])
        self.assertEqual(info["face_skeleton_source"], "face_mesh")
        mesh[:] = np.nan
        states, info = classify_head_turn(pose, 1000, 1000, face_landmarks=mesh)
        self.assertTrue(states["turning_left"])
        self.assertEqual(info["face_skeleton_source"], "pose_face")

    def test_missing_shoulders_or_faces_give_unknown_and_hips_are_not_required(self):
        pose, _ = speaker(face_x=0.45)
        pose[[23, 24]] = np.nan
        states, _ = BodyGestureDetector(1000, 1000).update(pose, None, 0)
        self.assertTrue(states["turning_left"])
        pose[:11] = np.nan
        states, _ = classify_head_turn(pose, 1000, 1000)
        self.assertTrue(all(value is None for value in states.values()))
        pose, _ = speaker(face_x=0.45)
        pose[12, 3] = 0.1
        states, _ = classify_head_turn(pose, 1000, 1000)
        self.assertTrue(all(value is None for value in states.values()))

    def test_head_side_needs_confirmation_and_can_coexist_with_class_facing(self):
        detector, gate = BodyGestureDetector(1000, 1000), TemporalGate(0.25)
        for t in (0, 0.1, 0.2, 0.3):
            raw, _ = detector.update(*speaker(face_x=0.45), t)
            states = gate.update(raw, t)
            if t < 0.25:
                self.assertFalse(states["turning_left"])
        self.assertTrue(states["turning_left"])
        self.assertTrue(states["turning"])
        self.assertTrue(states["facing_class"])
        raw, _ = detector.update(*speaker(), 0.4)
        states = gate.update(raw, 0.4)
        self.assertFalse(states["turning"])
        self.assertFalse(states["turning_left"])

    def test_face_extractor_does_not_forward_fill_mesh_on_missed_detection(self):
        from face_hand_landmarks import BLENDSHAPE_NAMES
        extractor = FacialExpressionExtractor(Path("unused.task"))
        arrays = {
            "face_detected": np.array([False]),
            "face_blendshapes": np.full((1, len(BLENDSHAPE_NAMES)), np.nan),
            "face_landmarks": np.full((1, 478, 3), np.nan),
        }
        def detect(*args):
            arrays["face_detected"][0] = True
            arrays["face_blendshapes"][0] = 0
            arrays["face_landmarks"][0] = [0.45, 0.15, 0]
        extractor.model = SimpleNamespace(arrays=arrays, process=Mock())
        # The first mock callback must mutate the same arrays as the helper.
        extractor.model.process.side_effect = lambda *args: detect() if args[1] == 0 else None
        extractor.process(None, 0, 1000, 1000)
        self.assertIsNotNone(extractor.landmarks)
        extractor.process(None, 33, 1000, 1000)
        self.assertIsNone(extractor.landmarks)


if __name__ == "__main__":
    unittest.main()
