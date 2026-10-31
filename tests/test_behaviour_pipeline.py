"""Check temporal behaviour rules with controlled skeletons and face scores."""

import csv
import io
import math
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from behaviour_common import TemporalGate
from body_gestures import (BodyGestureDetector, BodyRules, draw_skeleton,
                           polygon_distance, torso_yaw, validate_scene, visible_points)
from facial_expressions import classify_expressions
from main import DEFAULT_INPUT, PROJECT_DIR, EventWriter, build_parser, load_settings, render_mask


def body(x=0.5, leg=0.04, yaw=0):
    pose = np.full((33, 5), np.nan)
    for index, xy in {11: (x + 0.07, 0.25), 12: (x - 0.07, 0.25),
                      13: (x + 0.10, 0.4), 14: (x - 0.10, 0.4),
                      15: (x + 0.12, 0.52), 16: (x - 0.12, 0.52),
                      23: (x + 0.05, 0.5), 24: (x - 0.05, 0.5),
                      25: (x + leg, 0.7), 26: (x - leg, 0.7),
                      27: (x + leg, 0.9), 28: (x - leg, 0.9)}.items():
        pose[index] = [*xy, 0, 0.99, 0.99]
    world = pose.copy()
    theta = math.radians(yaw)
    across = np.array([math.cos(theta), 0, math.sin(theta)]) * 0.15
    world[11, :3], world[12, :3] = across, -across
    world[23, :3], world[24, :3] = across + [0, 0.4, 0], -across + [0, 0.4, 0]
    return pose, world


def region(name, left, right):
    return {"name": name, "polygon": [[left, 0.8], [right, 0.8], [right, 1], [left, 1]]}


class BehaviourTests(unittest.TestCase):
    def test_default_calculus_test_and_duration_overrides(self):
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.input, DEFAULT_INPUT)
        self.assertTrue(args.input.name.startswith("Calculus 1 Lecture 0.1"))
        self.assertEqual(args.input.suffix, ".webm")
        self.assertEqual(args.input.parent, PROJECT_DIR)
        self.assertEqual(args.seconds, 30)
        self.assertEqual(args.pose_model.parent, PROJECT_DIR / "models")
        self.assertEqual(args.output_dir, PROJECT_DIR / "outputs/behaviours")
        self.assertIsNone(parser.parse_args(["--full-video"]).seconds)
        args = parser.parse_args(["--input", "other.mp4", "--seconds", "10"])
        self.assertEqual(args.input, Path("other.mp4"))
        self.assertEqual(args.seconds, 10)

    def test_body_drawing_helpers_without_legacy_extractor(self):
        landmarks = [SimpleNamespace(x=0.4, y=0.4, visibility=0.99, presence=0.99),
                     SimpleNamespace(x=0.5, y=0.5, visibility=0.1, presence=0.99),
                     SimpleNamespace(x=1.1, y=0.6, visibility=0.99, presence=0.99)]
        points = visible_points(landmarks, 100, 100, 0.5)
        self.assertEqual(points, {0: (40, 40)})
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_skeleton(image, {11: (25, 25), 12: (75, 25)})
        self.assertTrue(image[25, 50].any())
        self.assertFalse(image[90, 90].any())

    def test_temporal_gate_resets_unknown_and_negative(self):
        gate = TemporalGate(0.25)
        self.assertFalse(gate.update({"smiling": True}, 0)["smiling"])
        self.assertTrue(gate.update({"smiling": True}, 0.3)["smiling"])
        self.assertIsNone(gate.update({"smiling": None}, 0.4)["smiling"])
        self.assertFalse(gate.update({"smiling": True}, 0.5)["smiling"])
        self.assertFalse(gate.update({"smiling": False}, 0.9)["smiling"])

    def test_stationary_standing_and_unconfigured_rules(self):
        detector = BodyGestureDetector(1000, 1000)
        pose, world = body()
        first, _ = detector.update(pose, world, 0)
        self.assertIsNone(first["stationary"])
        for t in np.arange(0.1, 1.51, 0.1):
            states, _ = detector.update(pose, world, float(t))
        self.assertTrue(states["stationary"])
        self.assertTrue(states["standing"])
        self.assertFalse(states["walking"])
        self.assertIsNone(states["facing_board"])
        self.assertIsNone(states["moving_between_zones"])
        self.assertIsNone(states["approaching_group"])

    def test_displacement_alone_is_not_walking_but_alternating_legs_are(self):
        fixed_legs, stepping = BodyGestureDetector(1000, 1000), BodyGestureDetector(1000, 1000)
        for index in range(16):
            x, t = 0.3 + index * 0.015, index / 10
            states_fixed, _ = fixed_legs.update(*body(x=x), t)
            states_step, info = stepping.update(*body(x=x, leg=0.05 * math.cos(index * math.pi / 3)), t)
        self.assertFalse(states_fixed["walking"])
        self.assertTrue(states_step["walking"])
        self.assertGreaterEqual(info["leg_reversals"], 2)
        self.assertFalse(states_step["stationary"])

    def test_missing_pose_and_time_gap_break_history(self):
        detector = BodyGestureDetector(1000, 1000)
        for index in range(12):
            states, _ = detector.update(*body(), index / 10)
        self.assertTrue(states["stationary"])
        states, _ = detector.update(np.full((33, 5), np.nan), None, 1.2)
        self.assertIsNone(states["stationary"])
        states, _ = detector.update(*body(), 1.3)
        self.assertIsNone(states["stationary"])
        states, _ = detector.update(*body(), 3.0)
        self.assertIsNone(states["stationary"])

    def test_turn_and_orientation_wrap_at_180(self):
        detector = BodyGestureDetector(1000, 1000, scene={"board_yaw_deg": 180, "class_yaw_deg": 0})
        for index in range(16):
            states, info = detector.update(*body(yaw=150 + index * 4), index / 10)
        self.assertTrue(states["turning"])
        self.assertTrue(states["facing_board"])
        self.assertFalse(states["facing_class"])
        self.assertAlmostEqual(info["yaw_change_deg"], 60)
        pose, world = body(yaw=90)
        self.assertAlmostEqual(torso_yaw(world, np.isfinite(pose).all(axis=1)), 90)

    def test_zones_require_dwell_and_do_not_cross_after_missing_feet(self):
        scene = {"zones": [region("A", 0.1, 0.45), region("B", 0.55, 0.9)]}
        detector = BodyGestureDetector(1000, 1000, scene=scene)
        for index in range(6):
            states, _ = detector.update(*body(x=0.3), index / 10)
        self.assertFalse(states["moving_between_zones"])
        for index in range(6, 12):
            states, info = detector.update(*body(x=0.7), index / 10)
        self.assertTrue(states["moving_between_zones"])
        self.assertEqual(info["zone_crossing"], "A -> B")
        pose, world = body(x=0.7)
        pose[[27, 28]] = np.nan
        states, _ = detector.update(pose, world, 1.2)
        self.assertIsNone(states["moving_between_zones"])
        for index in range(13, 19):
            states, _ = detector.update(*body(x=0.3), index / 10)
        self.assertFalse(states["moving_between_zones"])

    def test_group_approach_arrival_and_desk_standing(self):
        scene = {"groups": [region("group1", 0.8, 0.95)], "desks": [region("desk1", 0.4, 0.6)]}
        detector = BodyGestureDetector(1000, 1000, scene=scene)
        for index in range(16):
            states, info = detector.update(*body(x=0.3 + index * 0.02), index / 10)
        self.assertTrue(states["approaching_group"])
        self.assertTrue(states["group_arrival"])
        self.assertEqual(info["arrived_groups"], ["group1"])
        self.assertFalse(states["standing_beside_desk"])
        detector = BodyGestureDetector(1000, 1000, scene=scene)
        for index in range(16):
            states, _ = detector.update(*body(x=0.5), index / 10)
        self.assertTrue(states["standing_beside_desk"])

    def test_face_missing_data_and_separate_brow_rules(self):
        states, _ = classify_expressions({})
        self.assertTrue(all(value is None for value in states.values()))
        scores = dict(mouth_smile_left=0.7, mouth_smile_right=0.5, brow_inner_up=0.5,
                      brow_outer_up_left=0.1, brow_outer_up_right=0.1,
                      brow_down_left=0.2, brow_down_right=0.1)
        states, _ = classify_expressions(scores)
        self.assertTrue(states["smiling"])
        self.assertTrue(states["brow_raising"])
        self.assertFalse(states["brow_lowering"])
        states, _ = classify_expressions(scores, detected=False)
        self.assertTrue(all(value is None for value in states.values()))

    def test_geometry_config_and_mask(self):
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        self.assertEqual(polygon_distance(np.array([1, 1]), polygon), 0)
        self.assertAlmostEqual(polygon_distance(np.array([3, 1]), polygon), 1)
        frame = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask = np.zeros((32, 32))
        mask[:16] = 1
        output = render_mask(frame, mask)
        np.testing.assert_array_equal(output[16:], frame[16:])
        self.assertFalse(np.array_equal(output[:16], frame[:16]))
        self.assertFalse(render_mask(frame, None, "mask").any())
        load_settings(Path(__file__).resolve().parents[1] / "classroom_config.example.json")
        with self.assertRaises(ValueError):
            validate_scene({"zones": [region("bad", -1, 0.5)]})
        with self.assertRaises(ValueError):
            BodyRules(history_s=0.5, motion_min_s=1)

    def test_events_end_on_unknown_and_flush_at_clip_end(self):
        buffer = io.StringIO()
        events = EventWriter(csv.writer(buffer), 10)
        for index, value in enumerate([False, True, True, None, True]):
            events.update({"smiling": value}, index)
        events.close(5)
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["end_frame_exclusive"], "3")
        self.assertEqual(rows[1]["end_s_exclusive"], "0.500000")


if __name__ == "__main__":
    unittest.main()
