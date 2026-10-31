"""Geometry tests for arm and finger rules used by the current body extractor."""

import unittest

import numpy as np

from body_gestures import classify_frame, direction


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


if __name__ == "__main__":
    unittest.main()
