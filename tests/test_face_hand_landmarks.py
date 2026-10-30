"""Export-contract tests; native model inference is checked separately on video."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from face_hand_landmarks import BLENDSHAPE_NAMES, FaceHandLandmarkers


def points(count, x):
    return [SimpleNamespace(x=x, y=0.5, z=0.01) for _ in range(count)]


def category_name(name):
    parts = name.split("_")
    return "_neutral" if name == "neutral" else parts[0] + "".join(p.title() for p in parts[1:])


class FaceHandExportTests(unittest.TestCase):
    def test_missing_frames_and_duplicate_handedness_are_preserved(self):
        details = FaceHandLandmarkers(2, Path("face.task"), Path("hands.task"))
        # Return blendshapes out of order: export columns must still have named scores.
        scores = [SimpleNamespace(category_name=category_name(name), score=i / 100)
                  for i, name in reversed(list(enumerate(BLENDSHAPE_NAMES)))]
        details.face = Mock()
        details.face.detect_for_video.side_effect = [
            SimpleNamespace(face_landmarks=[points(478, 0.5)], face_blendshapes=[scores],
                            facial_transformation_matrixes=[np.eye(4)]),
            SimpleNamespace(face_landmarks=[]),
        ]
        details.hands = Mock()
        # Both detections may receive the same model label; neither may overwrite the other.
        handedness = [SimpleNamespace(category_name="Left", score=0.9)]
        details.hands.detect_for_video.side_effect = [
            SimpleNamespace(hand_landmarks=[points(21, 0.2), points(21, 0.8)],
                            hand_world_landmarks=[points(21, 0.02), points(21, 0.08)],
                            handedness=[handedness, handedness]),
            SimpleNamespace(hand_landmarks=[]),
        ]
        details.process(None, 0, 0, 1920, 1080)
        details.process(None, 33, 1, 1920, 1080)
        arrays = details.export_arrays(2)
        np.testing.assert_array_equal(arrays["face_detected"], [True, False])
        np.testing.assert_array_equal(arrays["hand_detected"], [[True, True], [False, False]])
        np.testing.assert_allclose(arrays["face_blendshapes"][0], np.arange(52) / 100)
        np.testing.assert_array_equal(arrays["hand_handedness"][0], ["Left", "Left"])
        np.testing.assert_allclose(arrays["hand_landmarks"][0, :, 0, 0], [0.2, 0.8])
        for name in ("face_landmarks", "face_blendshapes", "face_transformation_matrices",
                     "hand_landmarks", "hand_world_landmarks", "hand_handedness_scores"):
            self.assertTrue(np.isnan(arrays[name][1]).all(), name)
        self.assertEqual(details.face_points, {})
        self.assertEqual(details.hand_points, [])

        with TemporaryDirectory() as directory:
            paths = {name: Path(directory) / f"{name}.csv"
                     for name in ("face_blendshapes", "hand_landmarks")}
            details.export_csv(paths, 2, 30)
            with paths["face_blendshapes"].open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["face_detected"], "0")
            self.assertEqual(rows[1]["mouth_smile_left"], "")
            with paths["hand_landmarks"].open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2 * 2 * 21)
            self.assertEqual(rows[21]["hand_slot"], "1")
            self.assertEqual(rows[42]["hand_detected"], "0")
            self.assertEqual(rows[42]["x_normalized"], "")
            self.assertEqual(rows[42]["handedness_score"], "")
            self.assertEqual(rows[42]["timestamp_s"], "0.033333")

    def test_pose_only_has_no_face_or_hand_arrays(self):
        with FaceHandLandmarkers(2) as details:
            details.process(None, 0, 0, 1920, 1080)
            self.assertEqual(details.export_arrays(1), {})
            self.assertEqual(details.metadata(1, 30, None), {
                "face": {"enabled": False}, "hands": {"enabled": False},
            })


if __name__ == "__main__":
    unittest.main()
