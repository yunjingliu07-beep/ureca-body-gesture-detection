"""Face and hand extraction helpers for the single-speaker video pipeline."""

from contextlib import ExitStack
import csv
import hashlib
import re

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import (
    Blendshapes,
    FaceLandmarksConnections,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmark,
    HandLandmarksConnections,
)


MODEL_URLS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hands": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
BLENDSHAPE_NAMES = [item.name.lower() for item in Blendshapes]
HAND_LANDMARK_NAMES = [item.name.lower() for item in HandLandmark]


def in_frame_points(landmarks, width, height):
    """Face/hand results have no per-point visibility or presence confidence."""
    return {
        i: (round(point.x * (width - 1)), round(point.y * (height - 1)))
        for i, point in enumerate(landmarks)
        if np.isfinite([point.x, point.y]).all()
        and 0 <= point.x < 1 and 0 <= point.y < 1
    }


def draw_connections(image, points, connections, color, thickness=1):
    for edge in connections:
        if edge.start in points and edge.end in points:
            cv2.line(image, points[edge.start], points[edge.end], color, thickness, cv2.LINE_AA)


def csv_numbers(values):
    return [f"{float(value):.7f}" if np.isfinite(value) else "" for value in values]


class FaceHandLandmarkers:
    """Independent Tasks models; result slots are not persistent person/hand IDs."""

    def __init__(self, frame_limit, face_model=None, hand_model=None):
        self.models = {"face": face_model, "hands": hand_model}
        self.arrays = {}
        if face_model is not None:
            self.arrays.update(
                face_landmarks=np.full((frame_limit, 478, 3), np.nan, dtype=np.float32),
                face_detected=np.zeros(frame_limit, dtype=bool),
                face_blendshapes=np.full((frame_limit, len(BLENDSHAPE_NAMES)), np.nan, dtype=np.float32),
                face_transformation_matrices=np.full((frame_limit, 4, 4), np.nan, dtype=np.float32),
            )
        if hand_model is not None:
            self.arrays.update(
                hand_landmarks=np.full((frame_limit, 2, 21, 3), np.nan, dtype=np.float32),
                hand_world_landmarks=np.full((frame_limit, 2, 21, 3), np.nan, dtype=np.float32),
                hand_detected=np.zeros((frame_limit, 2), dtype=bool),
                hand_handedness=np.full((frame_limit, 2), "", dtype="U8"),
                hand_handedness_scores=np.full((frame_limit, 2), np.nan, dtype=np.float32),
            )
        self.face = None
        self.hands = None
        self.face_points = {}
        self.hand_points = []
        self.stack = ExitStack()

    def __enter__(self):
        vision = mp.tasks.vision
        try:
            if self.models["face"] is not None:
                self.face = self.stack.enter_context(vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=str(self.models["face"]),
                            delegate=mp.tasks.BaseOptions.Delegate.CPU,
                        ),
                        running_mode=vision.RunningMode.VIDEO,
                        num_faces=1,
                        output_face_blendshapes=True,
                        output_facial_transformation_matrixes=True,
                    )
                ))
            if self.models["hands"] is not None:
                self.hands = self.stack.enter_context(vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=str(self.models["hands"]),
                            delegate=mp.tasks.BaseOptions.Delegate.CPU,
                        ),
                        running_mode=vision.RunningMode.VIDEO,
                        num_hands=2,
                    )
                ))
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def process(self, image, timestamp_ms, frame_index, width, height):
        """Run each enabled model even if the body pose was not detected."""
        self.face_points = {}
        self.hand_points = []
        if self.face is not None:
            result = self.face.detect_for_video(image, timestamp_ms)
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                self.arrays["face_detected"][frame_index] = True
                self.arrays["face_landmarks"][frame_index] = [[p.x, p.y, p.z] for p in landmarks]
                scores = {
                    re.sub(r"(?<!^)(?=[A-Z])", "_", c.category_name).lower().lstrip("_"): c.score
                    for c in result.face_blendshapes[0]
                }
                self.arrays["face_blendshapes"][frame_index] = [scores[name] for name in BLENDSHAPE_NAMES]
                self.arrays["face_transformation_matrices"][frame_index] = result.facial_transformation_matrixes[0]
                self.face_points = in_frame_points(landmarks, width, height)
        if self.hands is not None:
            result = self.hands.detect_for_video(image, timestamp_ms)
            for slot, landmarks in enumerate(result.hand_landmarks):
                category = max(result.handedness[slot], key=lambda item: item.score)
                self.arrays["hand_detected"][frame_index, slot] = True
                self.arrays["hand_landmarks"][frame_index, slot] = [[p.x, p.y, p.z] for p in landmarks]
                self.arrays["hand_world_landmarks"][frame_index, slot] = [
                    [p.x, p.y, p.z] for p in result.hand_world_landmarks[slot]
                ]
                self.arrays["hand_handedness"][frame_index, slot] = category.category_name
                self.arrays["hand_handedness_scores"][frame_index, slot] = category.score
                self.hand_points.append((category.category_name, in_frame_points(landmarks, width, height)))

    def draw(self, image):
        draw_connections(image, self.face_points, FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS, (255, 220, 0))
        for label, points in self.hand_points:
            color = (255, 100, 255) if label == "Left" else (0, 160, 255)
            draw_connections(image, points, HandLandmarksConnections.HAND_CONNECTIONS, color, 2)
            for point in points.values():
                cv2.circle(image, point, 2, color, -1, cv2.LINE_AA)
            if 0 in points:
                cv2.putText(image, label, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def export_arrays(self, count):
        arrays = {name: values[:count] for name, values in self.arrays.items()}
        if self.models["face"] is not None:
            arrays["face_blendshape_names"] = np.array(BLENDSHAPE_NAMES)
        if self.models["hands"] is not None:
            arrays["hand_landmark_names"] = np.array(HAND_LANDMARK_NAMES)
        if self.arrays:
            arrays["detail_coordinate_names"] = np.array(["x", "y", "z"])
        return arrays

    def export_csv(self, paths, count, fps):
        if self.models["face"] is not None:
            with paths["face_blendshapes"].open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["frame_index", "timestamp_s", "face_detected", *BLENDSHAPE_NAMES])
                for index in range(count):
                    writer.writerow([index, f"{index / fps:.6f}", int(self.arrays["face_detected"][index]),
                                     *csv_numbers(self.arrays["face_blendshapes"][index])])
        if self.models["hands"] is not None:
            with paths["hand_landmarks"].open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow([
                    "frame_index", "timestamp_s", "hand_slot", "hand_detected", "handedness",
                    "handedness_score", "landmark_index", "landmark_name", "x_normalized",
                    "y_normalized", "z_normalized", "world_x_m", "world_y_m", "world_z_m",
                ])
                for index in range(count):
                    for slot in range(2):
                        for landmark_id, name in enumerate(HAND_LANDMARK_NAMES):
                            writer.writerow([
                                index, f"{index / fps:.6f}", slot, int(self.arrays["hand_detected"][index, slot]),
                                self.arrays["hand_handedness"][index, slot],
                                *csv_numbers([self.arrays["hand_handedness_scores"][index, slot]]),
                                landmark_id, name,
                                *csv_numbers(self.arrays["hand_landmarks"][index, slot, landmark_id]),
                                *csv_numbers(self.arrays["hand_world_landmarks"][index, slot, landmark_id]),
                            ])

    def metadata(self, count, fps, missing_intervals):
        result = {}
        for modality, path in self.models.items():
            if path is None:
                result[modality] = {"enabled": False}
                continue
            mask = self.arrays["face_detected"][:count] if modality == "face" else self.arrays["hand_detected"][:count].any(axis=1)
            result[modality] = {
                "enabled": True,
                "model": str(path.resolve()),
                "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_url": MODEL_URLS[modality],
                "frames_detected": int(mask.sum()),
                "detection_fraction": float(mask.mean()),
                "no_detection_intervals": missing_intervals(mask.tolist(), fps),
            }
        return result
