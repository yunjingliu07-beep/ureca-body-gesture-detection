"""Estimate teacher approach/retreat using head depth and corroborating torso scale.

CameraGuard checks background optical flow before allowing camera-relative labels.
These are relative-depth heuristics, not calibrated walking speed measurements.
"""

from collections import deque
import math

import cv2
import numpy as np

from .config import CameraMovementRules


CAMERA_LABELS = ("moving_toward_camera", "moving_away_from_camera")


class CameraGuard:
    def __init__(self, rules=None):
        self.rules = rules or CameraMovementRules()
        self.previous = None
        self.previous_mask = None
        self.history = deque()

    def update(self, frame, timestamp_s, person_mask=None):
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        mask = np.full_like(gray, 255)
        if person_mask is not None:
            foreground = cv2.resize(person_mask, (320, 180)) > 0.2
            foreground = cv2.dilate(foreground.astype(np.uint8), np.ones((15, 15), np.uint8))
            mask[foreground > 0] = 0
        info = {"camera_stable": False, "camera_cut": False, "camera_reason": "initial_frame"}
        if self.previous is not None:
            difference = float(np.mean(cv2.absdiff(gray, self.previous)) / 255)
            info["camera_frame_difference"] = difference
            info["camera_cut"] = difference >= self.rules.cut_difference
            points = cv2.goodFeaturesToTrack(self.previous, 150, 0.02, 8, mask=self.previous_mask)
            info["camera_reason"] = "insufficient_background_features"
            if not info["camera_cut"] and points is not None and len(points) >= 12:
                following, valid, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
                if following is not None:
                    a, b = points[valid.ravel() == 1], following[valid.ravel() == 1]
                    if len(a) >= 12:
                        transform, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                                       ransacReprojThreshold=1.5)
                        if transform is not None and inliers is not None and float(inliers.mean()) >= 0.6:
                            zoom = abs(math.log(max(1e-12, float(np.linalg.norm(transform[:, 0])))))
                            shift = float(np.linalg.norm(transform[:, 2]) / math.hypot(320, 180))
                            self.history.append((timestamp_s, zoom, shift))
                            while self.history and timestamp_s - self.history[0][0] > self.rules.window_s:
                                self.history.popleft()
                            zoom_total = sum(v[1] for v in self.history)
                            shift_total = sum(v[2] for v in self.history)
                            stable = (zoom_total < self.rules.background_zoom_log
                                      and shift_total < self.rules.background_translation)
                            info.update(camera_stable=stable, camera_reason="stable" if stable else "camera_motion",
                                        background_zoom_log=zoom_total, background_translation=shift_total)
            if not info["camera_stable"]:
                if info["camera_cut"]:
                    info["camera_reason"] = "camera_cut"
                # Retain motion history until it rolls out; failed flow breaks evidence.
                if info["camera_reason"] != "camera_motion":
                    self.history.clear()
        self.previous, self.previous_mask = gray, mask
        return info


class CameraMovementDetector:
    def __init__(self, rules=None, visibility=0.5):
        self.rules = rules or CameraMovementRules()
        self.visibility = visibility
        self.history = deque()

    def update(self, timestamp_s, head_depth, pose, width, height, torso_yaw_deg, camera_stable):
        states = dict.fromkeys(CAMERA_LABELS)
        info = {"camera_movement_reason": "insufficient_evidence"}
        indices = [11, 12, 23, 24]
        observable = (pose is not None and np.isfinite(pose[indices]).all()
                      and (pose[indices, 3:5] >= self.visibility).all()
                      and ((pose[indices, :2] >= 0) & (pose[indices, :2] <= 1)).all())
        if not camera_stable or not observable or head_depth is None or not math.isfinite(head_depth) or head_depth <= 0:
            self.history.clear()
            info["camera_movement_reason"] = "invalid_camera_body_or_depth"
            return states, info
        points = pose[:, :2] * [width, height]
        torso_vector = points[[23, 24]].mean(axis=0) - points[[11, 12]].mean(axis=0)
        torso_size = float(np.linalg.norm(torso_vector))
        shoulder_size = float(np.linalg.norm(points[11] - points[12]))
        tilt = math.degrees(math.atan2(float(torso_vector[0]), float(torso_vector[1])))
        if torso_size < 5 or shoulder_size < 5 or torso_yaw_deg is None:
            self.history.clear()
            return states, info
        if self.history and timestamp_s - self.history[-1][0] > self.rules.max_gap_s:
            self.history.clear()
        self.history.append((timestamp_s, head_depth, torso_size, shoulder_size, torso_yaw_deg, tilt))
        while self.history and timestamp_s - self.history[0][0] > self.rules.window_s + 1e-9:
            self.history.popleft()
        span = timestamp_s - self.history[0][0]
        if span + 1e-9 < self.rules.min_s:
            return states, info
        values = np.asarray(self.history)
        yaw_range = np.ptp(np.unwrap(np.radians(values[:, 4]))) * 180 / math.pi
        tilt_range = np.ptp(np.unwrap(np.radians(values[:, 5]))) * 180 / math.pi
        if max(yaw_range, tilt_range) > self.rules.orientation_change_deg:
            info["camera_movement_reason"] = "changing_body_orientation"
            return states, info
        # Medians at both ends suppress individual depth/landmark outliers.
        first = values[values[:, 0] <= values[0, 0] + 0.2]
        last = values[values[:, 0] >= values[-1, 0] - 0.2]
        elapsed = float(np.median(last[:, 0]) - np.median(first[:, 0]))
        rates = (np.median(last[:, 1:4], axis=0) - np.median(first[:, 1:4], axis=0)) / np.median(values[:, 1:4], axis=0) / elapsed
        depth, torso, shoulder = map(float, rates)
        info.update(depth_relative_rate_s=depth, torso_scale_rate_s=torso,
                    shoulder_scale_rate_s=shoulder, camera_motion_span_s=span)
        toward = depth < -self.rules.depth_rate and min(torso, shoulder) > self.rules.body_scale_rate
        away = depth > self.rules.depth_rate and max(torso, shoulder) < -self.rules.body_scale_rate
        still = abs(depth) <= self.rules.depth_rate and max(abs(torso), abs(shoulder)) <= self.rules.body_scale_rate
        if toward or away or still:
            states.update(moving_toward_camera=bool(toward), moving_away_from_camera=bool(away))
            info["camera_movement_reason"] = "consistent_depth_and_body_scale"
        else:
            info["camera_movement_reason"] = "conflicting_or_head_only_motion"
        return states, info
