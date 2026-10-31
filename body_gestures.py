"""Extract the speaker's skeleton/mask/hands and classify body behaviours.

BodyGestureExtractor owns the MediaPipe Pose and optional Hand models.
BodyGestureDetector contains arm/finger geometry and temporal classroom rules.
It needs a fixed camera, one visible
speaker, and scene annotations for location-dependent labels. See
BEHAVIOUR_RULES.md for every equation, unit, threshold, and limitation.
"""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
import math

import numpy as np

GESTURES = (
    "left_hand_raised", "right_hand_raised",
    "left_arm_extended", "right_arm_extended",
    "arms_crossed", "left_hand_pointing", "right_hand_pointing",
)
DIRECTIONS = ("image_right", "image_up_right", "image_up", "image_up_left",
              "image_left", "image_down_left", "image_down", "image_down_right")
FINGERS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def angle(a, vertex, b):
    first, second = a - vertex, b - vertex
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-9:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(first, second) / denominator, -1, 1))))


def direction(vector):
    """Eight image-relative directions; image y increases downward."""
    if np.linalg.norm(vector) <= 1e-9:
        return ""
    degrees = math.degrees(math.atan2(-float(vector[1]), float(vector[0])))
    return DIRECTIONS[int(math.floor((degrees + 22.5) / 45)) % 8]


def point_segment_distance(point, start, end):
    edge = end - start
    squared_length = float(np.dot(edge, edge))
    fraction = 0 if squared_length <= 1e-9 else float(np.clip(np.dot(point - start, edge) / squared_length, 0, 1))
    return float(np.linalg.norm(point - (start + fraction * edge)))


def index_finger_pointing(points):
    """A visible index extended with at least two other fingers curled; thumb ignored."""
    wrist = points[0]
    mcp, pip, dip, tip = points[list(FINGERS[0])]
    if not (angle(mcp, pip, dip) >= 155 and angle(pip, dip, tip) >= 150
            and np.linalg.norm(tip - wrist) > 1.1 * np.linalg.norm(pip - wrist)):
        return False
    curled = 0
    for indices in FINGERS[1:]:
        mcp, pip, _, tip = points[list(indices)]
        curled += bool(angle(mcp, pip, tip) < 140
                       or np.linalg.norm(tip - wrist) < 1.05 * np.linalg.norm(pip - wrist))
    return curled >= 2


def classify_frame(pose, pose_detected, width, height, hands=None,
                   hand_detected=None, handedness=None, handedness_scores=None,
                   visibility=0.5, handedness_confidence=0.7):
    """Return True/False/None states. None means insufficient observable data."""
    states = dict.fromkeys(GESTURES)
    directions = {}
    scale = np.array([width, height], dtype=float)
    points = pose[:, :2] * scale
    valid = (np.isfinite(pose).all(axis=1) & (pose[:, 3] >= visibility)
             & (pose[:, 4] >= visibility)
             & ((pose[:, :2] >= 0) & (pose[:, :2] <= 1)).all(axis=1))
    if not pose_detected:
        valid[:] = False
    shoulders_valid = bool(valid[[11, 12]].all())
    shoulder_width = float(np.linalg.norm(points[11] - points[12])) if shoulders_valid else 0
    arms = {}
    if shoulder_width > 1e-6:
        for side, indices in (("left", (11, 13, 15)), ("right", (12, 14, 16))):
            if valid[[indices[0], indices[2]]].all():
                shoulder, wrist = points[[indices[0], indices[2]]]
                states[f"{side}_hand_raised"] = bool(wrist[1] < shoulder[1] - 0.15 * shoulder_width)
            if valid[list(indices)].all():
                shoulder, elbow, wrist = points[list(indices)]
                bend = angle(shoulder, elbow, wrist)
                if math.isfinite(bend):
                    states[f"{side}_arm_extended"] = bool(bend >= 155)
                    arms[side] = (shoulder, elbow, wrist, bend)
                    if states[f"{side}_arm_extended"]:
                        directions[f"{side}_arm_extended"] = direction(wrist - shoulder)
        if len(arms) == 2:
            ls, le, lw, left_bend = arms["left"]
            rs, re, rw, right_bend = arms["right"]
            middle = (ls + rs) / 2
            body_axis = (ls - rs) / shoulder_width
            across_body = (np.dot(lw - middle, body_axis) < 0.1 * shoulder_width
                           and np.dot(rw - middle, body_axis) > -0.1 * shoulder_width)
            at_chest = all(middle[1] - 0.2 * shoulder_width <= wrist[1]
                           <= middle[1] + 1.2 * shoulder_width for wrist in (lw, rw))
            near_opposite_arm = (point_segment_distance(lw, rs, re) <= 0.65 * shoulder_width
                                 and point_segment_distance(rw, ls, le) <= 0.65 * shoulder_width)
            states["arms_crossed"] = bool(across_body and at_chest and near_opposite_arm
                                          and left_bend < 150 and right_bend < 150)

    if hands is not None:
        for side in ("left", "right"):
            # Duplicate side predictions are ambiguous; do not silently discard a hand.
            slots = [slot for slot in range(len(hands))
                     if hand_detected[slot] and str(handedness[slot]).lower() == side
                     and np.isfinite(handedness_scores[slot])
                     and handedness_scores[slot] >= handedness_confidence]
            if len(slots) != 1:
                continue
            hand = hands[slots[0]]
            if not (np.isfinite(hand).all() and ((hand[:, :2] >= 0) & (hand[:, :2] <= 1)).all()):
                continue
            hand_points = hand[:, :2] * scale
            label = f"{side}_hand_pointing"
            states[label] = index_finger_pointing(hand_points)
            if states[label]:
                directions[label] = direction(hand_points[8] - hand_points[5])
    return states, directions


def visible_points(landmarks, width, height, threshold):
    points = {}
    for i, point in enumerate(landmarks):
        if (
            point.visibility >= threshold
            and point.presence >= threshold
            and 0 <= point.x < 1
            and 0 <= point.y < 1
        ):
            points[i] = (round(point.x * (width - 1)), round(point.y * (height - 1)))
    return points


def draw_skeleton(image, points):
    import cv2
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

    for edge in PoseLandmarksConnections.POSE_LANDMARKS:
        if edge.start in points and edge.end in points:
            cv2.line(image, points[edge.start], points[edge.end], (80, 240, 80), 3, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(image, point, 5, (0, 180, 255), -1, cv2.LINE_AA)


@dataclass(frozen=True)
class BodyRules:
    visibility: float = 0.5
    handedness_confidence: float = 0.7
    history_s: float = 1.5
    motion_min_s: float = 1.0
    max_gap_s: float = 0.25
    hand_motion_window_s: float = 0.5
    hand_motion_min_s: float = 0.3
    hand_motion_range: float = 0.12
    stationary_radius: float = 0.12
    walking_speed: float = 0.25
    leg_deadband: float = 0.10
    leg_reversals: int = 2
    facing_tolerance_deg: float = 35.0
    turn_min_deg: float = 40.0
    turn_max_deg: float = 150.0
    zone_dwell_s: float = 0.4
    crossing_display_s: float = 0.8
    approach_speed: float = 0.25
    approach_radius: float = 3.0
    arrival_radius: float = 1.0
    desk_radius: float = 0.8
    standing_knee_deg: float = 160.0
    standing_tilt_deg: float = 25.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Body thresholds must be positive finite numbers")
        if not 0 < self.visibility <= 1 or not 0 < self.handedness_confidence <= 1:
            raise ValueError("Confidence thresholds must be in (0, 1]")
        if self.history_s < self.motion_min_s or self.turn_max_deg > 180 or self.turn_max_deg <= self.turn_min_deg:
            raise ValueError("Invalid temporal window or turning angle range")
        if self.hand_motion_window_s < self.hand_motion_min_s:
            raise ValueError("hand_motion_window_s must be at least hand_motion_min_s")
        if self.arrival_radius > self.approach_radius:
            raise ValueError("arrival_radius cannot exceed approach_radius")
        if not isinstance(self.leg_reversals, int):
            raise ValueError("leg_reversals must be an integer")


BODY_LABELS = (*GESTURES, "left_pointing_direction", "right_pointing_direction",
               "left_hand_movement", "right_hand_movement",
               "stationary", "walking", "standing", "moving_between_zones",
               "facing_board", "facing_class", "turning", "approaching_group",
               "group_arrival", "standing_beside_desk")


def validate_scene(scene):
    """Reject malformed annotations rather than silently classifying a wrong map."""
    allowed = {"zones", "groups", "desks", "board_yaw_deg", "class_yaw_deg"}
    if set(scene) - allowed:
        raise ValueError(f"Unknown scene settings: {sorted(set(scene) - allowed)}")
    for key in ("board_yaw_deg", "class_yaw_deg"):
        value = scene.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"{key} must be a finite angle or null")
    for key in ("zones", "groups", "desks"):
        names = set()
        for region in scene.get(key, []):
            if set(region) != {"name", "polygon"} or not isinstance(region["name"], str) or not region["name"]:
                raise ValueError(f"Each {key} region needs a nonempty name and polygon")
            polygon = np.asarray(region["polygon"], dtype=float)
            if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                    or not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any()):
                raise ValueError(f"{key} polygons need at least three normalized [x, y] vertices")
            area = abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1))
                       - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
            if area < 1e-8 or region["name"] in names:
                raise ValueError(f"{key} polygons must have area and unique names")
            names.add(region["name"])


def polygon_distance(point, polygon):
    """Pixel distance to a simple polygon; zero inside, including its boundary."""
    inside = False
    best = float("inf")
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        best = min(best, point_segment_distance(point, start, end))
        if (start[1] > point[1]) != (end[1] > point[1]):
            cross_x = (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0]
            if point[0] < cross_x:
                inside = not inside
    return 0.0 if inside or best < 1e-8 else best


def angular_difference(first, second):
    return (first - second + 180) % 360 - 180


def torso_yaw(world, valid):
    """0 faces camera; +90 faces image right; +/-180 faces away from camera."""
    if world is None or not valid[[11, 12, 23, 24]].all():
        return None
    p = world[:, :3]
    if not np.isfinite(p[[11, 12, 23, 24]]).all():
        return None
    across = p[12] - p[11]
    down = (p[23] + p[24] - p[11] - p[12]) / 2
    normal = np.cross(across, down)
    norm = np.linalg.norm(across) * np.linalg.norm(down)
    if norm < 1e-8 or np.linalg.norm(normal[[0, 2]]) / norm < 0.2:
        return None
    return math.degrees(math.atan2(float(normal[0]), float(-normal[2])))


def any_observed(values):
    values = list(values)
    return True if any(v is True for v in values) else (None if not values or any(v is None for v in values) else False)


class BodyGestureDetector:
    def __init__(self, width, height, rules=None, scene=None):
        self.rules = rules or BodyRules()
        self.scene = scene or {}
        validate_scene(self.scene)
        self.width, self.height = width, height
        self.scale = np.array([width, height], dtype=float)
        self.history = deque()
        self.hand_history = {side: deque() for side in ("left", "right")}
        self.last_t = None
        self.zone_candidate = None
        self.zone_since = None
        self.stable_zone = None
        self.crossing_until = -1.0
        self.crossing = ""

    def _reset(self):
        self.history.clear()
        self.zone_candidate = self.zone_since = self.stable_zone = None
        self.crossing_until = -1.0
        self.crossing = ""

    def _hand_movement(self, points, valid, timestamp_s, states, info):
        """Track each wrist relative to its shoulder, independently of torso/legs."""
        r = self.rules
        width = float(np.linalg.norm(points[11] - points[12])) if valid[[11, 12]].all() else 0.0
        for side, shoulder, wrist in (("left", 11, 15), ("right", 12, 16)):
            history = self.hand_history[side]
            if width < 1 or not valid[wrist]:
                history.clear()
                continue
            if history and timestamp_s - history[-1][0] > r.max_gap_s:
                history.clear()
            # Remove common image translation and normalize apparent body size.
            position = (points[wrist] - points[shoulder]) / width
            history.append((timestamp_s, position))
            while history and timestamp_s - history[0][0] > r.hand_motion_window_s + 1e-9:
                history.popleft()
            span = timestamp_s - history[0][0]
            info[f"{side}_hand_movement_span_s"] = span
            if span + 1e-9 < r.hand_motion_min_s:
                continue
            positions = np.array([item[1] for item in history])
            # Pairwise extent also catches a hand moving out and back to its start.
            extent = float(np.linalg.norm(positions[:, None] - positions[None, :], axis=2).max())
            info[f"{side}_hand_movement_range_shoulder_widths"] = extent
            states[f"{side}_hand_movement"] = bool(extent >= r.hand_motion_range)

    def update(self, pose, world, timestamp_s, **hand_args):
        r = self.rules
        if not math.isfinite(timestamp_s) or (self.last_t is not None and timestamp_s <= self.last_t):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if self.last_t is not None and timestamp_s - self.last_t > r.max_gap_s:
            self._reset()
        self.last_t = timestamp_s
        detected = bool(np.isfinite(pose[:, :3]).any())
        states, directions = classify_frame(pose, detected, self.width, self.height,
                                            visibility=r.visibility,
                                            handedness_confidence=r.handedness_confidence, **hand_args)
        states.update(dict.fromkeys(label for label in BODY_LABELS if label not in states))
        info = {"directions": directions, "zone": None, "zone_crossing": ""}
        for side in ("left", "right"):
            # A coarse arm-direction proxy is explicitly separate from finger pointing.
            extended = states[f"{side}_arm_extended"]
            arm_direction = directions.get(f"{side}_arm_extended", "")
            states[f"{side}_pointing_direction"] = (None if extended is None else
                                                     bool(extended and "down" not in arm_direction))
            if states[f"{side}_pointing_direction"]:
                directions[f"{side}_pointing_direction"] = arm_direction

        valid = (np.isfinite(pose).all(axis=1) & (pose[:, 3] >= r.visibility)
                 & (pose[:, 4] >= r.visibility)
                 & ((pose[:, :2] >= 0) & (pose[:, :2] <= 1)).all(axis=1))
        points = pose[:, :2] * self.scale
        self._hand_movement(points, valid, timestamp_s, states, info)
        if not valid[[11, 12, 23, 24]].all():
            self._reset()
            return states, info
        shoulders, hips = points[[11, 12]].mean(axis=0), points[[23, 24]].mean(axis=0)
        torso = float(np.linalg.norm(hips - shoulders))
        if torso < 1:
            self._reset()
            return states, info
        feet = points[[27, 28]].mean(axis=0) if valid[[27, 28]].all() else None
        leg_vector = (points[27] - points[28]) / torso if feet is not None else None
        yaw = torso_yaw(world, valid)
        row = {"t": timestamp_s, "hips": hips, "torso": torso, "feet": feet,
               "legs": leg_vector, "yaw": yaw}
        self.history.append(row)
        while self.history and timestamp_s - self.history[0]["t"] > r.history_s + 1e-9:
            self.history.popleft()
        span = timestamp_s - self.history[0]["t"]
        ready = span + 1e-9 >= r.motion_min_s
        info.update(torso_px=torso, torso_yaw_deg=yaw, history_span_s=span)

        if ready:
            centers = np.array([h["hips"] for h in self.history])
            unit = float(np.median([h["torso"] for h in self.history]))
            radius = float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max() / unit)
            speed = float(np.linalg.norm(centers[-1] - centers[0]) / unit / span)
            states["stationary"] = bool(radius <= r.stationary_radius)
            info.update(position_radius_torso=radius, displacement_speed_torso_s=speed)
            if all(h["legs"] is not None for h in self.history):
                legs = np.array([h["legs"] for h in self.history])
                axis = int(np.argmax(np.ptp(legs, axis=0)))
                signs = np.sign(legs[:, axis][np.abs(legs[:, axis]) >= r.leg_deadband])
                reversals = int(np.count_nonzero(signs[1:] != signs[:-1]))
                states["walking"] = bool(speed >= r.walking_speed and reversals >= r.leg_reversals
                                         and not states["stationary"])
                info["leg_reversals"] = reversals
            yaws = [h["yaw"] for h in self.history]
            if all(y is not None for y in yaws):
                change = math.degrees(float(np.unwrap(np.radians(yaws))[-1] - np.unwrap(np.radians(yaws))[0]))
                states["turning"] = bool(r.turn_min_deg <= abs(change) <= r.turn_max_deg)
                info["yaw_change_deg"] = change

        if yaw is not None:
            for label, key in (("facing_board", "board_yaw_deg"), ("facing_class", "class_yaw_deg")):
                target = self.scene.get(key)
                if target is not None:
                    states[label] = bool(abs(angular_difference(yaw, target)) <= r.facing_tolerance_deg)

        if valid[[23, 24, 25, 26, 27, 28]].all():
            knee_angles = [angle(points[hip], points[knee], points[ankle])
                           for hip, knee, ankle in ((23, 25, 27), (24, 26, 28))]
            tilt = math.degrees(math.atan2(abs(float(hips[0] - shoulders[0])), float(hips[1] - shoulders[1])))
            if all(math.isfinite(a) for a in knee_angles):
                states["standing"] = bool(min(knee_angles) >= r.standing_knee_deg and tilt <= r.standing_tilt_deg)
                info.update(knee_angles_deg=knee_angles, torso_tilt_deg=tilt)

        if feet is None:
            # Do not carry a zone identity across unobservable foot placement.
            self.zone_candidate = self.zone_since = self.stable_zone = None
            self.crossing_until = -1
            return states, info
        distances = {key: {region["name"]: polygon_distance(feet, np.asarray(region["polygon"]) * self.scale)
                           for region in self.scene.get(key, [])} for key in ("zones", "groups", "desks")}
        row["group_distances"] = distances["groups"]
        info["region_distances_torso"] = {key: {name: d / torso for name, d in values.items()}
                                           for key, values in distances.items()}
        if distances["zones"]:
            matches = [name for name, d in distances["zones"].items() if d == 0]
            # An overlapping boundary is ambiguous, not a forced choice.
            zone = matches[0] if len(matches) == 1 else None
            info["zone"] = zone
            if zone != self.zone_candidate:
                self.zone_candidate, self.zone_since = zone, timestamp_s
            if zone is not None and self.zone_since is not None and timestamp_s - self.zone_since + 1e-9 >= r.zone_dwell_s:
                if self.stable_zone is not None and zone != self.stable_zone:
                    self.crossing = f"{self.stable_zone} -> {zone}"
                    self.crossing_until = timestamp_s + r.crossing_display_s
                self.stable_zone = zone
            states["moving_between_zones"] = bool(timestamp_s < self.crossing_until)
            info["zone_crossing"] = self.crossing if states["moving_between_zones"] else ""

        if distances["groups"] and ready:
            approaching, arrivals = {}, {}
            first = self.history[0]
            for name, distance in distances["groups"].items():
                if not all(h.get("group_distances", {}).get(name) is not None for h in self.history):
                    approaching[name] = arrivals[name] = None
                    continue
                unit = float(np.median([h["torso"] for h in self.history]))
                old = first["group_distances"][name]
                closing_speed = (old - distance) / unit / span
                approaching[name] = bool(closing_speed >= r.approach_speed
                                         and old / unit > r.arrival_radius
                                         and distance / unit <= r.approach_radius)
                arrivals[name] = bool(approaching[name] and distance / unit <= r.arrival_radius)
            states["approaching_group"] = any_observed(approaching.values())
            states["group_arrival"] = any_observed(arrivals.values())
            info.update(approaching_groups=[name for name, v in approaching.items() if v],
                        arrived_groups=[name for name, v in arrivals.items() if v])
        if distances["desks"]:
            nearby = [name for name, d in distances["desks"].items() if d / torso <= r.desk_radius]
            info["nearby_desks"] = nearby
            if not nearby or states["standing"] is False or states["stationary"] is False:
                states["standing_beside_desk"] = False
            elif states["standing"] is True and states["stationary"] is True:
                states["standing_beside_desk"] = True
        return states, info


class BodyGestureExtractor:
    """Streaming Tasks extraction; segmentation is the Pose model's person mask."""

    def __init__(self, pose_model, width, height, rules=None, scene=None, hand_model=None):
        self.pose_model, self.hand_model = pose_model, hand_model
        self.detector = BodyGestureDetector(width, height, rules, scene)
        self.width, self.height = width, height
        self.stack = ExitStack()
        self.points = {}

    def __enter__(self):
        import mediapipe as mp
        from face_hand_landmarks import FaceHandLandmarkers
        try:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.pose_model),
                                                 delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
                output_segmentation_masks=True,
            )
            self.pose = self.stack.enter_context(mp.tasks.vision.PoseLandmarker.create_from_options(options))
            self.hands = self.stack.enter_context(FaceHandLandmarkers(1, hand_model=self.hand_model))
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def process(self, image, timestamp_ms, timestamp_s):
        result = self.pose.detect_for_video(image, timestamp_ms)
        pose = np.full((33, 5), np.nan)
        world = np.full_like(pose, np.nan)
        self.points = {}
        mask = None
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            pose[:] = [[p.x, p.y, p.z, p.visibility, p.presence] for p in landmarks]
            world[:] = [[p.x, p.y, p.z, p.visibility, p.presence] for p in result.pose_world_landmarks[0]]
            self.points = visible_points(landmarks, self.width, self.height, self.detector.rules.visibility)
            if result.segmentation_masks:
                mask = result.segmentation_masks[0].numpy_view().copy()
        args = {}
        if self.hand_model is not None:
            for name, array in self.hands.arrays.items():
                if array.dtype.kind == "b":
                    array[:] = False
                elif array.dtype.kind in "f":
                    array[:] = np.nan
                else:
                    array[:] = ""
            self.hands.process(image, timestamp_ms, 0, self.width, self.height)
            args = {"hands": self.hands.arrays["hand_landmarks"][0],
                    "hand_detected": self.hands.arrays["hand_detected"][0],
                    "handedness": self.hands.arrays["hand_handedness"][0],
                    "handedness_scores": self.hands.arrays["hand_handedness_scores"][0]}
        states, info = self.detector.update(pose, world, timestamp_s, **args)
        info["pose_detected"] = bool(result.pose_landmarks)
        return states, info, mask, pose, world

    def draw(self, image):
        draw_skeleton(image, self.points)
        self.hands.draw(image)
