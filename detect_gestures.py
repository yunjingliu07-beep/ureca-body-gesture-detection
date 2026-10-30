"""Extract observable arm gestures from a video or saved landmark arrays.

Examples:
    python detect_gestures.py --input classroom.mp4 --seconds 120
    python detect_gestures.py --landmarks outputs/session_arrays.npz
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

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


def event_rows(timestamps, states, directions, frame_step, min_duration):
    """Unknown or negative frames end an episode; duration uses an exclusive end."""
    events = []
    for gesture in GESTURES:
        start = None
        for index in range(len(timestamps) + 1):
            active = index < len(timestamps) and states[index][gesture] is True
            if active and start is None:
                start = index
            if not active and start is not None:
                end = float(timestamps[index - 1] + frame_step)
                duration = end - float(timestamps[start])
                if duration + 1e-9 >= min_duration:
                    labels = [row.get(gesture, "") for row in directions[start:index]]
                    labels = [label for label in labels if label]
                    events.append({
                        "gesture": gesture, "start_frame": start, "end_frame_exclusive": index,
                        "start_s": float(timestamps[start]), "end_s_exclusive": end,
                        "duration_s": duration,
                        "dominant_direction": Counter(labels).most_common(1)[0][0] if labels else "",
                    })
                start = None
    return sorted(events, key=lambda row: (row["start_s"], row["gesture"]))


def load_landmarks(path, fps=None):
    with np.load(path, allow_pickle=False) as archive:
        required = ("normalized_landmarks", "pose_detected", "timestamps_s")
        data = {key: archive[key] for key in required}
        if "hand_landmarks" in archive:
            for key in ("hand_landmarks", "hand_detected", "hand_handedness", "hand_handedness_scores"):
                data[key] = archive[key]
    timestamps = data["timestamps_s"]
    count = len(timestamps)
    if count == 0 or timestamps.shape != (count,) or not np.isfinite(timestamps).all():
        raise ValueError("Expected a nonempty, finite timestamp vector")
    if data["normalized_landmarks"].shape != (count, 33, 5) or data["pose_detected"].shape != (count,):
        raise ValueError("Expected body arrays shaped (N, 33, 5) and a detection mask shaped (N,)")
    if "hand_landmarks" in data:
        if data["hand_landmarks"].shape != (count, 2, 21, 3):
            raise ValueError("Expected hand landmarks shaped (N, 2, 21, 3)")
        if any(data[key].shape != (count, 2) for key in
               ("hand_detected", "hand_handedness", "hand_handedness_scores")):
            raise ValueError("Expected hand masks, labels, and scores shaped (N, 2)")
    metadata_path = path.with_name(path.stem.removesuffix("_arrays") + "_metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if count > 1:
        differences = np.diff(timestamps)
        step = float(np.median(differences))
        if step <= 0 or not np.allclose(differences, step, rtol=1e-4, atol=1e-6):
            raise ValueError("Landmarks must have regularly sampled, increasing timestamps")
        if fps is not None and not math.isclose(step, 1 / fps, rel_tol=1e-4):
            raise ValueError("--fps disagrees with the saved timestamps")
    else:
        rate = fps if fps is not None else metadata.get("fps")
        if rate is None or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Provide --fps when a single-frame archive has no FPS metadata")
        step = 1 / rate
    width, height = metadata.get("width"), metadata.get("height")
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("Matching extraction metadata with image width and height is required")
    return data, metadata, step


def analyze(path, output_dir, visibility=0.5, handedness_confidence=0.7,
            min_duration=0.3, fps=None):
    data, source_metadata, step = load_landmarks(path, fps)
    timestamps = data["timestamps_s"]
    states, directions = [], []
    for index in range(len(timestamps)):
        hand_args = {}
        if "hand_landmarks" in data:
            hand_args = {
                "hands": data["hand_landmarks"][index],
                "hand_detected": data["hand_detected"][index],
                "handedness": data["hand_handedness"][index],
                "handedness_scores": data["hand_handedness_scores"][index],
            }
        state, directions_for_frame = classify_frame(
            data["normalized_landmarks"][index], bool(data["pose_detected"][index]),
            source_metadata["width"], source_metadata["height"],
            visibility=visibility, handedness_confidence=handedness_confidence, **hand_args,
        )
        states.append(state)
        directions.append(directions_for_frame)
    events = event_rows(timestamps, states, directions, step, min_duration)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = path.stem.removesuffix("_arrays")
    paths = {name: output_dir / f"{prefix}_gestures_{name}.{extension}"
             for name, extension in (("frames", "csv"), ("events", "csv"), ("summary", "json"))}
    direction_columns = [f"{gesture}_direction" for gesture in GESTURES if gesture.endswith(("extended", "pointing"))]
    with paths["frames"].open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["frame_index", "timestamp_s", "pose_detected", *GESTURES, *direction_columns])
        for index, state in enumerate(states):
            writer.writerow([
                index, f"{timestamps[index]:.6f}", int(data["pose_detected"][index]),
                *["" if state[name] is None else int(state[name]) for name in GESTURES],
                *[directions[index].get(column.removesuffix("_direction"), "") for column in direction_columns],
            ])
    with paths["events"].open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "gesture", "start_frame", "end_frame_exclusive", "start_s", "end_s_exclusive",
            "duration_s", "dominant_direction",
        ])
        writer.writeheader()
        writer.writerows(events)
    summary = {
        "landmark_source": str(path.resolve()), "processed_frames": len(timestamps),
        "start_s": float(timestamps[0]), "end_s_exclusive": float(timestamps[-1] + step),
        "visibility_threshold": visibility, "handedness_confidence_threshold": handedness_confidence,
        "minimum_event_duration_s": min_duration,
        "gestures": {name: {
            "event_count": sum(row["gesture"] == name for row in events),
            "event_duration_s": sum(row["duration_s"] for row in events if row["gesture"] == name),
            "observable_duration_s": sum(state[name] is not None for state in states) * step,
        } for name in GESTURES},
        "notes": [
            "Geometric rules for observable arm gestures; thresholds require validation on the intended recordings.",
            "Frame CSV: 1=positive, 0=observed negative, blank=insufficient data. Events use positive consecutive frames only.",
            "Minimum duration filters event rows, not raw frame labels. Unknown frames end events; missing data is not filled.",
            "Left/right hand labels follow the Hand Landmarker predictions. Duplicate or low-confidence side predictions remain unknown.",
            "Directions refer to the unmirrored image. Pointing describes the finger shape/direction, not a target object.",
            "Multiple gestures can overlap; their total durations need not sum to the video duration.",
        ],
        "outputs": {name: str(file.resolve()) for name, file in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Video to extract and analyze")
    source.add_argument("--landmarks", type=Path, help="Existing *_arrays.npz with matching metadata JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gestures"))
    parser.add_argument("--seconds", type=float, default=120, help="Video duration to process from its beginning")
    parser.add_argument("--visibility", type=float, default=0.5)
    parser.add_argument("--handedness-confidence", type=float, default=0.7)
    parser.add_argument("--min-duration", type=float, default=0.3, help="Minimum continuous gesture duration in seconds")
    parser.add_argument("--fps", type=float, help="FPS for single-frame archives; otherwise checked against timestamps")
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_full.task"))
    parser.add_argument("--face-model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--hand-model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--no-hands", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.visibility <= 1 or not 0 <= args.handedness_confidence <= 1:
        parser.error("Confidence thresholds must be between 0 and 1")
    if not math.isfinite(args.min_duration) or args.min_duration < 0:
        parser.error("--min-duration must be finite and nonnegative")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    path = args.landmarks
    if args.input is not None:
        from extract_speaker_pose import main as extract
        extraction_args = ["--input", str(args.input), "--seconds", str(args.seconds),
                           "--output-dir", str(args.output_dir / "landmarks"),
                           "--visibility", str(args.visibility), "--model", str(args.model),
                           "--face-model", str(args.face_model), "--hand-model", str(args.hand_model)]
        for flag in ("no_face", "no_hands"):
            if getattr(args, flag):
                extraction_args.append("--" + flag.replace("_", "-"))
        metadata = extract(extraction_args)
        path = Path(metadata["outputs"]["arrays"])
    try:
        summary = analyze(path, args.output_dir, args.visibility, args.handedness_confidence,
                          args.min_duration, args.fps)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"processed_frames": summary["processed_frames"],
                      "events": {name: item["event_count"] for name, item in summary["gestures"].items()},
                      "outputs": summary["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
