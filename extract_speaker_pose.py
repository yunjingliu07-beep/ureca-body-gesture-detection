"""Extract synchronized MediaPipe body, face, and hand landmarks from a video.

Run with the project's existing environment:
    conda run -n ureca python extract_speaker_pose.py
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ureca-matplotlib")

import cv2
import mediapipe as mp
import numpy as np
from face_hand_landmarks import FaceHandLandmarkers
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarksConnections,
)
try:
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark
except ImportError:
    # Older releases expose the same landmark indices through the legacy API.
    from mediapipe.python.solutions.pose import PoseLandmark


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
    for edge in PoseLandmarksConnections.POSE_LANDMARKS:
        if edge.start in points and edge.end in points:
            cv2.line(image, points[edge.start], points[edge.end], (80, 240, 80), 3, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(image, point, 5, (0, 180, 255), -1, cv2.LINE_AA)


def missing_intervals(detected, fps):
    intervals = []
    start = None
    for i, found in enumerate([*detected, True]):
        if not found and start is None:
            start = i
        elif found and start is not None:
            intervals.append({"start_s": start / fps, "end_s_exclusive": i / fps})
            start = None
    return intervals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("linear algebra.mp4"))
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_full.task"))
    parser.add_argument("--face-model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--hand-model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument("--no-face", action="store_true", help="Disable face landmark extraction")
    parser.add_argument("--no-hands", action="store_true", help="Disable hand landmark extraction")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--visibility", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 0 <= args.visibility <= 1:
        parser.error("seconds must be positive and visibility must be between 0 and 1")
    models = {"Pose": args.model}
    if not args.no_face:
        models["Face"] = args.face_model
    if not args.no_hands:
        models["Hand"] = args.hand_model
    for kind, path in models.items():
        if not path.is_file():
            parser.error(f"{kind} model not found: {path}")
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = (int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata")
    frame_limit = min(math.ceil(args.seconds * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_limit <= 0:
        cap.release()
        raise RuntimeError("Video contains no decodable frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.input.stem.replace(' ', '_')}_first_{args.seconds:g}s"
    if not (args.no_face and args.no_hands):
        prefix += "_pose" + ("_face" if not args.no_face else "") + ("_hands" if not args.no_hands else "")
    paths = {kind: args.output_dir / f"{prefix}_{kind}.{ext}" for kind, ext in (
        ("overlay", "mp4"), ("skeleton", "mp4"), ("landmarks", "csv"),
        ("arrays", "npz"), ("metadata", "json"), ("preview", "jpg"),
    )}
    if not args.no_face:
        paths["face_blendshapes"] = args.output_dir / f"{prefix}_face_blendshapes.csv"
    if not args.no_hands:
        paths["hand_landmarks"] = args.output_dir / f"{prefix}_hand_landmarks.csv"
    writers = [cv2.VideoWriter(str(paths[kind]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
               for kind in ("overlay", "skeleton")]
    if not all(writer.isOpened() for writer in writers):
        for writer in writers:
            writer.release()
        cap.release()
        raise RuntimeError("Cannot create MP4 output")
    normalized = np.full((frame_limit, 33, 5), np.nan, dtype=np.float32)
    world = np.full_like(normalized, np.nan)
    detected = np.zeros(frame_limit, dtype=bool)
    details = FaceHandLandmarkers(
        frame_limit,
        face_model=None if args.no_face else args.face_model,
        hand_model=None if args.no_hands else args.hand_model,
    )
    names = [item.name.lower() for item in PoseLandmark]
    sample_frames = {round(t * fps) for t in (10, 20, 30, 60, 90, 119) if t < args.seconds}
    previews = []
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(args.model), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    started = time.monotonic()
    count = 0
    try:
        with details, mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker, paths["landmarks"].open("w", newline="") as output:
            csv_writer = csv.writer(output)
            csv_writer.writerow([
                "frame_index", "timestamp_s", "pose_detected", "landmark_index", "landmark_name",
                "x_normalized", "y_normalized", "z_normalized", "visibility", "presence",
                "world_x_m", "world_y_m", "world_z_m", "world_visibility", "world_presence",
                "drawn_in_video",
            ])
            for index in range(frame_limit):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video decoding failed at frame {index} before expected clip end")
                timestamp_s = index / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_input = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = round(timestamp_s * 1000)
                result = landmarker.detect_for_video(image_input, timestamp_ms)
                details.process(image_input, timestamp_ms, index, width, height)
                points = {}
                if result.pose_landmarks:
                    detected[index] = True
                    landmarks = result.pose_landmarks[0]
                    world_landmarks = result.pose_world_landmarks[0]
                    normalized[index] = [[p.x, p.y, p.z, p.visibility, p.presence] for p in landmarks]
                    world[index] = [[p.x, p.y, p.z, p.visibility, p.presence] for p in world_landmarks]
                    points = visible_points(landmarks, width, height, args.visibility)
                for landmark_id, name in enumerate(names):
                    values = [*normalized[index, landmark_id], *world[index, landmark_id]]
                    csv_writer.writerow([
                        index, f"{timestamp_s:.6f}", int(detected[index]), landmark_id, name,
                        *[f"{float(v):.7f}" if np.isfinite(v) else "" for v in values],
                        int(landmark_id in points),
                    ])
                skeleton = np.zeros_like(frame)
                for image in (frame, skeleton):
                    draw_skeleton(image, points)
                    details.draw(image)
                    label = f"{timestamp_s:06.2f}s | {'Pose detected' if detected[index] else 'No pose detected'}"
                    if not args.no_face:
                        label += f" | Face: {int(details.arrays['face_detected'][index])}"
                    if not args.no_hands:
                        label += f" | Hands: {int(details.arrays['hand_detected'][index].sum())}"
                    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
                    font_scale = 0.8 * min(1, max(1, width - 48) / max(1, text_width))
                    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
                    cv2.rectangle(image, (12, 12), (min(width - 1, text_width + 36), 65), (0, 0, 0), -1)
                    cv2.putText(image, label, (24, 49), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
                writers[0].write(frame)
                writers[1].write(skeleton)
                if index in sample_frames:
                    previews.append(cv2.resize(frame, (640, 360)))
                count += 1
                if count % round(fps * 10) == 0 or count == frame_limit:
                    elapsed = time.monotonic() - started
                    print(f"Processed {count}/{frame_limit} frames ({count / fps:.1f}s); "
                          f"pose found in {detected[:count].sum()} frames; {count / elapsed:.1f} frames/s", flush=True)
    finally:
        cap.release()
        for writer in writers:
            writer.release()
    np.savez_compressed(
        paths["arrays"], normalized_landmarks=normalized[:count], world_landmarks=world[:count],
        pose_detected=detected[:count], timestamps_s=np.arange(count) / fps,
        landmark_names=np.array(names), coordinate_names=np.array(["x", "y", "z", "visibility", "presence"]),
        **details.export_arrays(count),
    )
    details.export_csv(paths, count, fps)
    if previews:
        while len(previews) % 3:
            previews.append(np.zeros_like(previews[0]))
        cv2.imwrite(str(paths["preview"]), np.vstack([
            np.hstack(previews[i:i + 3]) for i in range(0, len(previews), 3)
        ]))
    metadata = {
        "input": str(args.input.resolve()), "mediapipe_version": mp.__version__,
        "model": str(args.model.resolve()), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "model_url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "fps": fps, "width": width, "height": height, "start_s": 0,
        "requested_end_s_exclusive": args.seconds, "processed_duration_s": count / fps,
        "processed_frames": count, "frames_with_pose": int(detected[:count].sum()),
        "detection_fraction": float(detected[:count].mean()),
        "no_pose_intervals": missing_intervals(detected[:count].tolist(), fps),
        "draw_visibility_and_presence_threshold": args.visibility,
        "landmarkers": details.metadata(count, fps, missing_intervals),
        "notes": [
            "Configured for one speaker: one pose, up to one face and two hands per frame. Models run independently; person association is not implemented.",
            "All source frames are processed, with original timing and resolution preserved. Output videos have no audio.",
            "Undetected frames have blank CSV coordinates and NaN array entries; no interpolation or forward fill is applied.",
            "CSV retains all 33 model estimates when a pose is found; only in-frame points meeting both confidence thresholds are rendered.",
            "Normalized x/y are image-relative; normalized z is model-relative depth, not meters.",
            "World coordinates are model-estimated meters centered at the hip midpoint, not measured camera or room coordinates.",
            "Occluded and off-camera body parts may be unreliable; use visibility and presence for filtering.",
            "Face/hand results have no per-landmark visibility or presence scores; only body drawing uses the visibility threshold.",
            "Face coordinates are normalized model-relative estimates, not room coordinates. Facial transformation matrices map the canonical face to the detected face.",
            "Hand world coordinates are model-estimated meters centered on each hand, independent of pose world coordinates.",
            "Hand slots follow each frame's detection order and are not persistent identities. Handedness labels and scores are raw model predictions on unmirrored input.",
            "Blendshape scores are raw coefficients, not behaviour labels or calibrated probabilities of smiling or brow movement.",
        ],
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {key: metadata[key] for key in ("processed_frames", "processed_duration_s", "frames_with_pose", "detection_fraction")}
    summary["landmarkers"] = {name: {key: value for key, value in item.items() if key in ("enabled", "frames_detected", "detection_fraction")}
                             for name, item in metadata["landmarkers"].items()}
    print(json.dumps(summary, indent=2))
    return metadata


if __name__ == "__main__":
    main()
