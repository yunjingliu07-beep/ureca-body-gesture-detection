"""Combine body gestures and facial expressions into an annotated video.

Reads a video once, imports BodyGestureExtractor and FacialExpressionExtractor,
and writes a translucent person mask with skeleton/face/hand landmarks on top.
Confirmed behaviours appear in the upper-right corner. Frame CSV, event CSV,
and metadata JSON preserve the rules and measurements for manual review.

Default: python main.py (first 30 seconds of the Calculus 1 Lecture 0.1 video).
Scene-dependent rules: add --config classroom_config.example.json after editing
its example coordinates and orientation angles for your own fixed camera.
"""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ureca-matplotlib")

import cv2
import numpy as np

from behaviour_common import TemporalGate
from body_gestures import BODY_LABELS, DEFAULT_SCENE, BodyGestureExtractor, BodyRules, validate_scene
from facial_expressions import FACE_LABELS, FaceRules, FacialExpressionExtractor


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / (
    "Calculus 1 Lecture 0.1：  Lines, Angle of Inclination, and the Distance Formula [fYyARMqiaag].webm"
)


def load_settings(path=None):
    settings = json.loads(path.read_text()) if path else {}
    if set(settings) - {"body", "face", "scene", "confirmation_s"}:
        raise ValueError("Config accepts only body, face, scene, and confirmation_s")
    body = BodyRules(**settings.get("body", {}))
    face = FaceRules(**settings.get("face", {}))
    scene = {**DEFAULT_SCENE, **settings.get("scene", {})}
    validate_scene(scene)
    confirmation = settings.get("confirmation_s", 0.25)
    TemporalGate(confirmation)
    return body, face, scene, confirmation


def render_mask(frame, mask, view="overlay", opacity=0.30, threshold=0.5):
    """Keep source video by default; tint only pixels inside the person mask."""
    output = frame.copy() if view == "overlay" else np.zeros_like(frame)
    if mask is not None and view != "skeleton":
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        foreground = np.isfinite(mask) & (mask >= threshold)
        color = np.array([180, 90, 40], dtype=np.float32)  # BGR blue silhouette.
        if view == "mask":
            output[foreground] = color.astype(np.uint8)
        else:
            output[foreground] = ((1 - opacity) * output[foreground] + opacity * color).astype(np.uint8)
    return output


def draw_scene(frame, scene):
    size = np.array([frame.shape[1] - 1, frame.shape[0] - 1])
    for key, color in (("zones", (180, 180, 180)), ("groups", (220, 130, 255)), ("desks", (255, 180, 80))):
        for region in scene.get(key, []):
            polygon = np.rint(np.asarray(region["polygon"]) * size).astype(np.int32)
            cv2.polylines(frame, [polygon], True, color, 1, cv2.LINE_AA)
            cv2.putText(frame, region["name"], tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_panel(frame, states, info, timestamp_s):
    """Fit a readable status panel to the upper-right, with overflow in CSV."""
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.25, min(0.65, width / 1900))
    line_h = max(13, round(31 * scale / 0.65))
    panel_w = min(width - 8, max(150, round(width * 0.38)))
    x, y = max(0, width - panel_w - 8), 8
    lines = [f"Behaviours | {timestamp_s:.2f}s"]
    active = [label for label, value in states.items() if value is True]
    for label in active:
        text = label.replace("_", " ")
        direction = info.get("directions", {}).get(label)
        if direction:
            text += " (" + direction.replace("image_", "").replace("_", " ") + ")"
        # Wrap full words rather than drawing through the edge of the video.
        words, current = text.split(), ""
        for word in words:
            candidate = (current + " " + word).strip()
            if current and cv2.getTextSize(candidate, font, scale, 1)[0][0] > panel_w - 18:
                lines.append(current)
                current = "  " + word
            else:
                current = candidate
        if current:
            lines.append(current)
    if not active:
        lines.append("No confirmed behaviour")
    unknown = sum(v is None for v in states.values())
    lines += [f"Unknown rules: {unknown} (CSV)"]
    if not info.get("pose_detected"):
        lines.append("Body not detected")
    if info.get("zone"):
        lines.append("Zone: " + info["zone"])
    if info.get("zone_crossing"):
        lines.append(info["zone_crossing"])
    maximum = max(1, (height - 24) // line_h)
    if len(lines) > maximum:
        lines = lines[:maximum]
        lines[-1] = "More labels in frame CSV"
    bottom = min(height - 1, y + len(lines) * line_h + 10)
    frame[y:bottom, x:width - 8] = (frame[y:bottom, x:width - 8].astype(float) * 0.18).astype(np.uint8)
    for index, line in enumerate(lines):
        # Long annotation names also need to fit the panel.
        text_width = cv2.getTextSize(line, font, scale, 1)[0][0]
        fitted = scale * min(1, (panel_w - 18) / max(1, text_width))
        cv2.putText(frame, line, (x + 8, y + (index + 1) * line_h), font,
                    fitted, (255, 255, 255), 1, cv2.LINE_AA)


class EventWriter:
    """Write confirmed episodes as they end, without holding a clip in memory."""

    def __init__(self, writer, fps):
        self.writer, self.fps, self.active = writer, fps, {}
        writer.writerow(["behaviour", "start_frame", "end_frame_exclusive", "start_s",
                         "end_s_exclusive", "duration_s"])

    def update(self, states, frame_index):
        for label, value in states.items():
            if value is True:
                self.active.setdefault(label, frame_index)
            elif label in self.active:
                self.end(label, frame_index)

    def end(self, label, end):
        start = self.active.pop(label)
        self.writer.writerow([label, start, end, f"{start / self.fps:.6f}",
                              f"{end / self.fps:.6f}", f"{(end - start) / self.fps:.6f}"])

    def close(self, count):
        for label in list(self.active):
            self.end(label, count)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Source video; default: Calculus 1 Lecture 0.1 in the project folder")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/behaviours")
    parser.add_argument("--config", type=Path)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--seconds", type=float, default=30, help="Process the first N seconds; default: 30")
    duration.add_argument("--full-video", dest="seconds", action="store_const", const=None,
                          help="Process the entire video instead of the default 30-second test")
    parser.add_argument("--pose-model", type=Path, default=PROJECT_DIR / "models/pose_landmarker_full.task")
    parser.add_argument("--face-model", type=Path, default=PROJECT_DIR / "models/face_landmarker.task")
    parser.add_argument("--hand-model", type=Path, default=PROJECT_DIR / "models/hand_landmarker.task")
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--no-hands", action="store_true", help="Finger pointing stays unknown; coarse arm direction remains available")
    parser.add_argument("--view", choices=("overlay", "mask", "skeleton"), default="overlay")
    parser.add_argument("--mask-opacity", type=float, default=0.30)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--show-scene", action="store_true", help="Draw annotated zone, group and desk polygons")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be a positive finite number")
    if not 0 <= args.mask_opacity <= 1 or not 0 <= args.mask_threshold <= 1:
        parser.error("Mask opacity and threshold must be between 0 and 1")
    try:
        body_rules, face_rules, scene, confirmation = load_settings(args.config)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    models = {"pose": args.pose_model}
    if not args.no_face:
        models["face"] = args.face_model
    if not args.no_hands:
        models["hands"] = args.hand_model
    for name, path in models.items():
        if not path.is_file():
            parser.error(f"Missing {name} model: {path}; see models/README.md")
    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")
    import mediapipe as mp
    with ExitStack() as stack:
        cap = cv2.VideoCapture(str(args.input))
        stack.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.input}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width < 32 or height < 32:
            raise RuntimeError("Invalid FPS or frame dimensions (minimum 32 x 32)")
        limit = math.ceil(args.seconds * fps) if args.seconds is not None else None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {key: args.output_dir / f"{args.input.stem}_{key}.{extension}" for key, extension in
                 (("behaviours", "mp4"), ("frames", "csv"), ("events", "csv"), ("metadata", "json"), ("preview", "jpg"))}
        # Prevent accidental replacement of either the source or a previous analysis.
        for path in paths.values():
            if path.exists() or path.resolve() == args.input.resolve():
                parser.error(f"Output already exists: {path}. Choose a different --output-dir.")
        video = cv2.VideoWriter(str(paths["behaviours"]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        stack.callback(video.release)
        if not video.isOpened():
            raise RuntimeError("Cannot create MP4 output")
        body = stack.enter_context(BodyGestureExtractor(args.pose_model, width, height, body_rules, scene,
                                                       None if args.no_hands else args.hand_model))
        face = None if args.no_face else stack.enter_context(FacialExpressionExtractor(args.face_model, face_rules))
        frame_file = stack.enter_context(paths["frames"].open("w", newline=""))
        event_file = stack.enter_context(paths["events"].open("w", newline=""))
        labels = (*BODY_LABELS, *FACE_LABELS)
        columns = ["frame_index", "timestamp_s", "pose_detected", "face_detected",
                   *["raw_" + name for name in labels], *labels, "measurements_json", "face_scores_json"]
        writer = csv.DictWriter(frame_file, fieldnames=columns)
        writer.writeheader()
        events = EventWriter(csv.writer(event_file), fps)
        gate = TemporalGate(confirmation)
        count, pose_count, face_count = 0, 0, 0
        preview_quality = None
        last_ms = -1
        started = time.monotonic()
        while limit is None or count < limit:
            ok, frame = cap.read()
            if not ok:
                expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if expected > 0 and count < expected and (limit is None or count < limit):
                    raise RuntimeError(f"Video decoding failed at frame {count}; outputs are incomplete")
                break
            t = count / fps
            timestamp_ms = max(last_ms + 1, round(t * 1000))
            last_ms = timestamp_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_states, face_info, scores = (dict.fromkeys(FACE_LABELS), {}, {}) if face is None else face.process(image, timestamp_ms, width, height)
            raw, info, mask, _, _ = body.process(image, timestamp_ms, t,
                                                face_landmarks=None if face is None else face.landmarks)
            raw.update(face_states)
            info.update(face_info)
            face_detected = bool(scores)
            states = gate.update(raw, t)
            output = render_mask(frame, mask, args.view, args.mask_opacity, args.mask_threshold)
            if args.show_scene:
                draw_scene(output, scene)
            body.draw(output)
            if face is not None:
                face.draw(output)
            draw_panel(output, states, info, t)
            video.write(output)
            # Prefer a visible skeleton with confirmed labels over opening titles.
            quality = (bool(info["pose_detected"]), any(v is True for v in states.values()),
                       len(body.points), face_detected)
            if preview_quality is None or quality > preview_quality:
                if not cv2.imwrite(str(paths["preview"]), output):
                    raise RuntimeError("Cannot write preview image")
                preview_quality = quality
            row = {"frame_index": count, "timestamp_s": f"{t:.6f}",
                   "pose_detected": int(info["pose_detected"]), "face_detected": int(face_detected),
                   "measurements_json": json.dumps(info, allow_nan=False),
                   "face_scores_json": json.dumps(scores, allow_nan=False)}
            for name in labels:
                row["raw_" + name] = "" if raw[name] is None else int(raw[name])
                row[name] = "" if states[name] is None else int(states[name])
            writer.writerow(row)
            events.update(states, count)
            pose_count += int(info["pose_detected"])
            face_count += int(face_detected)
            count += 1
            if count % max(1, round(fps * 10)) == 0:
                print(f"Processed {count} frames ({count / fps:.1f}s), {count / (time.monotonic() - started):.1f} frames/s", flush=True)
        if count == 0:
            raise RuntimeError("No frames decoded; outputs are incomplete")
        events.close(count)
    metadata = {
        "input": str(args.input.resolve()), "fps": fps, "width": width, "height": height,
        "processed_frames": count, "duration_s": count / fps,
        "frames_with_pose": pose_count, "frames_with_face": face_count,
        "mediapipe_version": mp.__version__, "view": args.view,
        "mask_opacity": args.mask_opacity, "mask_threshold": args.mask_threshold,
        "body": asdict(body_rules), "face": asdict(face_rules), "scene": scene,
        "confirmation_s": confirmation,
        "models": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in models.items()},
        "notes": ["One visible speaker; no active-speaker identification or persistent person tracking.",
                  "Fixed-camera, constant-frame-rate video assumed; scene geometry is image-relative, not room-calibrated.",
                  "Silent MP4; original audio is not copied.",
                  "CSV: 1=true, 0=false, blank=unknown. Raw columns precede temporal confirmation.",
                  "Events describe confirmed video labels; their starts exclude the initial confirmation delay.",
                  "Rules are reviewable heuristics, not validated activity or emotion recognition.",
                  "Back-of-classroom camera: default class yaw is 0 (toward camera), board yaw is 180 (away).",
                  "Facing labels use torso yaw; turning labels use face-skeleton center relative to shoulder-midpoint neck, in image left/right coordinates.",
                  "Face and hand models run independently; person association is not implemented."],
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved {count} frames ({count / fps:.2f}s): {paths['behaviours']}")
    return metadata


if __name__ == "__main__":
    main()
