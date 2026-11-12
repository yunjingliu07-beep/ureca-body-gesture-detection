"""Combine body gestures and facial expressions into an annotated video.

Runs OpenFace/HMAD offline, merges body/hand/object observations by source time,
and writes a translucent person mask with skeleton, face, gaze and held objects.
Confirmed behaviours appear in the upper-right corner. Frame CSV, event CSV,
and metadata JSON preserve the rules and measurements for manual review.

Default: python main.py (first 30 seconds of the Calculus 1 Lecture 0.1 video).
Teaching-aid detection is off by default; select --teaching-aids on to enable it.
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
from facial_expressions import FACE_LABELS, DIRECT_LABELS, FacialExpressionExtractor, build_face_timeline
from tracking_config import (FaceRules, HeadRules, GazeRules, CameraMovementRules,
                             TeachingAidRules, TrackingRules)
from camera_movement import CAMERA_LABELS, CameraGuard, CameraMovementDetector
from native_tracking import prepare_native, resolve_executable, sha256
from teaching_aids import OBJECT_LABELS, GroundingDinoDetector, TeachingAidTracker


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / (
    "Calculus 1 Lecture 0.1：  Lines, Angle of Inclination, and the Distance Formula [fYyARMqiaag].webm"
)


def load_settings(path=None):
    settings = json.loads(path.read_text()) if path else {}
    allowed = {"body", "face", "head", "gaze", "camera_motion", "objects", "scene", "confirmation_s"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown config sections: {sorted(set(settings) - allowed)}")
    body = BodyRules(**settings.get("body", {}))
    face = TrackingRules(FaceRules(**settings.get("face", {})), HeadRules(**settings.get("head", {})),
                         GazeRules(**settings.get("gaze", {})), CameraMovementRules(**settings.get("camera_motion", {})),
                         TeachingAidRules(**settings.get("objects", {})))
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
    if info.get("smile_level"):
        lines.append("Smile level: " + info["smile_level"])
    held = info.get("held_objects", [])
    if info.get("teaching_aids_enabled") is False:
        lines.append("Teaching aids: disabled")
    elif held:
        lines.append("Teaching aids: " + ", ".join(sorted({o["class"] + " (" + o["hand"] + ")" for o in held})))
    elif "holding_status" in info:
        lines.append("Teaching aids: " + info["holding_status"].replace("_", " "))
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
    """Export gated frame episodes and native-timed intervals/cycles without duplicates."""

    def __init__(self, writer, fps):
        self.writer, self.fps, self.active = writer, fps, {}
        writer.writerow(["behaviour", "start_frame", "end_frame_exclusive", "start_s",
                         "end_s_exclusive", "duration_s", "timing_type", "source",
                         "completion_s", "peak_s", "native_value"])

    def update(self, states, frame_index):
        for label, value in states.items():
            if value is True:
                self.active.setdefault(label, frame_index)
            elif label in self.active:
                self.end(label, frame_index)

    def end(self, label, end):
        start = self.active.pop(label)
        source = ("teaching_aid_heuristic" if label in OBJECT_LABELS else
                  "camera_movement_heuristic" if label in CAMERA_LABELS else
                  "OpenFace_pose_gaze" if label in FACE_LABELS else "MediaPipe_body")
        self.writer.writerow([label, start, end, f"{start/self.fps:.6f}", f"{end/self.fps:.6f}",
                              f"{(end-start)/self.fps:.6f}", "confirmed_posture", source, "", "", ""])

    def annotation(self, record, timing_type, duration_s):
        start, end = max(0, record["start_s"]), min(duration_s, record["end_s"])
        if end <= start:
            return
        self.writer.writerow([record["behaviour"], math.ceil(start*self.fps-1e-8),
                              math.ceil(end*self.fps-1e-8), f"{start:.6f}", f"{end:.6f}", f"{end-start:.6f}",
                              timing_type, "OpenFace_cycle" if timing_type == "completed_cycle" else "HMAD",
                              record.get("completion_s", ""), record.get("peak_s", ""), record.get("native_value", "")])

    def close(self, count):
        for label in list(self.active):
            self.end(label, count)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/behaviours")
    parser.add_argument("--config", type=Path)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--seconds", type=float, default=30, help="First N seconds; default: 30")
    duration.add_argument("--full-video", dest="seconds", action="store_const", const=None)
    parser.add_argument("--pose-model", type=Path, default=PROJECT_DIR / "models/pose_landmarker_full.task")
    parser.add_argument("--hand-model", type=Path, default=PROJECT_DIR / "models/hand_landmarker.task")
    local_openface = PROJECT_DIR / "external/OpenFace/build/bin/FeatureExtraction"
    parser.add_argument("--openface-bin", default=os.environ.get("OPENFACE_BIN", str(local_openface) if local_openface.is_file() else "FeatureExtraction"))
    parser.add_argument("--hmad-dir", type=Path, default=PROJECT_DIR / "external/HMAD")
    local_rscript = PROJECT_DIR / ".runtime/r/bin/Rscript"
    parser.add_argument("--rscript", default=str(local_rscript) if local_rscript.is_file() else "Rscript")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--openface-csv", type=Path, help="Import OpenFace output from a matching 25 fps, zero-start clip")
    parser.add_argument("--hmad-results", type=Path, help="Import canonical HMAD results with matching manifest.json")
    parser.add_argument("--no-face", action="store_true", help="Explicitly disable head/expression/gaze/depth analysis")
    parser.add_argument("--no-hands", action="store_true", help="Hand/object holding and finger pointing become unknown")
    objects = parser.add_mutually_exclusive_group()
    objects.add_argument("--teaching-aids", choices=("on", "off"), default="off",
                         help="Enable or disable teaching-aid detection before analysis; default: off (faster)")
    objects.add_argument("--no-objects", dest="teaching_aids", action="store_const", const="off",
                         help="Compatibility alias for --teaching-aids off")
    parser.add_argument("--objects-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--objects-revision", default="a2bb814dd30d776dcf7e30523b00659f4f141c71", help="HF revision or commit, recorded in metadata")
    parser.add_argument("--objects-device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--teacher-roi", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                        default=(0, 0, 1, 1), help="Fixed normalized crop containing only the teacher")
    parser.add_argument("--check-dependencies", action="store_true", help="Report required tools/models without running analysis")
    parser.add_argument("--view", choices=("overlay", "mask", "skeleton"), default="overlay")
    parser.add_argument("--mask-opacity", type=float, default=.30)
    parser.add_argument("--mask-threshold", type=float, default=.5)
    parser.add_argument("--show-scene", action="store_true")
    return parser


def dependency_report(args):
    import importlib.util
    import shutil
    return {"FeatureExtraction": shutil.which(str(args.openface_bin)), "Rscript": shutil.which(args.rscript),
            "ffmpeg": shutil.which(args.ffmpeg), "HMAD": (args.hmad_dir / "sourceR").is_dir(),
            "pose_model": args.pose_model.is_file(), "hand_model": args.hand_model.is_file(),
            "teaching_aids": args.teaching_aids,
            "object_dependencies_required": args.teaching_aids == "on",
            **{name: importlib.util.find_spec(name) is not None for name in ("mediapipe", "torch", "transformers")}}


def scan_cut_times(path, duration_s, fps, roi, threshold):
    cap = cv2.VideoCapture(str(path))
    previous, cuts, count = None, [], 0
    x0, y0, x1, y1 = roi
    try:
        while count/fps < duration_s - 1e-9:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame[y0:y1, x0:x1], (160, 90)), cv2.COLOR_BGR2GRAY)
            if previous is not None and float(np.mean(cv2.absdiff(gray, previous))/255) >= threshold:
                cuts.append(count/fps)
            previous = gray
            count += 1
    finally:
        cap.release()
    return cuts


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check_dependencies:
        report = dependency_report(args)
        print(json.dumps(report, indent=2))
        return report
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite")
    if not 0 <= args.mask_opacity <= 1 or not 0 <= args.mask_threshold <= 1:
        parser.error("Invalid mask settings")
    if not all(math.isfinite(v) for v in args.teacher_roi) or not (0 <= args.teacher_roi[0] < args.teacher_roi[2] <= 1 and 0 <= args.teacher_roi[1] < args.teacher_roi[3] <= 1):
        parser.error("--teacher-roi must satisfy 0 <= X0 < X1 <= 1 and 0 <= Y0 < Y1 <= 1")
    objects_enabled = args.teaching_aids == "on"
    if objects_enabled and args.no_hands:
        parser.error("--teaching-aids on requires hand tracking; remove --no-hands or use --teaching-aids off")
    print(f"Teaching-aid detection: {args.teaching_aids.upper()}", flush=True)
    try:
        body_rules, tracking, scene, confirmation = load_settings(args.config)
        if not args.no_face:
            if args.hmad_results and not args.openface_csv:
                raise ValueError("--hmad-results requires its matching --openface-csv")
            if not args.openface_csv:
                resolve_executable(args.openface_bin, "OpenFace FeatureExtraction")
                resolve_executable(args.ffmpeg, "ffmpeg")
            if not args.hmad_results:
                resolve_executable(args.rscript, "Rscript")
                if not (args.hmad_dir / "sourceR").is_dir():
                    raise ValueError("HMAD is missing; run setup_tracking.py --download-hmad")
        if objects_enabled:
            import importlib.util
            if any(importlib.util.find_spec(name) is None for name in ("torch", "transformers")):
                raise ValueError("Object detection dependencies missing; install requirements-objects.txt")
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    models = {"pose": args.pose_model}
    if not args.no_hands:
        models["hands"] = args.hand_model
    for name, path in models.items():
        if not path.is_file():
            parser.error(f"Missing {name} model: {path}; see models/README.md")
    if not args.input.is_file():
        parser.error(f"Input not found: {args.input}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import mediapipe as mp
    native_manifest, face_assets = None, None
    started = time.monotonic()
    with ExitStack() as stack:
        cap = cv2.VideoCapture(str(args.input))
        stack.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.input}")
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or total <= 0 or min(width, height) < 32:
            raise RuntimeError("Invalid video metadata / frame dimensions")
        limit = min(total, math.ceil(args.seconds*fps)) if args.seconds is not None else total
        duration_s = limit/fps
        roi = tuple(round(v*s) for v, s in zip(args.teacher_roi, (width, height, width, height)))
        x0, y0, x1, y1 = roi
        if min(x1-x0, y1-y0) < 32:
            raise ValueError("Teacher ROI must be at least 32 x 32 pixels")
        roi_width, roi_height = x1-x0, y1-y0
        cut_times = scan_cut_times(args.input, duration_s, fps, roi, tracking.camera_motion.cut_difference)
        if not args.no_face:
            csv_path, hmad_directory, native_manifest = prepare_native(args, args.output_dir / "native", duration_s, roi)
            face_assets = build_face_timeline(csv_path, hmad_directory, args.output_dir / "face_analysis.jsonl", tracking, cut_times)
        paths = {key: args.output_dir / f"{args.input.stem}_{key}.{extension}" for key, extension in
                 (("behaviours", "mp4"), ("frames", "csv"), ("events", "csv"), ("metadata", "json"), ("preview", "jpg"))}
        video = cv2.VideoWriter(str(paths["behaviours"]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        stack.callback(video.release)
        if not video.isOpened():
            raise RuntimeError("Cannot create output MP4")
        body = stack.enter_context(BodyGestureExtractor(args.pose_model, roi_width, roi_height, body_rules, scene,
                                                       None if args.no_hands else args.hand_model))
        face = None if face_assets is None else stack.enter_context(FacialExpressionExtractor(face_assets["timeline"], face_assets["cycles"]))
        objects, object_detector = None, None
        if objects_enabled:
            object_detector = stack.enter_context(GroundingDinoDetector(args.objects_model, args.objects_revision,
                args.objects_device, tracking.objects, str(PROJECT_DIR / "models/hf-cache")))
            objects = TeachingAidTracker(object_detector, tracking.objects, body_rules.visibility)
        motion = CameraMovementDetector(tracking.camera_motion, body_rules.visibility)
        guard = CameraGuard(tracking.camera_motion)
        frame_file = stack.enter_context(paths["frames"].open("w", newline=""))
        event_file = stack.enter_context(paths["events"].open("w", newline=""))
        labels = (*BODY_LABELS, *FACE_LABELS, *CAMERA_LABELS, *OBJECT_LABELS)
        columns = ["frame_index", "timestamp_s", "pose_detected", "face_detected",
                   *["raw_"+name for name in labels], *labels, "smile_level", "holding_status", "held_object_types",
                   "measurements_json", "face_scores_json"]
        writer = csv.DictWriter(frame_file, fieldnames=columns)
        writer.writeheader()
        events = EventWriter(csv.writer(event_file), fps)
        gate = TemporalGate(confirmation)
        preview_quality, last_ms = None, -1
        count = pose_count = face_count = 0
        coverage = {"camera_motion": 0, "objects": 0}
        while count < limit:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video decoding failed at frame {count}; outputs are incomplete")
            t, end = count/fps, (count+1)/fps
            crop = frame[y0:y1, x0:x1]
            timestamp_ms = max(last_ms+1, round(t*1000))
            last_ms = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            raw, info, mask, pose, _ = body.process(image, timestamp_ms, t)
            camera_info = guard.update(crop, t, mask)
            face_raw, face_info, scores = (dict.fromkeys(FACE_LABELS), {"face_detected": False}, {}) if face is None else face.process(t, end)
            if camera_info["camera_cut"]:
                gate = TemporalGate(confirmation)
                body.detector._reset()
                for history in body.detector.hand_history.values():
                    history.clear()
                raw = dict.fromkeys(BODY_LABELS)
                face_raw = dict.fromkeys(FACE_LABELS)
                face_info.update(face_valid=False, head_valid=False, face_reason="camera_cut")
                if objects:
                    objects.reset()
            camera_raw, camera_measurements = motion.update(t, face_info.get("head_depth_mm") if face_info.get("head_valid") else None,
                pose, roi_width, roi_height, info.get("torso_yaw_deg"), camera_info["camera_stable"])
            raw.update(face_raw)
            raw.update(camera_raw)
            info.update(face_info)
            info.update(camera_info)
            info.update(camera_measurements)
            object_raw, object_states, object_info = (dict.fromkeys(OBJECT_LABELS), dict.fromkeys(OBJECT_LABELS),
                {"holding_status": "unknown", "held_objects": [], "object_reason": "disabled_by_option"}) if objects is None else objects.process(crop, t, pose, body.hand_observations)
            object_info["teaching_aids_enabled"] = objects_enabled
            raw.update(object_raw)
            info.update(object_info)
            states = gate.update({k: v for k, v in raw.items() if k not in DIRECT_LABELS and k not in OBJECT_LABELS}, t)
            states.update({k: raw[k] for k in DIRECT_LABELS})
            states.update(object_states)
            output = frame.copy() if args.view == "overlay" else np.zeros_like(frame)
            output[y0:y1, x0:x1] = render_mask(crop, mask, args.view, args.mask_opacity, args.mask_threshold)
            rendered_crop = output[y0:y1, x0:x1]
            if args.show_scene:
                draw_scene(rendered_crop, scene)
            body.draw(rendered_crop)
            if face is not None and face_info.get("face_valid"):
                face.draw(rendered_crop)
            if objects:
                objects.draw(rendered_crop)
            draw_panel(output, states, info, t)
            video.write(output)
            quality = (bool(info["pose_detected"]), any(v is True for v in states.values()),
                       len(body.points), bool(info.get("face_valid")))
            if preview_quality is None or quality > preview_quality:
                if not cv2.imwrite(str(paths["preview"]), output):
                    raise RuntimeError("Cannot write preview")
                preview_quality = quality
            row = {"frame_index": count, "timestamp_s": f"{t:.6f}", "pose_detected": int(info["pose_detected"]),
                   "face_detected": int(info.get("face_detected", False)), "smile_level": info.get("smile_level") or "",
                   "holding_status": info["holding_status"], "held_object_types": ",".join(sorted({o["class"] for o in info["held_objects"]})),
                   "measurements_json": json.dumps(info, allow_nan=False), "face_scores_json": json.dumps(scores, allow_nan=False)}
            for name in labels:
                row["raw_"+name] = "" if raw[name] is None else int(raw[name])
                row[name] = "" if states[name] is None else int(states[name])
            writer.writerow(row)
            events.update({k: v for k, v in states.items() if k not in DIRECT_LABELS}, count)
            pose_count += int(info["pose_detected"])
            face_count += int(info.get("face_valid", False))
            coverage["camera_motion"] += int(camera_raw["moving_toward_camera"] is not None)
            coverage["objects"] += int(object_raw["holding_teaching_aid"] is not None)
            count += 1
            if count % max(1, round(fps*10)) == 0:
                print(f"Processed {count}/{limit} frames ({count/fps:.1f}s)", flush=True)
        events.close(count)
        if face_assets:
            for key, timing in (("annotations", "hmad_interval"), ("cycles", "completed_cycle")):
                with face_assets[key].open() as handle:
                    for line in handle:
                        events.annotation(json.loads(line), timing, duration_s)
    metadata = {"input": str(args.input.resolve()), "fps": fps, "width": width, "height": height,
        "processed_frames": count, "duration_s": count/fps, "teacher_roi_px": roi,
        "frames_with_pose": pose_count, "frames_with_face": face_count, "mediapipe_version": mp.__version__,
        "body": asdict(body_rules), "tracking": asdict(tracking), "scene": scene, "confirmation_s": confirmation,
        "native": native_manifest, "objects": None if object_detector is None else object_detector.metadata,
        "teaching_aid_detection": {"requested": args.teaching_aids, "enabled": objects_enabled},
        "coverage": {**coverage, "face_analysis": None if face_assets is None else face_assets["coverage"]},
        "ignored_zero_duration_intervals": None if face_assets is None else face_assets["ignored_zero_duration_intervals"],
        "cut_times_s": cut_times, "processing_seconds": time.monotonic()-started,
        "models": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in models.items()},
        "notes": ["One selected teacher; use --teacher-roi for a fixed single-speaker crop.",
                  "Fixed camera and constant source frame rate; silent output MP4.",
                  "Face backend is OpenFace/HMAD with no legacy facial fallback.",
                  "Camera-relative movement and hand/object contact are unvalidated project heuristics.",
                  "No target object observed does not prove an empty hand; object classes are configured.",
                  "Native HMAD intervals and completed cycles bypass the posture confirmation gate.",
                  "Head/gaze directions are image directions; reference gaze requires explicit calibration."],
        "outputs": {name: str(path.resolve()) for name, path in paths.items()}}
    paths["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(f"Saved {count} frames ({count/fps:.2f}s): {paths['behaviours']}")
    return metadata


if __name__ == "__main__":
    main()
