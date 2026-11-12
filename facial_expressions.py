"""Adapt native OpenFace/HMAD outputs into head, expression, gaze and blink labels.

The face branch consumes a 25 fps offline analysis. It never falls back to
MediaPipe facial blendshapes or shoulder-relative head position.
"""

from bisect import bisect_right
from collections import deque
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from tracking_config import FaceRules, GazeRules, HeadRules, TrackingRules


HEAD_LABELS = ("turning", "turning_left", "turning_right", "head_nodding", "head_shaking")
EXPRESSION_LABELS = ("smiling", "laughing_smile", "brow_raising", "brow_lowering")
GAZE_LABELS = ("blinking", "gaze_left", "gaze_right", "gaze_up", "gaze_down",
               "gaze_toward_camera", "gaze_toward_reference", "gaze_away_reference")
FACE_LABELS = (*HEAD_LABELS, *EXPRESSION_LABELS, *GAZE_LABELS)
NATIVE_LABELS = (*EXPRESSION_LABELS, "blinking", "gaze_toward_reference", "gaze_away_reference")
DIRECT_LABELS = (*NATIVE_LABELS, "head_nodding", "head_shaking")
AU_NUMBERS = (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


class IntervalTable:
    """Validated half-open intervals; ambiguous overlaps are rejected."""

    def __init__(self, path=None, allowed=()):
        self.rows = []
        self.zero_duration_rows = 0
        if path is not None:
            with Path(path).open(newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if set(reader.fieldnames or []) != {"start_s", "end_s", "value"}:
                    raise ValueError(f"Invalid HMAD adapter table schema: {path}")
                for row in reader:
                    a, b = number(row["start_s"]), number(row["end_s"])
                    value = row["value"]
                    if a is None or b is None or a < 0 or b < a or value not in allowed:
                        raise ValueError(f"Invalid HMAD interval: {row}")
                    if a == b:
                        # Upstream EBMAD quality boundaries may round to zero width.
                        self.zero_duration_rows += 1
                        continue
                    self.rows.append((a, b, value))
        self.rows.sort()
        if any(self.rows[i][0] < self.rows[i-1][1] - 1e-7 for i in range(1, len(self.rows))):
            raise ValueError(f"Overlapping HMAD intervals: {path}")
        self.starts = [r[0] for r in self.rows]

    def value(self, t):
        i = bisect_right(self.starts, t + 1e-9) - 1
        return self.rows[i][2] if i >= 0 and t < self.rows[i][1] - 1e-9 else None


class HMADTables:
    def __init__(self, directory):
        directory = Path(directory)
        with (directory / "status.tsv").open() as handle:
            self.status = {r["component"]: r["status"] for r in csv.DictReader(handle, delimiter="\t")}
        self.tables = {}
        for name, component, allowed in (
            ("smile", "smile", ("S0", "S1", "S2", "S3", "S4", "X", "NA")),
            ("brow_raising", "brows", tuple("ABCDENX") + ("NA",)),
            ("brow_lowering", "brows", tuple("ABCDENX") + ("NA",)),
            ("blinking", "bagmad", ("blink", "no blink", "X", "NA")),
            ("gaze", "bagmad", ("eye contact", "no eye contact", "X", "NA")),
        ):
            specific = "bagmad_blinks" if name == "blinking" else "bagmad_gaze"
            if component == "bagmad" and specific in self.status:
                component = specific
            path = directory / f"{name}.tsv"
            if self.status.get(component) == "ok" and not path.is_file():
                raise ValueError(f"HMAD claims {component} succeeded but {path} is missing")
            self.tables[name] = IntervalTable(path if self.status.get(component) == "ok" else None, allowed)

    def sample(self, t):
        return {name: table.value(t) for name, table in self.tables.items()}


def classify_expressions(smile_level=None, brow_raise_class=None, brow_lower_class=None, valid=True):
    states = dict.fromkeys(EXPRESSION_LABELS)
    if valid:
        if smile_level in ("S0", "S1", "S2", "S3", "S4"):
            states.update(smiling=smile_level != "S0", laughing_smile=smile_level == "S4")
        for label, value in (("brow_raising", brow_raise_class), ("brow_lowering", brow_lower_class)):
            if value in ("A", "B", "C", "D", "E"):
                states[label] = True
            elif value == "N":
                states[label] = False
    return states


def classify_observation(row, hmad, rules=None):
    rules = rules or TrackingRules()
    quality = row.get("success") == 1 and row.get("confidence") is not None and row["confidence"] > rules.face.confidence
    states = dict.fromkeys(FACE_LABELS)
    info = {"face_detected": bool(row.get("success") == 1), "face_valid": bool(quality),
            "head_valid": False, "gaze_valid": False, "expression_valid": False,
            "face_reason": "valid" if quality else "failed_or_low_confidence",
            "smile_level": None, "bagmad_gaze_state": None}
    if not quality:
        return states, info
    head_valid = all(row.get(k) is not None for k in ("pose_Rx", "pose_Ry", "pose_Rz"))
    if head_valid:
        yaw = math.degrees(row["pose_Ry"]) * rules.head.yaw_sign
        pitch = math.degrees(row["pose_Rx"]) * rules.head.pitch_up_sign
        left, right = yaw < -rules.head.turn_yaw_deg - 1e-8, yaw > rules.head.turn_yaw_deg + 1e-8
        states.update(turning_left=left, turning_right=right, turning=left or right,
                      head_nodding=False, head_shaking=False)
        info.update(head_valid=True, yaw_image_deg=yaw, pitch_up_deg=pitch,
                    roll_deg=math.degrees(row["pose_Rz"]), head_depth_mm=row.get("pose_Tz"))
    au_valid = all(row.get(f"AU{i:02d}_r") is not None for i in AU_NUMBERS)
    states.update(classify_expressions(hmad.get("smile"), hmad.get("brow_raising"), hmad.get("brow_lowering")))
    if not au_valid:
        states["smiling"] = states["laughing_smile"] = None
    landmarks_valid = all(row.get(f"{axis}_{i}") is not None for axis in "xy" for i in range(68))
    if not landmarks_valid:
        states["brow_raising"] = states["brow_lowering"] = None
    info["expression_valid"] = any(states[k] is not None for k in EXPRESSION_LABELS)
    if states["smiling"] is not None:
        info["smile_level"] = hmad["smile"]
    info["brow_raise_class"], info["brow_lower_class"] = hmad.get("brow_raising"), hmad.get("brow_lowering")
    blink = hmad.get("blinking")
    if row.get("AU45_r") is not None and blink in ("blink", "no blink"):
        states["blinking"] = blink == "blink"
    vectors_valid = all(row.get(f"gaze_{eye}_{axis}") is not None for eye in (0, 1) for axis in "xyz")
    if vectors_valid:
        vectors_valid = all(.5 < sum(row[f"gaze_{eye}_{axis}"]**2 for axis in "xyz") < 1.5 for eye in (0, 1))
    gaze_valid = vectors_valid and all(row.get(k) is not None for k in ("gaze_angle_x", "gaze_angle_y"))
    if gaze_valid and states["blinking"] is False:
        x = math.degrees(row["gaze_angle_x"]) * rules.gaze.horizontal_sign
        y = math.degrees(row["gaze_angle_y"]) * rules.gaze.vertical_sign
        states.update(gaze_left=x < -rules.gaze.horizontal_deg, gaze_right=x > rules.gaze.horizontal_deg,
                      gaze_up=y < -rules.gaze.vertical_deg, gaze_down=y > rules.gaze.vertical_deg,
                      gaze_toward_camera=abs(x) <= rules.gaze.horizontal_deg and abs(y) <= rules.gaze.vertical_deg)
        info.update(gaze_valid=True, gaze_right_deg=x, gaze_down_deg=y)
        if hmad.get("gaze") in ("eye contact", "no eye contact"):
            info["bagmad_gaze_state"] = hmad["gaze"]
            if rules.gaze.reference_validated:
                states["gaze_toward_reference"] = hmad["gaze"] == "eye contact"
                states["gaze_away_reference"] = hmad["gaze"] == "no eye contact"
    info["gaze_reason"] = "valid" if info["gaze_valid"] else "missing_eye_data_blink_or_bagmad"
    return states, info


class CycleDetector:
    """A hysteretic excursion/return detector; each completion is emitted once."""

    def __init__(self, label, rules=None):
        self.label, self.rules = label, rules or HeadRules()
        self.reset()

    def reset(self):
        self.start = self.peak = self.last = None
        self.direction = 0
        self.previous_raw = self.unwrapped = None

    def update(self, t, angle):
        if angle is None:
            self.reset()
            return None
        if self.last is not None and t - self.last[0] > .25:
            self.reset()
        if self.previous_raw is None:
            value = angle
        else:
            value = self.unwrapped + (angle - self.previous_raw + 180) % 360 - 180
        self.previous_raw, self.unwrapped = angle, value
        point = (t, value)
        if self.start is None:
            self.start = self.peak = point
        if t - self.start[0] > self.rules.cycle_max_s:
            self.start = self.peak = self.last or point
            self.direction = 0
        delta = value - self.start[1]
        if not self.direction:
            if abs(delta) >= self.rules.reversal_deg:
                self.direction = 1 if delta > 0 else -1
                self.peak = point
            elif abs(value - self.peak[1]) > 0:
                self.peak = point
        elif self.direction * (value - self.peak[1]) > 0:
            self.peak = point
        if self.direction:
            excursion = self.direction * (self.peak[1] - self.start[1])
            returned = self.direction * (self.peak[1] - value)
            elapsed = t - self.start[0]
            if returned >= self.rules.reversal_deg and abs(delta) <= self.rules.cycle_return_deg:
                event = None
                if excursion >= self.rules.cycle_excursion_deg and self.rules.cycle_min_s <= elapsed <= self.rules.cycle_max_s:
                    event = {"behaviour": self.label, "start_s": self.start[0], "peak_s": self.peak[0],
                             "completion_s": t, "end_s": t, "excursion_deg": excursion}
                self.start = self.peak = point
                self.direction = 0
                self.last = point
                return event
        self.last = point
        return None


def iter_openface(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        reader.fieldnames = [v.strip() for v in reader.fieldnames]
        for row in reader:
            yield {k: number(v) for k, v in row.items()}


def build_face_timeline(csv_path, hmad_directory, output, rules=None, cut_times=()):
    """Stream native measurements to JSONL; retain only three landmark frames."""
    rules = rules or TrackingRules()
    tables = HMADTables(hmad_directory)
    output = Path(output)
    cycles_path = output.with_name("head_cycles.jsonl")
    detectors = {"head_nodding": CycleDetector("head_nodding", rules.head),
                 "head_shaking": CycleDetector("head_shaking", rules.head)}
    counts = {"face_valid": 0, "head_valid": 0, "gaze_valid": 0, "expression_valid": 0}
    valid_spans = {label: [] for label in NATIVE_LABELS}
    cut_times = iter(sorted(cut_times))
    next_cut = next(cut_times, math.inf)
    pending_cut = False
    rows = iter(iter_openface(csv_path))
    previous, current = None, next(rows, None)
    with output.open("w") as handle, cycles_path.open("w") as cycle_handle:
        while current is not None:
            following = next(rows, None)
            t = current["timestamp"]
            while next_cut < t + .04 - 1e-7:
                pending_cut = True
                next_cut = next(cut_times, math.inf)
            native = tables.sample(t)
            states, info = classify_observation(current, native, rules)
            info["head_cycle_lookahead_s"] = 0.0
            if pending_cut:
                states = dict.fromkeys(FACE_LABELS)
                info.update(face_valid=False, head_valid=False, gaze_valid=False,
                            expression_valid=False, face_reason="camera_cut")
                pending_cut = False
            for label, key in (("head_nodding", "pose_Rx"), ("head_shaking", "pose_Ry")):
                angle = None
                if info["head_valid"]:
                    samples = [r.get(key) for r in (previous, current, following) if r is not None]
                    all_good = len(samples) == 3 and all(s is not None for s in samples)
                    all_good = all_good and all(r.get("success") == 1 and (r.get("confidence") or 0) > rules.face.confidence
                                                for r in (previous, current, following))
                    angle = math.degrees(float(np.median(np.unwrap(samples)))) if all_good else math.degrees(current[key])
                    if all_good:
                        info["head_cycle_lookahead_s"] = .04
                event = detectors[label].update(t, angle)
                if event:
                    cycle_handle.write(json.dumps(event, allow_nan=False) + "\n")
            for key in counts:
                counts[key] += int(info[key])
            for label in NATIVE_LABELS:
                if states[label] is not None:
                    spans = valid_spans[label]
                    if spans and abs(spans[-1][1] - t) < 1e-6:
                        spans[-1][1] = t + .04
                    else:
                        spans.append([t, t + .04])
            points = [[current.get(f"x_{i}"), current.get(f"y_{i}")] for i in range(68)]
            info.update(analysis_timestamp_s=t, analysis_frame=current["frame"],
                        openface_confidence=current["confidence"], openface_success=current["success"],
                        native_hmad=native)
            scores = {k: v for k, v in current.items() if k.startswith("AU") or k.startswith("pose_") or k.startswith("gaze_")}
            record = {"t": t, "states": states, "info": info, "scores": scores,
                      "landmarks": points if info["face_valid"] else [],
                      "gaze_vectors": [[current.get(f"gaze_{eye}_{a}") for a in "xyz"] for eye in (0, 1)]}
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            previous, current = current, following
    annotations = output.with_name("face_annotations.jsonl")
    definitions = {"smiling": ("smile", lambda v: v in ("S1", "S2", "S3", "S4")),
                   "laughing_smile": ("smile", lambda v: v == "S4"),
                   "brow_raising": ("brow_raising", lambda v: v in tuple("ABCDE")),
                   "brow_lowering": ("brow_lowering", lambda v: v in tuple("ABCDE")),
                   "blinking": ("blinking", lambda v: v == "blink"),
                   "gaze_toward_reference": ("gaze", lambda v: v == "eye contact"),
                   "gaze_away_reference": ("gaze", lambda v: v == "no eye contact")}
    with annotations.open("w") as handle:
        for label, (name, positive) in definitions.items():
            spans = valid_spans[label]
            span_index = 0
            for a, b, value in tables.tables[name].rows:
                if not positive(value):
                    continue
                while span_index < len(spans) and spans[span_index][1] <= a:
                    span_index += 1
                i = span_index
                while i < len(spans) and spans[i][0] < b:
                    lo, hi = max(a, spans[i][0]), min(b, spans[i][1])
                    if hi > lo:
                        handle.write(json.dumps({"behaviour": label, "start_s": lo, "end_s": hi,
                                                 "native_value": value}) + "\n")
                    i += 1
    return {"timeline": output, "cycles": cycles_path, "annotations": annotations, "coverage": counts,
            "ignored_zero_duration_intervals": {name: table.zero_duration_rows for name, table in tables.tables.items()}}


class FacialExpressionExtractor:
    """Forward-only source-time join of precomputed face analysis and cycle events."""

    def __init__(self, timeline, cycles):
        self.timeline, self.cycles = Path(timeline), Path(cycles)
        self.record = None
        self.available = False

    def __enter__(self):
        self.handle = self.timeline.open()
        self.cycle_handle = self.cycles.open()
        self.following = self._next(self.handle)
        self.next_cycle = self._next(self.cycle_handle)
        return self

    @staticmethod
    def _next(handle):
        line = handle.readline()
        return json.loads(line) if line else None

    def __exit__(self, *args):
        self.handle.close()
        self.cycle_handle.close()

    def process(self, timestamp_s, end_s):
        self.available = False
        while self.following is not None and self.following["t"] <= timestamp_s + 1e-8:
            self.record = self.following
            self.following = self._next(self.handle)
        if self.record is None or timestamp_s - self.record["t"] >= .04 + 1e-7:
            return dict.fromkeys(FACE_LABELS), {"face_valid": False, "head_valid": False, "face_reason": "uncovered_time"}, {}
        states, info = self.record["states"].copy(), self.record["info"].copy()
        self.available = True
        for label in ("head_nodding", "head_shaking"):
            states[label] = False if info["head_valid"] else None
        while self.next_cycle is not None and self.next_cycle["completion_s"] < end_s - 1e-9:
            if self.next_cycle["completion_s"] >= timestamp_s - 1e-8:
                states[self.next_cycle["behaviour"]] = True
            self.next_cycle = self._next(self.cycle_handle)
        return states, info, self.record["scores"]

    def draw(self, frame):
        if not self.available or self.record is None or not self.record["info"]["face_valid"]:
            return
        points = {i: tuple(round(v) for v in point) for i, point in enumerate(self.record["landmarks"])
                  if all(v is not None for v in point) and 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]}
        for a, b in ((0, 16), (17, 21), (22, 26), (27, 30), (31, 35), (36, 41), (42, 47), (48, 59), (60, 67)):
            for i in range(a, b):
                if i in points and i+1 in points:
                    cv2.line(frame, points[i], points[i+1], (255, 220, 0), 1, cv2.LINE_AA)
        if self.record["info"]["gaze_valid"]:
            for eye, indices in enumerate((range(36, 42), range(42, 48))):
                visible = [points[i] for i in indices if i in points]
                vector = self.record["gaze_vectors"][eye]
                if visible and all(v is not None for v in vector):
                    origin = np.mean(visible, axis=0)
                    target = origin + 70 * np.asarray(vector[:2])
                    cv2.arrowedLine(frame, tuple(origin.astype(int)), tuple(target.astype(int)), (0, 220, 255), 2)
