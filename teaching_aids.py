"""Detect teaching aids and estimate sustained hand-object holding relationships.

Grounding DINO supplies object candidates. Hand association, grasp/contact and
relative motion are project heuristics; outputs do not claim physical contact
or instructional use has been proven from a monocular image.
"""

from dataclasses import asdict
import inspect
import math
from pathlib import Path

import cv2
import numpy as np

from behaviour_common import TemporalGate
from tracking_config import DEFAULT_AIDS, TeachingAidRules


PROMPTS = {"pen": "a pen", "pencil": "a pencil", "marker": "a marker",
           "chalk": "a piece of chalk", "pointer": "a pointing stick", "remote": "a remote control",
           "book": "a book", "paper": "a sheet of paper", "tablet": "a tablet computer"}
OBJECT_LABELS = ("left_holding_object", "right_holding_object", "holding_teaching_aid",
                 *(f"holding_{name}" for name in DEFAULT_AIDS))


def known_or(values):
    return True if any(v is True for v in values) else (None if any(v is None for v in values) else False)


def iou(first, second):
    a, b = np.asarray(first), np.asarray(second)
    intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def point_box_distance(point, box):
    return float(np.linalg.norm(np.maximum(np.maximum(np.asarray(box[:2]) - point, point - box[2:]), 0)))


def grasp_evidence(points):
    """Return pinch/curl evidence and palm width, without treating a fist as an object."""
    palm = float(np.linalg.norm(points[5] - points[17]))
    curled = 0
    for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        a, b = points[mcp] - points[pip], points[tip] - points[pip]
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        angle = math.degrees(math.acos(float(np.clip(np.dot(a, b) / denominator, -1, 1)))) if denominator > 1e-9 else 180
        curled += int(angle < 155 or np.linalg.norm(points[tip] - points[0]) < 1.05 * np.linalg.norm(points[pip] - points[0]))
    pinch = np.linalg.norm(points[4] - points[8]) < .65 * palm
    return bool(pinch or curled >= 1), palm


def teacher_hands(pose, observations, width, height, rules, visibility=.5):
    """Match each detected hand wrist to a unique teacher Pose wrist in the same frame."""
    matches = {"left": [], "right": []}
    if (pose is None or not np.isfinite(pose[[11, 12]]).all() or not (pose[[11, 12], 3:5] >= visibility).all()
            or not ((pose[[11, 12], :2] >= 0) & (pose[[11, 12], :2] <= 1)).all()):
        return {}
    shoulder_width = float(np.linalg.norm((pose[11, :2] - pose[12, :2]) * [width, height]))
    if shoulder_width < 5:
        return {}
    hands = observations.get("hands", [])
    detected = observations.get("hand_detected", [])
    scores = observations.get("handedness_scores", [])
    for slot, hand in enumerate(hands):
        if slot >= len(detected) or not detected[slot] or slot >= len(scores) or not math.isfinite(scores[slot]) or scores[slot] < .7:
            continue
        if hand.shape[0] != 21 or not np.isfinite(hand).all() or not ((hand[:, :2] >= 0) & (hand[:, :2] <= 1)).all():
            continue
        points = hand[:, :2] * [width, height]
        distances = {}
        for side, wrist in (("left", 15), ("right", 16)):
            if (np.isfinite(pose[wrist]).all() and (pose[wrist, 3:5] >= visibility).all()
                    and ((pose[wrist, :2] >= 0) & (pose[wrist, :2] <= 1)).all()):
                distances[side] = float(np.linalg.norm(points[0] - pose[wrist, :2] * [width, height]) / shoulder_width)
        if not distances:
            continue
        ordered = sorted(distances, key=distances.get)
        if distances[ordered[0]] > rules.hand_association_shoulder_widths:
            continue
        if len(ordered) > 1 and distances[ordered[1]] - distances[ordered[0]] < .10:
            continue  # Crossed/overlapping wrists cannot be attributed reliably.
        grasp, palm = grasp_evidence(points)
        if palm >= rules.min_palm_px:
            matches[ordered[0]].append({"points": points, "palm": palm, "grasp": grasp,
                                         "shoulder_width": shoulder_width})
    return {side: values[0] for side, values in matches.items() if len(values) == 1}


class GroundingDinoDetector:
    def __init__(self, model="IDEA-Research/grounding-dino-tiny", revision="main", device="cpu", rules=None,
                 cache_dir=None):
        self.model_name, self.revision, self.device = model, revision, device
        self.rules = rules or TeachingAidRules()
        self.cache_dir = cache_dir
        self.inference_calls = 0

    def __enter__(self):
        try:
            import torch
            import transformers
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Teaching-aid detection needs requirements-objects.txt; see README.md") from error
        self.torch = torch
        source = self.model_name
        # Reuse a complete downloaded snapshot without HEAD requests on every run.
        cache = Path(self.cache_dir) if self.cache_dir else None
        if cache is not None and not Path(source).is_dir():
            repository = cache / ("models--" + source.replace("/", "--"))
            ref = repository / "refs" / self.revision
            revision = ref.read_text().strip() if ref.is_file() else self.revision
            snapshot = repository / "snapshots" / revision
            if (snapshot / "model.safetensors").is_file() and (snapshot / "tokenizer.json").is_file():
                source = str(snapshot)
        local = Path(source).is_dir()
        self.processor = AutoProcessor.from_pretrained(source, revision=self.revision, cache_dir=self.cache_dir,
                                                       local_files_only=local)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            source, revision=self.revision, cache_dir=self.cache_dir, local_files_only=local).to(self.device).eval()
        self.metadata = {"backend": "Grounding DINO", "model": self.model_name,
                         "requested_revision": self.revision, "resolved_revision": getattr(self.model.config, "_commit_hash", None) or (Path(source).name if local else None),
                         "device": self.device, "torch_version": torch.__version__,
                         "transformers_version": transformers.__version__, "rules": asdict(self.rules)}
        if local and (Path(source) / "model.safetensors").is_file():
            from native_tracking import sha256
            self.metadata["weights_sha256"] = sha256(Path(source) / "model.safetensors")
        return self

    def __exit__(self, *args):
        self.metadata["inference_calls"] = self.inference_calls
        self.model = self.processor = None

    def detect(self, frame, crop):
        self.inference_calls += 1
        from PIL import Image
        x0, y0, x1, y1 = map(int, crop)
        image = Image.fromarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        prompt = ". ".join(PROMPTS[c] for c in self.rules.classes) + "."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs)
        postprocess = self.processor.post_process_grounded_object_detection
        threshold_name = "threshold" if "threshold" in inspect.signature(postprocess).parameters else "box_threshold"
        results = postprocess(output, inputs.input_ids, **{threshold_name: self.rules.detection_threshold},
                              text_threshold=self.rules.text_threshold, target_sizes=[(y1-y0, x1-x0)])[0]
        labels = results["text_labels"] if "text_labels" in results else results.get("labels", [])
        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], labels):
            # Ambiguous generated text is retained only when one configured noun matches.
            label = str(label).lower().strip(" .")
            candidates = [c for c in self.rules.classes if label in (c, PROMPTS[c])]
            if len(candidates) != 1:
                continue
            box = np.asarray(box.detach().cpu().tolist(), dtype=float)
            box += [x0, y0, x0, y0]
            box[::2] = np.clip(box[::2], 0, frame.shape[1])
            box[1::2] = np.clip(box[1::2], 0, frame.shape[0])
            if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
                detections.append({"class": candidates[0], "score": float(score), "box": box.tolist()})
        return detections


class TeachingAidTracker:
    def __init__(self, detector, rules=None, visibility=.5):
        self.detector, self.rules, self.visibility = detector, rules or TeachingAidRules(), visibility
        self.gate = TemporalGate(self.rules.confirmation_s)
        self.previous = {}
        self.last_t = None
        self.boxes = []

    def reset(self):
        self.gate = TemporalGate(self.rules.confirmation_s)
        self.previous.clear()
        self.last_t = None
        self.boxes = []

    def process(self, frame, t, pose, observations):
        if self.last_t is not None and t - self.last_t > self.rules.max_gap_s:
            self.reset()
        self.last_t = t
        hands = teacher_hands(pose, observations, frame.shape[1], frame.shape[0], self.rules, self.visibility)
        raw = dict.fromkeys(OBJECT_LABELS)
        info = {"holding_status": "unknown", "held_objects": [], "object_candidates": [], "holding_by_hand": {}}
        self.boxes = []
        side_classes = {}
        selected = {}
        for side in ("left", "right"):
            hand = hands.get(side)
            if hand is None:
                self.previous.pop(side, None)
                info["holding_by_hand"][side] = {"status": "unknown", "reason": "hand_unobservable_or_unassociated"}
                continue
            points, palm = hand["points"], hand["palm"]
            center = points[[0, 5, 9, 13, 17]].mean(axis=0)
            radius = max(3 * palm, .65 * hand["shoulder_width"])
            x0, y0 = np.maximum(0, np.floor(center - radius)).astype(int)
            x1, y1 = np.minimum(frame.shape[1::-1], np.ceil(center + radius)).astype(int)
            detections = self.detector.detect(frame, (x0, y0, x1, y1))
            candidates = []
            for obj in detections:
                if obj["class"] not in self.rules.classes or obj["score"] < self.rules.detection_threshold:
                    continue
                contacts = sum(point_box_distance(point, obj["box"]) <= self.rules.contact_palm_widths * palm
                               for point in points[[4, 8, 12, 16, 20]])
                info["object_candidates"].append({**obj, "hand": side, "contacts": contacts})
                if hand["grasp"] and contacts >= 2:
                    candidates.append(obj)
            candidates.sort(key=lambda obj: obj["score"], reverse=True)
            # Two different plausible object types near the same grip remain ambiguous.
            ambiguous = len(candidates) > 1 and candidates[0]["class"] != candidates[1]["class"] and candidates[0]["score"] - candidates[1]["score"] < .10
            if ambiguous:
                self.previous.pop(side, None)
                info["holding_by_hand"][side] = {"status": "unknown", "reason": "ambiguous_object_type"}
                continue
            obj = candidates[0] if candidates else None
            raw[f"{side}_holding_object"] = obj is not None
            side_classes[side] = obj["class"] if obj else None
            if obj:
                object_center = (np.asarray(obj["box"][:2]) + obj["box"][2:]) / 2
                previous = self.previous.get(side)
                if previous:
                    relative_change = np.linalg.norm((object_center - center) - (previous["object_center"] - previous["hand_center"])) / palm
                    hand_motion = np.linalg.norm(center - previous["hand_center"]) / palm
                    same = obj["class"] == previous["class"] and relative_change <= self.rules.relative_motion_palm_widths
                    same = same and (hand_motion > .25 or iou(obj["box"], previous["box"]) > .1)
                    if not same:
                        self.gate.since.pop(f"{side}_holding_object", None)
                        for name in self.rules.classes:
                            self.gate.since.pop(f"holding_{name}", None)
                        self.gate.since.pop("holding_teaching_aid", None)
                self.previous[side] = {"class": obj["class"], "object_center": object_center,
                                       "hand_center": center, "box": obj["box"]}
                selected[side] = obj
            else:
                self.previous.pop(side, None)
            info["holding_by_hand"][side] = {"status": "candidate" if obj else "no_target_object_observed"}
        # One detected object cannot be attributed to both hands unless both have grip/contact evidence.
        for name in self.rules.classes:
            raw[f"holding_{name}"] = known_or([None if side not in side_classes else side_classes[side] == name for side in ("left", "right")])
        raw["holding_teaching_aid"] = known_or([raw["left_holding_object"], raw["right_holding_object"]])
        confirmed_sides = self.gate.update({k: raw[k] for k in ("left_holding_object", "right_holding_object")}, t)
        confirmed = dict.fromkeys(OBJECT_LABELS)
        confirmed.update(confirmed_sides)
        confirmed["holding_teaching_aid"] = known_or(list(confirmed_sides.values()))
        for name in self.rules.classes:
            confirmed[f"holding_{name}"] = known_or([
                None if confirmed_sides[f"{side}_holding_object"] is None else
                bool(confirmed_sides[f"{side}_holding_object"] and side_classes.get(side) == name)
                for side in ("left", "right")])
        for side, obj in selected.items():
            if confirmed[f"{side}_holding_object"]:
                info["held_objects"].append({**obj, "hand": side})
                info["holding_by_hand"][side]["status"] = "holding"
                self.boxes.append(obj)
        state = confirmed["holding_teaching_aid"]
        info["holding_status"] = "holding" if state is True else ("no_target_object_observed" if raw["holding_teaching_aid"] is False else "pending" if raw["holding_teaching_aid"] is True else "unknown")
        return raw, confirmed, info

    def draw(self, frame):
        for obj in self.boxes:
            x0, y0, x1, y1 = map(round, obj["box"])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 120, 220), 2)
            cv2.putText(frame, obj["class"], (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (255, 120, 220), 1, cv2.LINE_AA)
