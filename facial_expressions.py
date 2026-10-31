"""Extract face landmarks and classify visible smile and eyebrow movements.

FacialExpressionExtractor runs MediaPipe Face Landmarker independently of pose.
classify_expressions is a model-free rule function using named blendshapes.
Thresholds are reviewable in FaceRules and BEHAVIOUR_RULES.md; these are facial
movements, not emotion or intention classifications. main.py combines outputs.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FaceRules:
    smile_threshold: float = 0.45
    brow_raise_threshold: float = 0.35
    brow_lower_threshold: float = 0.35

    def __post_init__(self):
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in vars(self).values()):
            raise ValueError("Face thresholds must be finite values between 0 and 1")


FACE_LABELS = ("smiling", "brow_raising", "brow_lowering")


def classify_expressions(scores, detected=True, rules=None):
    """Accept snake-case blendshape names; missing coefficients stay unknown."""
    rules = rules or FaceRules()
    definitions = {
        "smiling": (("mouth_smile_left", "mouth_smile_right"), rules.smile_threshold, "mean"),
        "brow_raising": (("brow_inner_up", "brow_outer_up_left", "brow_outer_up_right"),
                         rules.brow_raise_threshold, "max"),
        "brow_lowering": (("brow_down_left", "brow_down_right"), rules.brow_lower_threshold, "mean"),
    }
    states = dict.fromkeys(FACE_LABELS)
    measurements = {}
    if not detected:
        return states, measurements
    for label, (names, threshold, reduction) in definitions.items():
        values = [scores.get(name, float("nan")) for name in names]
        if not all(math.isfinite(v) for v in values):
            continue
        value = max(values) if reduction == "max" else sum(values) / len(values)
        measurements[label + "_score"] = float(value)
        states[label] = bool(value >= threshold)
    return states, measurements


class FacialExpressionExtractor:
    """Streaming face extraction; only the current frame is retained in memory."""

    def __init__(self, model_path, rules=None):
        self.model_path = model_path
        self.rules = rules or FaceRules()
        self.landmarks = None

    def __enter__(self):
        from face_hand_landmarks import FaceHandLandmarkers
        self.model = FaceHandLandmarkers(1, face_model=self.model_path)
        self.model.__enter__()
        return self

    def __exit__(self, *exc):
        return self.model.__exit__(*exc)

    def process(self, image, timestamp_ms, width, height):
        from face_hand_landmarks import BLENDSHAPE_NAMES
        self.landmarks = None
        # The helper normally stores a complete clip; slot 0 is reused here.
        self.model.arrays["face_detected"][0] = False
        self.model.arrays["face_blendshapes"][0] = float("nan")
        self.model.process(image, timestamp_ms, 0, width, height)
        detected = bool(self.model.arrays["face_detected"][0])
        if detected:
            self.landmarks = self.model.arrays["face_landmarks"][0].copy()
        scores = dict(zip(BLENDSHAPE_NAMES, map(float, self.model.arrays["face_blendshapes"][0]))) if detected else {}
        states, measurements = classify_expressions(scores, detected, self.rules)
        return states, measurements, scores

    def draw(self, image):
        self.model.draw(image)
