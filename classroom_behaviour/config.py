"""Validated settings for native face analysis, camera movement and held objects."""

from dataclasses import dataclass, field
import math


def positive_settings(instance, skip=()):
    for name, value in vars(instance).items():
        if name in skip:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{type(instance).__name__}.{name} must be positive and finite")


@dataclass(frozen=True)
class FaceRules:
    confidence: float = 0.8

    def __post_init__(self):
        if not math.isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("face.confidence must be in (0, 1)")


@dataclass(frozen=True)
class HeadRules:
    turn_yaw_deg: float = 15.0
    cycle_excursion_deg: float = 8.0
    cycle_return_deg: float = 4.0
    reversal_deg: float = 2.0
    cycle_min_s: float = 0.25
    cycle_max_s: float = 2.0
    # OpenFace's left-handed rotation: Ry > 0 points the nose to image left.
    yaw_sign: int = -1
    pitch_up_sign: int = -1

    def __post_init__(self):
        positive_settings(self, ("yaw_sign", "pitch_up_sign"))
        if self.yaw_sign not in (-1, 1) or self.pitch_up_sign not in (-1, 1):
            raise ValueError("Head angle signs must be -1 or 1")
        if self.cycle_min_s >= self.cycle_max_s or self.cycle_return_deg >= self.cycle_excursion_deg:
            raise ValueError("Invalid head cycle duration/return thresholds")


@dataclass(frozen=True)
class GazeRules:
    horizontal_deg: float = 15.0
    vertical_deg: float = 10.0
    horizontal_sign: int = 1
    vertical_sign: int = 1
    reference_validated: bool = False
    reference_description: str = ""

    def __post_init__(self):
        positive_settings(self, ("horizontal_sign", "vertical_sign", "reference_validated", "reference_description"))
        if self.horizontal_sign not in (-1, 1) or self.vertical_sign not in (-1, 1):
            raise ValueError("Gaze signs must be -1 or 1")
        if not isinstance(self.reference_validated, bool) or not isinstance(self.reference_description, str):
            raise ValueError("Invalid gaze reference settings")
        if self.reference_validated and not self.reference_description.strip():
            raise ValueError("A validated gaze reference needs a description")


@dataclass(frozen=True)
class CameraMovementRules:
    window_s: float = 1.5
    min_s: float = 1.0
    max_gap_s: float = 0.25
    depth_rate: float = 0.03
    body_scale_rate: float = 0.02
    orientation_change_deg: float = 10.0
    background_zoom_log: float = 0.01
    background_translation: float = 0.02
    cut_difference: float = 0.28

    def __post_init__(self):
        positive_settings(self)
        if self.window_s < self.min_s:
            raise ValueError("camera_motion.window_s must be >= min_s")


DEFAULT_AIDS = ("pen", "pencil", "marker", "chalk", "pointer", "remote", "book", "paper", "tablet")


@dataclass(frozen=True)
class TeachingAidRules:
    classes: tuple = DEFAULT_AIDS
    detection_threshold: float = 0.35
    text_threshold: float = 0.25
    confirmation_s: float = 0.4
    max_gap_s: float = 0.25
    contact_palm_widths: float = 0.30
    relative_motion_palm_widths: float = 0.65
    hand_association_shoulder_widths: float = 0.5
    min_palm_px: float = 8.0

    def __post_init__(self):
        object.__setattr__(self, "classes", tuple(self.classes))
        if not self.classes or len(set(self.classes)) != len(self.classes) or any(c not in DEFAULT_AIDS for c in self.classes):
            raise ValueError(f"objects.classes must be a nonempty unique subset of {DEFAULT_AIDS}")
        positive_settings(self, ("classes",))
        if self.detection_threshold > 1 or self.text_threshold > 1:
            raise ValueError("Object detection thresholds must be <= 1")


@dataclass(frozen=True)
class TrackingRules:
    face: FaceRules = field(default_factory=FaceRules)
    head: HeadRules = field(default_factory=HeadRules)
    gaze: GazeRules = field(default_factory=GazeRules)
    camera_motion: CameraMovementRules = field(default_factory=CameraMovementRules)
    objects: TeachingAidRules = field(default_factory=TeachingAidRules)
