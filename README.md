# Classroom behaviour extraction

The default test video is **Calculus 1 Lecture 0.1: Lines, Angle of Inclination,
and the Distance Formula**, stored as the WebM in the project root. The default
run processes its first **30 seconds**.

## Run

```sh
conda activate ureca
python main.py
```

The input video, model files and default output directory resolve relative to
this project, so the command also works when launched from another directory.

```sh
# Change the test duration, or process the whole default lecture.
python main.py --seconds 60 --output-dir outputs/calculus_60s
python main.py --full-video --output-dir outputs/calculus_full

# Use another video or a classroom configuration.
python main.py --input "/path/to/classroom.mp4" --seconds 30
python main.py --config classroom_config.example.json --show-scene

# Filled person silhouette on black, with skeleton and labels.
python main.py --view mask --output-dir outputs/calculus_mask
```

Default output: `outputs/behaviours/`. Each run writes a silent MP4 containing a
translucent person mask, skeleton, face contours, hand landmarks and confirmed
behaviour labels in the upper-right. Frame CSV, event CSV, metadata JSON and a
representative preview image are also saved. Existing outputs are protected;
choose a new `--output-dir` for a repeat run.

The verified 30-second Calculus test is kept in
[`outputs/calculus_30s_test/`](outputs/calculus_30s_test/), including its video,
CSV files, preview and `verification.json`. All 900 frames decoded and matched
the frame CSV. The source is about 29.97 fps; selecting source frames before
30 s produces an encoded duration of about 30.03 s.

## Current scripts

| Script | Purpose |
| --- | --- |
| `main.py` | Combine body and facial extraction, render video, export review outputs |
| `body_gestures.py` | Pose/mask/hand extraction, arm/finger geometry, motion, orientation and classroom rules |
| `facial_expressions.py` | Face extraction and separate smile/brow movement rules |
| `face_hand_landmarks.py` | Shared low-level MediaPipe Face/Hand model and drawing helpers |
| `behaviour_common.py` | Continuous-evidence filter for displayed labels |

Every script starts with a note explaining its purpose. The body module contains
all active body geometry and drawing functions; only `main.py` is a video entry
point.

General `left_hand_movement` and `right_hand_movement` labels track each wrist
relative to its shoulder over time. They accept bent elbows and any visible
movement direction, and work without detailed finger tracking. Thresholds and
missing-data handling are documented in [BEHAVIOUR_RULES.md](BEHAVIOUR_RULES.md).

## Review the rules

[BEHAVIOUR_RULES.md](BEHAVIOUR_RULES.md) documents each rule, threshold, unit,
assumption and output field. Use
[classroom_config.example.json](classroom_config.example.json) to adjust the
thresholds and annotate your scene. The back-of-classroom camera defaults to
class-facing toward the camera (yaw 0), and board-facing away (yaw 180).
Zones, group locations and desk footprints need your annotations.

`turning_left` and `turning_right` compare the facial-skeleton center to the
shoulder-midpoint neck coordinate, using image left/right and a 10%-of-shoulder-
width deadband. `turning` is their combined label. These head-posture labels use
the current face mesh, with visible Pose face points as a fallback.

These are initial heuristics for validation. The pipeline assumes a fixed
camera and one visible speaker; it does not select the active speaker or
associate different people across the independent models. Missing landmarks
produce unknown labels. In the Calculus test, the legs are cropped out, so
walking and standing rules remain unknown. Facial labels describe visible
movements, not emotions. Without classroom annotations, location-specific labels
also remain unknown.

## Setup

Verified environment: Python 3.12 on macOS Apple Silicon.

```sh
conda create -n ureca python=3.12 pip
conda activate ureca
python -m pip install -r requirements.txt
```

Download the Pose, Face and Hand models following
[models/README.md](models/README.md). Local videos, downloaded model weights and
generated outputs are excluded from Git. See [THIRD_PARTY.md](THIRD_PARTY.md)
for attribution. On macOS, MediaPipe needs a terminal with graphics-service
access, even for CPU inference.

## Verification

```sh
python -m unittest discover -s tests -v
python main.py --help
```

Tests cover body and finger geometry, temporal filtering, missing detections,
orientation wrapping, classroom rules, facial thresholds, model helper exports,
rendering and command defaults.
