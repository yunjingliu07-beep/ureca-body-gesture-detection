# Body gesture analysis

`detect_gestures.py` is the main entry point for video-based gesture analysis.
It runs the existing body/face/hand extraction pipeline, then exports observable
arm gestures with timestamps, durations, and occurrence counts. It can also
analyze saved landmark arrays without running the models again.

## Gesture analysis

```sh
conda activate ureca
python detect_gestures.py --input "/path/to/classroom.mp4" --seconds 120

# Reuse an earlier extraction and its matching *_metadata.json:
python detect_gestures.py --landmarks "/path/to/session_arrays.npz"
```

The initial geometric rules detect:

| Labels | Definition |
| --- | --- |
| `left_hand_raised`, `right_hand_raised` | Wrist above its shoulder by at least 15% of shoulder width; bent elbows are accepted |
| `left_arm_extended`, `right_arm_extended` | Elbow angle at least 155 degrees, with image-relative arm direction |
| `arms_crossed` | Both arms bent, wrists crossing the torso center toward the opposite upper arms, at chest height |
| `left_hand_pointing`, `right_hand_pointing` | Index finger extended and at least two other fingers curled; thumb is ignored |

Pointing uses detailed hand landmarks. An extended arm alone is not labelled as
a pointing finger. Hand-side labels follow sufficiently confident model
predictions; ambiguous or missing predictions remain unknown. Directions are
relative to the unmirrored image, independently of anatomical left/right.

By default, results go to `outputs/gestures/`:

- `*_gestures_frames.csv`: frame-level labels and directions. `1` means positive,
  `0` means observed negative, and blank means insufficient data.
- `*_gestures_events.csv`: continuous gesture episodes, with start/end frames,
  start/end timestamps, duration, and dominant direction.
- `*_gestures_summary.json`: event counts, total event durations, observable
  durations, settings and output paths.
- `landmarks/`: raw extraction outputs when using `--input`.

Events require 0.3 seconds of consecutive positive frames by default. Change
this with `--min-duration`. Missing observations end events and are not counted
as negative observations. Multiple gestures can overlap. Changes in direction
do not split a continuous gesture episode; its dominant direction is reported.
The event filter does not modify raw frame labels.

Use `--visibility` (default 0.5) and `--handedness-confidence` (default 0.7) to
set observation thresholds. `--no-face` and `--no-hands` disable the respective
model during video extraction. Face landmarks and blendshapes remain available
as raw outputs; this script does not yet classify facial movements, walking,
body orientation, zones, or group/desk interactions. These gesture rules are
an initial implementation and require validation on the intended recordings.

Saved-array analysis requires the matching metadata JSON for image dimensions,
because geometric measurements account for image aspect ratio. Timestamps must
be regularly sampled. A one-frame archive needs FPS metadata or `--fps`.

## Setup after cloning

The verified setup targets Python 3.12 on macOS Apple Silicon:

```sh
conda create -n ureca python=3.12 pip
conda activate ureca
python -m pip install -r requirements.txt
```

Download the three models using [models/README.md](models/README.md), then run
the extractor with your own video:

```sh
python detect_gestures.py --input "/path/to/video.mp4" --seconds 120
```

This repository includes code, tests, configuration, and documentation. Input
videos, generated outputs, downloaded model binaries, and local caches are
excluded from Git. See [THIRD_PARTY.md](THIRD_PARTY.md) for dependency and model
attribution.

## Raw landmark extraction

The extractor now runs MediaPipe Pose, Face, and Hand Landmarkers on the same
frames and timestamps. Face and hands are enabled by default, with one body,
up to one face, and up to two hands. Run from this folder in `ureca`:

```sh
conda activate ureca
python extract_speaker_pose.py --input "linear algebra.mp4" --seconds 120
```

All three models belong in `models/`; download instructions, sources, and
checksums are in [models/README.md](models/README.md).
The [Face](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
and [Hand](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
Tasks video APIs run independently of body detection, so their results are
retained even when no body is detected. Person association in multi-person
scenes is not implemented.

## New face and hand outputs

The default output prefix is `outputs/linear_algebra_first_120s_pose_face_hands_`.
This keeps the original pose-only sample files below intact. The existing
overlay/skeleton videos now also show cyan face contours and hand skeletons.
The body CSV format is unchanged. Two additional files are exported:

- `face_blendshapes.csv`: one row per frame, with timestamp, face detection flag,
  and 52 named blendshape coefficients.
- `hand_landmarks.csv`: two result slots × 21 landmarks per frame, including
  normalized/world coordinates, handedness labels, and handedness scores.

The synchronized `arrays.npz` keeps the original body keys and adds:

| Key | Shape | Meaning |
| --- | --- | --- |
| `face_landmarks` | `(N, 478, 3)` | Normalized face x/y/z |
| `face_detected` | `(N,)` | Face detection mask |
| `face_blendshapes` | `(N, 52)` | Raw expression coefficients |
| `face_blendshape_names` | `(52,)` | Snake-case names matching CSV columns |
| `face_transformation_matrices` | `(N, 4, 4)` | Canonical-face-to-detected-face transforms |
| `hand_landmarks` | `(N, 2, 21, 3)` | Normalized hand x/y/z |
| `hand_world_landmarks` | `(N, 2, 21, 3)` | Estimated hand world x/y/z |
| `hand_detected` | `(N, 2)` | Per-slot detection masks |
| `hand_handedness` | `(N, 2)` | Raw `Left`/`Right` model predictions |
| `hand_handedness_scores` | `(N, 2)` | Handedness classification confidence |
| `hand_landmark_names` | `(21,)` | Hand landmark names |
| `detail_coordinate_names` | `(3,)` | x/y/z labels for face and hand arrays |

`N` is the number of processed frames. Load with `np.load(path, allow_pickle=False)`.
Missing numeric results are `NaN` in arrays and blank in CSV; missing handedness
labels are empty strings. Scores are never filled forward across missing frames.
Metadata records model sources/checksums, detection counts and missing intervals
separately for face and hands.

Hand slots follow each frame's detection order, **not persistent left/right IDs**.
Read the predicted label and its score alongside each slot; two results with
the same predicted label are both preserved. Input frames are not mirrored.
Handedness confidence is not confidence in a particular gesture.

Blendshapes include `mouth_smile_left`, `mouth_smile_right`, `brow_inner_up`,
`brow_outer_up_left`, `brow_outer_up_right`, `brow_down_left`, and `brow_down_right`.
These are raw coefficients. This step does not add behaviour labels, thresholds,
event durations, counts, or emotion interpretation.

Face/hand results have no per-landmark visibility or presence confidence. Their
finite, in-frame points are drawn after detection; `--visibility` applies only
to body drawing. Face normalized depth is model-relative. Hand world coordinates
are estimated meters centered on each hand, independently of the body coordinate
system. They are not classroom coordinates or directly comparable to pose world
coordinates.

## Options and verification

```sh
# Body only: original filenames and body CSV/NPZ layout.
python extract_speaker_pose.py --no-face --no-hands

# Body + face, or body + hands.
python extract_speaker_pose.py --no-hands
python extract_speaker_pose.py --no-face

# Short check with separate outputs.
python extract_speaker_pose.py --seconds 30 --output-dir outputs/face_hand_check

python -m unittest discover -s tests -v
```

Override models with `--model`, `--face-model`, or `--hand-model`; only enabled
model files are required. Disabled modalities add no arrays or CSV files and are
marked disabled in metadata. Unit tests cover missing data, named blendshape
ordering, duplicate handedness labels, timestamped CSV rows and disabled models.
On macOS, run from a terminal with graphics-service access, which MediaPipe
0.10.21 needs even for CPU inference.

## Original pose-only sample

The local sample files described below were generated before this extension,
using MediaPipe 1.1.1. They are excluded from Git. New runs record their actual
installed version in metadata.

MediaPipe Pose Landmarker (Full model) processes **00:00–02:00** of `linear algebra.mp4`, using the existing `ureca` Conda environment. Every source frame is processed at the original 30 fps and 1920 × 1080 resolution.

To reproduce the pose-only sample with the current script:

```sh
conda run --no-capture-output -n ureca python extract_speaker_pose.py --no-face --no-hands
```

The model is stored at `models/pose_landmarker_full.task`. It comes from [Google's MediaPipe model storage](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task). The script uses the [MediaPipe Pose Landmarker video API](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python).

## Outputs

All files are in `outputs/`, with prefix `linear_algebra_first_120s_`:

- `overlay.mp4`: skeleton drawn over the original video.
- `skeleton.mp4`: skeleton on a black background, at the original image positions.
- `landmarks.csv`: one row per frame and landmark, including frame index, timestamp, the 33 landmark names, normalized coordinates, estimated world coordinates, visibility, presence, and whether the point is drawn.
- `arrays.npz`: NumPy arrays shaped `(3600, 33, 5)` for normalized and world landmarks. The last axis is `[x, y, z, visibility, presence]`. Also includes timestamps, landmark names, and a per-frame detection mask.
- `metadata.json`: processing settings, model checksum, detection counts, and intervals with no pose.
- `preview.jpg`: sampled overlay frames for visual inspection.

Both videos are silent. Frame indices start at zero. The end time is exclusive: 3,600 frames cover exactly 120 seconds, with the last frame timestamp at approximately 119.967 seconds.

Missing poses have blank CSV coordinates and `NaN` array values. No missing poses are interpolated. All 33 estimates are retained when a pose is detected, including low-confidence estimates. Drawing requires visibility and presence of at least 0.5, and a point inside the image. The opening title sequence and off-screen or occluded limbs can therefore produce incomplete or absent skeletons.

Normalized x and y are relative to the image dimensions. Normalized z is relative model depth, not meters. World x, y, and z are model estimates in meters relative to the hip midpoint; they are not calibrated room coordinates.

```python
import numpy as np

data = np.load("outputs/linear_algebra_first_120s_arrays.npz")
poses = data["normalized_landmarks"]
valid_frames = data["pose_detected"]
left_wrist = poses[:, 15, :3]
```

Optional arguments include `--input`, `--seconds`, `--model`, `--output-dir`, and `--visibility`; run `python extract_speaker_pose.py --help` for details. Other durations produce correspondingly sized arrays and filenames.
