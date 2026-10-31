# Behaviour rules for review

These are initial, editable geometric heuristics. They implement the behaviours
in the requested table, but have not been validated against annotated classroom
footage. Concurrent labels are allowed. Facial movements do not infer emotions.
The arm rules use standalone geometry in `body_gestures.py`, without an
importable Gamebody dependency.

## Scripts

- `main.py`: reads the video once, combines both extractors, confirms labels over
  time, and writes the annotated MP4, preview, frame CSV, event CSV and metadata.
- `body_gestures.py`: extracts MediaPipe Pose skeletons/person masks and optional
  Hand landmarks; implements body, motion, orientation and scene rules.
- `facial_expressions.py`: independently extracts Face landmarks/blendshapes and
  implements smile, brow-raising and brow-lowering rules.
- `behaviour_common.py`: shared continuous-evidence timing filter.
- `face_hand_landmarks.py`: shared low-level Face/Hand extraction helper.

Each Python script starts with a note describing its purpose.

## Coordinates, missing data and timing

Geometry uses pixels (`x * width`, `y * height`) to account for aspect ratio.
`T` is the image distance between shoulder midpoint and hip midpoint, in pixels.
Motion/proximity distances are divided by `T` to reduce sensitivity to apparent
body size. Temporal comparisons use the median `T` in the window. These values
are **not measured meters or ground-plane distances**.

Body points must be finite, inside the image, and have both visibility and
presence >= 0.5. Arm rules need their listed points; torso-based temporal rules
require both shoulders and hips. General hand movement only needs both shoulders
and the corresponding wrist, so it still works with cropped hips or legs.
The floor-position anchor is the midpoint of both
ankles, and location rules are unknown when either ankle is unobservable.

The history window is 1.5 s, with at least 1.0 s of continuous torso evidence.
A missing torso or a frame gap > 0.25 s resets history. Walking needs valid
ankles throughout its window; turning needs valid torso yaw throughout it.
No points, scores or labels are filled forward through missing detections.

Every positive raw label needs 0.25 s of continuous evidence before display.
A negative or unknown immediately removes the confirmed label. Frame CSV has
both `raw_*` and confirmed columns: `1` = positive, `0` = negative (including
positive candidates awaiting confirmation), blank = unknown. Event CSV timings
describe confirmed labels as displayed, excluding the confirmation delay;
end timestamps are exclusive. A one-frame detection has nonzero frame duration.
All thresholds below are defaults unless overridden by the config.

## Body and classroom rules

| Requested behaviour / output label | Exact rule | Tunable settings |
| --- | --- | --- |
| Left hand movement / `left_hand_movement` | Over the last 0.5 s, the largest distance between any two observations of `(left_wrist - left_shoulder) / shoulder_width` is >= 0.12 shoulder widths. Require at least 0.3 s of consecutive valid observations. | `hand_motion_window_s`, `hand_motion_min_s`, `hand_motion_range` |
| Right hand movement / `right_hand_movement` | Same independent rule for `(right_wrist - right_shoulder) / shoulder_width`. Bent elbows and arbitrary movement directions are accepted. | Same hand-motion settings |
| Stationary / `stationary` | Maximum hip-midpoint distance from the window's mean position <= 0.12 T, over at least 1.0 s | `stationary_radius`, `history_s`, `motion_min_s` |
| Walking / `walking` | Net hip displacement / window duration >= 0.25 T/s, stationary is false, and relative ankle motion reverses sign at least twice. Choose the x/y axis with the greatest range of `(left_ankle - right_ankle)/T`; discard samples with magnitude < 0.10 before counting sign reversals. Displacement alone is insufficient. | `walking_speed`, `leg_deadband`, `leg_reversals` |
| Moving between zones / `moving_between_zones` | Ankle midpoint enters an annotated zone and stays there >= 0.4 s after a different previously established zone. Show the crossing for 0.8 s, then apply the common confirmation filter. Initial occupancy is not a crossing. Traversing an unannotated area keeps the last established zone; missing ankles/torso resets it. Overlapping polygon membership is ambiguous. | scene `zones`, `zone_dwell_s`, `crossing_display_s` |
| Facing the board / `facing_board` | Torso yaw lies within 35 degrees of your calibrated `board_yaw_deg` | scene `board_yaw_deg`, `facing_tolerance_deg` |
| Facing the class / `facing_class` | Same yaw comparison, against `class_yaw_deg` | scene `class_yaw_deg`, `facing_tolerance_deg` |
| Turning / `turning` | Absolute unwrapped yaw change over the window is between 40 and 150 degrees, then persists through confirmation. This is a rolling orientation-change episode; it can remain positive briefly after rotation stops. | `turn_min_deg`, `turn_max_deg` |
| Raising a hand / `left_hand_raised`, `right_hand_raised` | Wrist is above its same-side shoulder by >= 0.15 shoulder widths. Elbow straightness/visibility is not required, so a bent-elbow raise is accepted. | Existing rule in `body_gestures.py` |
| Extending arms / `left_arm_extended`, `right_arm_extended` | Shoulder-elbow-wrist image angle >= 155 degrees. Direction is one of eight image directions using 45-degree bins. | Existing rule in `body_gestures.py` |
| Coarse pointing / `left_pointing_direction`, `right_pointing_direction` | Arm is extended and its direction is horizontal or upward (exclude the three downward bins). This is a pointing-direction proxy; an open hand can qualify. | Dedicated proxy rule in `body_gestures.py` |
| Finger-confirmed pointing / `left_hand_pointing`, `right_hand_pointing` | Index PIP angle >= 155 degrees, DIP >= 150 degrees, and wrist-tip distance > 1.1 wrist-PIP distance. At least two of middle/ring/little fingers are curled: MCP-PIP-tip angle < 140 degrees OR wrist-tip distance < 1.05 wrist-PIP distance. Thumb ignored. All 21 hand points must be finite/in-frame, with exactly one qualifying handedness result per side and handedness score >= 0.7. | Existing rule in `body_gestures.py`; `handedness_confidence` |
| Approaching a group / `approaching_group` | For an annotated group polygon, ankle-midpoint distance to its boundary decreases >= 0.25 T/s over the history window, starts outside 1.0 T, and ends within 3.0 T. Distance is zero inside the polygon. | scene `groups`, `approach_speed`, `arrival_radius`, `approach_radius` |
| Group arrival / `group_arrival` | The approaching rule holds and current boundary distance <= 1.0 T. This is a window-based arrival episode, not permanent occupancy. | `arrival_radius` |
| Standing / `standing` | Both hip-knee-ankle angles >= 160 degrees, and shoulder-to-hip torso axis is within 25 degrees of downward image vertical | `standing_knee_deg`, `standing_tilt_deg` |
| Standing beside a desk / `standing_beside_desk` | Standing AND stationary AND ankle midpoint is inside, or within 0.8 T of, an annotated desk polygon | scene `desks`, `desk_radius` |

The additional `arms_crossed` label requires both elbow angles < 150 degrees,
wrists crossing toward the opposite upper arms, wrist height from 0.2 shoulder
widths above to 1.2 below the shoulder midpoint, and each wrist within 0.65
shoulder widths of the opposite shoulder-elbow segment. See `classify_frame`
in `body_gestures.py` for the exact projections.

### General left/right hand movement

These two broad labels describe visible wrist movement relative to its own
shoulder, independently of raising, extending or pointing. Both sides can be
positive together and can overlap other gesture labels. Pose wrist landmarks
are sufficient; detailed finger tracking is not required, including when
`--no-hands` is used. Left/right refer to the speaker's anatomical sides.

Each frame uses the current image shoulder width for normalization. Subtracting
the shoulder removes shared body translation; dividing by shoulder width also
removes ideal uniform image scaling. Maximum pairwise distance across the
window detects out-and-back motion even when net displacement is zero. Small
tracking jitter below the 0.12-width threshold is ignored. This measures 2D
wrist motion, not finger-only movements or reliable motion along camera depth.
Torso rotation, perspective changes and pose-estimation errors can still cause
apparent movement.

An invisible/low-confidence wrist resets only its own side's history; missing
shoulders reset both. A gap > 0.25 s resets the affected history. During the
initial 0.3 s, or with missing required points, the label is unknown. After
enough valid observations, movement below threshold is false. The common
0.25 s continuous-positive confirmation still applies before video display.
Because this is a rolling window, a movement label can remain positive for up
to 0.5 s after the wrist stops. CSV measurements include each side's history
span and movement range in shoulder-width units.

### Torso orientation definition

Use MediaPipe's estimated 3D **world** landmarks, while filtering observability
with the corresponding image landmarks:

```text
across = right_shoulder - left_shoulder
down = midpoint(hips) - midpoint(shoulders)
normal = cross(across, down)
yaw = degrees(atan2(normal.x, -normal.z))
```

With unmirrored input: approximately 0 degrees faces the camera, +90 faces image
right, -90 image left, and +/-180 away from the camera. Reject degenerate normals
when their horizontal magnitude divided by `|across| * |down|` is < 0.2.
Angle comparisons wrap correctly at +/-180 degrees.

This deliberately uses **torso facing**, rather than assuming nose direction is
gaze. A head turned independently of the torso will not change these labels.
Calibrate board/class angles using `torso_yaw_deg` in the frame CSV's
`measurements_json` during known examples. A board's 2D image location alone
cannot establish its 3D direction relative to the speaker. Pose depth and
front/back orientation are noisy, especially with occlusion or rear views.

## Facial expression rules

Face Landmarker provides named blendshape coefficients in [0, 1]. These scores
are model coefficients, not calibrated probabilities. The face model runs even
when the body is not detected. A missing face or required coefficient gives
unknown, rather than a negative expression.

| Behaviour / label | Exact raw rule | Config setting |
| --- | --- | --- |
| Smiling / `smiling` | mean(`mouth_smile_left`, `mouth_smile_right`) >= 0.45 | `face.smile_threshold` |
| Brow raising / `brow_raising` | max(`brow_inner_up`, `brow_outer_up_left`, `brow_outer_up_right`) >= 0.35 | `face.brow_raise_threshold` |
| Brow lowering / `brow_lowering` | mean(`brow_down_left`, `brow_down_right`) >= 0.35 | `face.brow_lower_threshold` |

These use face-model blendshapes instead of hand-coded eyebrow pixel distances,
which would need a separate neutral-face and pose calibration. Raising and
lowering have independent rules and can coexist for asymmetric expressions.
The same 0.25 s confirmation applies. Raw coefficients are saved per frame.

## Configure your classroom

`classroom_config.example.json` contains every new threshold. Scene fields are
intentionally empty/null so a first run does not invent your classroom layout.
Without annotations, those specific labels remain unknown; other rules run.

Copy it to `classroom_config.json`. Coordinates are normalized to the **original,
unmirrored source image**: x from left to right and y from top to bottom. Draw
simple, non-self-intersecting polygons around floor areas, desk footprints, and
group locations, not around the screen-space tops of desks. Keep zone interiors
nonoverlapping. Group locations are manual fixed annotations, not automatically
identified student clusters. For example, replace the `scene` section with your
own measurements using this schema (these numbers are illustrative only):

```json
{
  "board_yaw_deg": 180,
  "class_yaw_deg": 0,
  "zones": [
    {"name": "board_area", "polygon": [[0.05,0.65],[0.4,0.65],[0.4,0.98],[0.05,0.98]]},
    {"name": "class_area", "polygon": [[0.42,0.65],[0.95,0.65],[0.95,0.98],[0.42,0.98]]}
  ],
  "groups": [
    {"name": "group_A", "polygon": [[0.7,0.7],[0.9,0.7],[0.9,0.95],[0.7,0.95]]}
  ],
  "desks": [
    {"name": "desk_1", "polygon": [[0.45,0.7],[0.6,0.7],[0.6,0.9],[0.45,0.9]]}
  ]
}
```

Use `--show-scene` to check alignment in the output. Actual camera angles may
make image-distance rules unreliable; a calibrated ground plane is needed for
reliable room distances. Camera panning, cuts, zoom, hidden legs and tiny faces
can invalidate these heuristics. There is no active-speaker selection: run on a
single-speaker view/crop. Pose, Face and Hand models have no cross-model person
association, so results can belong to different people in a multi-person frame.

## Video and review outputs

```sh
conda activate ureca
python main.py  # Default: first 30 seconds of Calculus 1 Lecture 0.1
python main.py --input "your_classroom.mp4" --config classroom_config.json --show-scene
# Full silhouette on black, still with skeleton and behaviour panel:
python main.py --input "your_classroom.mp4" --view mask --output-dir outputs/mask_run
```

Default `--view overlay` retains the original frame and adds a blue person-mask
tint (30% opacity, mask threshold 0.5), green body bones, hand skeletons and face
contours. `--view mask` fills the person silhouette on black; `--view skeleton`
draws only landmarks on black. Labels are in the upper-right. On small frames,
overflow labels remain in the CSV. Missing person masks are not carried forward.

The output directory defaults to `outputs/behaviours` and contains:

- `*_behaviours.mp4`: silent, original frame rate and resolution.
- `*_frames.csv`: raw/confirmed behaviour states, direction, zone, measurements,
  and raw face coefficients per frame.
- `*_events.csv`: confirmed episodes with timestamps and durations.
- `*_metadata.json`: full effective settings, model hashes and detection counts.
- `*_preview.jpg`: a representative frame, preferring a visible skeleton and
  confirmed labels over opening titles.

The default test is the first 30 seconds of the Calculus WebM in the project
root. Override the source with `--input`, duration with `--seconds`, or use
`--full-video` to process the entire source. Default input/model/output paths
resolve relative to the project rather than your current directory. The implementation
streams frames with bounded landmark history rather than retaining the video's
landmark arrays in memory. Use a new output directory for each run; existing
results are protected against replacement. Interrupted/failed runs can leave
partial MP4/CSV files; only a successful run writes the final metadata JSON.

API references: [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
for skeleton/world landmarks and segmentation;
[Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
for face landmarks and blendshapes.
