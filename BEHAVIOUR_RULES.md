# Behaviour rules for review

**Implementation updated 12 November 2026: OpenFace + HMAD and teaching aids.**
The active face/head/gaze branch uses OpenFace 2.2 plus HMAD's SMAD, EBMAD and
BAGMAD modules. The former MediaPipe blendshape facial rules and face-to-neck
head-turn mechanism have been removed from the video pipeline. MediaPipe Pose
and Hands continue to supply body, mask, hand and finger measurements.

The pipeline also estimates toward/away movement from corroborating head depth
and body-scale changes, and holding of configured teaching aids using Grounding
DINO plus hand/object relationship rules. These are initial research heuristics,
not validated activity or emotion classifiers. Facial movements and held objects
do not establish emotions, intentions, instructional use or teaching effectiveness.
Missing evidence remains unknown. Several labels can be true together.

## Scripts and processing stages

| File | Purpose |
| --- | --- |
| `main.py` | Validate dependencies/config, prepare native analysis, merge source-time observations, render/export review outputs |
| `native_tracking.py` | Create the 25 fps analysis copy; run OpenFace and R/HMAD; retain provenance, hashes and component status |
| `hmad_bridge.R` | Run upstream head/residual, SMAD, EBMAD and BAGMAD models and export canonical interval tables |
| `facial_expressions.py` | Stream native measurements, map HMAD labels, detect completed nod/shake cycles and join to source video |
| `body_gestures.py` | MediaPipe Pose/Hands, body/mask/hand extraction and existing body/classroom rules |
| `camera_movement.py` | Background camera-motion/cut checks and corroborated toward/away detection |
| `teaching_aids.py` | Grounding DINO inference, teacher-hand association, grasp/contact/motion evidence and holding confirmation |
| `tracking_config.py` | Validated face, head, gaze, camera-movement and object settings |
| `behaviour_common.py` | Continuous-positive timing gate |
| `face_hand_landmarks.py` | Low-level hand helper; its legacy standalone Face utility is not the active face backend |
| `setup_tracking.py`, `build_native.py` | Download model/source assets; reproduce the isolated macOS arm64 native build |

The pipeline is offline: OpenFace extraction, R/HMAD processing, then merging and
rendering with body/hand/object inference. HMAD sequence models and the head
cycle median can use later observations. Outputs must not be described as
real-time detections. Each script begins with a note describing its purpose.

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
ankles throughout its window. Directional head labels need valid
OpenFace head pose; shoulders, hips and MediaPipe face points are not required.
No points, scores or labels are filled forward through missing detections.

Body labels and head-posture/coarse-gaze labels need 0.25 s of
continuous positive evidence before display. HMAD interval annotations and
completed nod/shake events use the separate timing policy below.
A negative or unknown immediately removes the confirmed label. Frame CSV has
both `raw_*` and confirmed columns: `1` = positive, `0` = negative (including
positive candidates awaiting confirmation), blank = unknown. Event CSV timings
describe confirmed labels as displayed, excluding the confirmation delay;
end timestamps are exclusive. A one-frame detection has nonzero frame duration.
All thresholds below are defaults unless overridden by the config. Body/head/gaze
postures use `confirmation_s` (0.25 s); object holding uses its own 0.4 s gate.

## Body and classroom rules

| Requested behaviour / output label | Exact rule | Tunable settings |
| --- | --- | --- |
| Left hand movement / `left_hand_movement` | Over the last 0.5 s, the largest distance between any two observations of `(left_wrist - left_shoulder) / shoulder_width` is >= 0.12 shoulder widths. Require at least 0.3 s of consecutive valid observations. | `hand_motion_window_s`, `hand_motion_min_s`, `hand_motion_range` |
| Right hand movement / `right_hand_movement` | Same independent rule for `(right_wrist - right_shoulder) / shoulder_width`. Bent elbows and arbitrary movement directions are accepted. | Same hand-motion settings |
| Stationary / `stationary` | Maximum hip-midpoint distance from the window's mean position <= 0.12 T, over at least 1.0 s | `stationary_radius`, `history_s`, `motion_min_s` |
| Walking / `walking` | Net hip displacement / window duration >= 0.25 T/s, stationary is false, and relative ankle motion reverses sign at least twice. Choose the x/y axis with the greatest range of `(left_ankle - right_ankle)/T`; discard samples with magnitude < 0.10 before counting sign reversals. Displacement alone is insufficient. | `walking_speed`, `leg_deadband`, `leg_reversals` |
| Moving between zones / `moving_between_zones` | Ankle midpoint enters an annotated zone and stays there >= 0.4 s after a different previously established zone. Show the crossing for 0.8 s, then apply the common confirmation filter. Initial occupancy is not a crossing. Traversing an unannotated area keeps the last established zone; missing ankles/torso resets it. Overlapping polygon membership is ambiguous. | scene `zones`, `zone_dwell_s`, `crossing_display_s` |
| Facing the board / `facing_board` | With a back-of-classroom camera, torso yaw is within 35 degrees of 180 degrees (away from camera, toward the front board) by default | scene `board_yaw_deg` (default 180), `facing_tolerance_deg` |
| Facing the class / `facing_class` | Torso yaw is within 35 degrees of 0 degrees (toward the camera and the class) by default | scene `class_yaw_deg` (default 0), `facing_tolerance_deg` |
| Turning left / `turning_left` | Valid OpenFace head yaw, normalized to image directions, is < -15 degrees. See head rules below. | `head.turn_yaw_deg = 15` |
| Turning right / `turning_right` | Same yaw is > +15 degrees. | Same head setting |
| Turning / `turning` | Either directional head-posture rule is true. A held side-facing head remains labelled. | Head setting and 0.25 s confirmation |
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

The camera is assumed to be at the back of the classroom: **facing the camera
means facing the class**. Defaults are `class_yaw_deg = 0` and
`board_yaw_deg = 180`; these apply even without a config file. Override them for
an off-axis camera/board, or set either to `null` to disable that facing label.
Facing labels describe torso orientation rather than measured gaze. A sideways
head turn can therefore coexist with a torso-facing-class label. Calibrate
angles using `torso_yaw_deg` in CSV `measurements_json` if needed. Estimated
pose depth and front/back orientation are noisy, particularly in rear views.

## OpenFace + HMAD input and quality rules

Use OpenFace's single-face `FeatureExtraction` on a single-speaker view/crop.
Keep its original CSV, including `frame`, `timestamp`, `confidence`, `success`,
`pose_Rx/Ry/Rz`, `pose_Tx/Ty/Tz`, 2D/3D facial landmarks, `AU*_r`, `AU*_c`,
the two gaze vectors and `gaze_angle_x/y`. AU intensities are on a 0-5 scale;
AU occurrence flags are 0/1. They are not MediaPipe blendshape coefficients.

An observation is usable only when `success == 1`, `confidence > 0.8`, and the
fields required by that rule are finite. The 0.8 quality boundary follows
HMAD's filtering convention. Additional HMAD quality rejection applies to its
own outputs: face-size/projection outliers, unreliable residuals, `X`/missing
intervals, or an unavailable head/reference model produce unknown labels.
Do not apply HMAD's frontal head-model calibration restrictions as a blanket
rejection of all OpenFace side-facing head poses.

Check each modality independently. Valid head pose can survive missing gaze;
missing body detections do not invalidate a usable face. Unknown data never
becomes a negative, a neutral expression or a previous valid observation.
End active episodes and reset custom angular history at missing observations,
detected camera cuts, or timestamp gaps > 0.25 s. Known speaker changes require
separate selected clips; identity changes are not detected automatically. Mask HMAD outputs
against these invalid intervals even if its sequence model predicts a state
there. Preserve any internal HMAD smoothing/blink correction as model processing,
without exporting invented raw observations during tracking loss.

Keep the single-speaker assumption explicit. With multiple people, the caller
must select a teacher crop or separate clips. Automatic cross-model face/body
identity association is not implemented; unselected multi-person footage can
mix identities and is unsuitable for the claimed teacher-specific outputs.

### Frame rate and source-time alignment

The inspected SMAD source constructs interval boundaries with a hard-coded
0.04 s step, assuming **25 fps**. The default project video is about 29.97 fps.
The compatibility approach is a **25 fps analysis copy** of the
selected source interval, preserving playback speed, duration and crop geometry.
Run both OpenFace and HMAD on that copy, while keeping the original video for
body/hand extraction and final rendering. Never relabel 29.97 fps frames as
25 fps, or multiply native-frame indices by 0.04.

Store source/analysis frame indices, timestamps and clip-start offset explicitly.
Map HMAD intervals by source time using `[start_s, end_s)` membership. For
per-frame head/gaze measurements, select the containing 40 ms analysis-frame
interval and record its analysis timestamp; do not interpolate between valid
and invalid observations. Events are clipped to the selected source duration.
Reject duplicate/nonmonotonic timestamps and duration mismatches beyond one
analysis frame. Provenance-matched imports may cover a longer source prefix;
the renderer uses the requested shorter prefix and retains the imported model's
full-sequence context. The existing 29.97 fps video is included in the timing
verification. Resampling does
not create higher temporal precision than the analysis frames.

### Angles and directions

OpenFace exports rotation and gaze angles in **radians**. Convert them to
degrees once at the adapter boundary. Its head rotation convention is
`R = Rx * Ry * Rz`; HMAD additionally negates OpenFace pitch in its own table.
Preserve both native columns and the normalized values with their units.
Default head signs follow OpenFace's rotation matrix (raw yaw positive toward
image left; raw pitch positive downward). Validate directions against your
footage; a raw gaze angle must never be mistaken for head yaw.

Project convention for unmirrored video: head `yaw_image_deg` is negative
toward image left and positive toward image right; zero faces the camera.
Head `pitch_up_deg` is positive upward. Gaze `gaze_right_deg` is positive toward
image right and `gaze_down_deg` is positive downward, with zero along the
camera-facing reference direction. Left/right here are image directions, not
the teacher's anatomical sides. Head and gaze angles use the camera reference,
not an automatically assumed neutral first frame. Save camera intrinsics, or
state explicitly when OpenFace uses estimated intrinsics.

## Head posture and movement rules

| Behaviour / label | Rule | Project setting |
| --- | --- | --- |
| Side-facing left / `turning_left` | Usable `yaw_image_deg < -15`. | `head.turn_yaw_deg = 15` |
| Side-facing right / `turning_right` | Usable `yaw_image_deg > +15`. | Same setting |
| Side-facing / `turning` | Logical OR of the two side labels; both false within the inclusive [-15, +15] deadband. | 0.25 s continuous confirmation |
| Nodding / `head_nodding` | A completed pitch excursion and return satisfying the cycle definition below. | Minimum excursion 8 degrees; return tolerance 4 degrees; cycle duration 0.25-2.0 s |
| Head shaking / `head_shaking` | Same completed-cycle rule on head yaw. | Same cycle settings |

The three existing `turning*` names are retained for output compatibility, but
now use estimated head rotation. They remain posture labels, so a held head
turn is not continuous motion. Translating/leaning sideways with unchanged
head orientation must not alone trigger them. `facing_class` and `facing_board`
continue to describe torso orientation; they can coexist with head turns and
do not describe eye gaze.

Nodding/shaking are **project rules derived from OpenFace pose**, not native
HMAD gesture classifiers. Use a trailing 2.0 s history on the 25 fps analysis
timeline. For each angle, unwrap before differencing and apply a centered
three-sample median only within a continuous valid segment. Record the one
analysis-frame look-ahead. Ignore direction reversals smaller than 2 degrees.
A cycle has a starting extremum `a`, the next opposite extremum `b`, and a
return sample `c`: `abs(b-a) >= 8` degrees, `abs(c-a) <= 4` degrees, and
`0.25 <= time(c)-time(a) <= 2.0` seconds. Require movement away from `b` by at
least 2 degrees before accepting the return. A monotonic turn/tilt is not a
cycle. Emit once when the return is complete; use `c` as the next possible
cycle start so overlapping windows cannot count the same cycle repeatedly.
Pitch and yaw cycles can coexist; no claim of agreement/disagreement is made.

The head branch no longer uses `body.turn_face_offset` or the MediaPipe
face/Pose-face fallback. Retired decision inputs include
`neck_xy_px`, `face_center_xy_px`, `face_neck_offset_shoulder_widths` and
`face_skeleton_source`. A failed component makes its
dependent labels unknown, without invalidating other usable modalities or
silently switching to the former mechanism.

## Facial expression rules using HMAD

Compute HMAD's head model/residuals and AU table, then run its supplied models:
**SMAD** for smiles and **EBMAD** for eyebrow movements. Keep the native tables,
interval labels and quality information alongside normalized project labels.

| Behaviour / label | Exact mapping from valid HMAD output |
| --- | --- |
| Smile level / `smile_level` | Preserve native SIS category `S0`, `S1`, `S2`, `S3` or `S4`; `X`, missing or rejected evidence is unknown. These are ordered categories, not a linear AU score. |
| Smiling / `smiling` | True for `S1`-`S4`; false for valid `S0`; otherwise unknown. `S0` means no detected smile, not emotionally neutral. |
| Laughing-smile appearance / `laughing_smile` | True only for `S4`; false for other valid SIS categories. This does not establish audible laughter. |
| Brow raising / `brow_raising` | True within an accepted EBMAD Raise interval; false in an explicitly valid no-raise interval; otherwise unknown. |
| Brow lowering/frowning / `brow_lowering` | True within an accepted EBMAD Frown interval; false in an explicitly valid no-frown interval; otherwise unknown. |

Use EBMAD's separate raise/frown outputs to preserve any overlaps; its valid
`N` no-movement class is negative and its `X`/missing quality regions are
unknown. These are temporal brow-movement annotations, so their semantics
differ from the former framewise blendshape posture scores. Do not classify
an unannotated gap as negative unless the HMAD quality/coverage data establishes
valid absence.

Retain OpenFace AU values (including AU01/02/04, AU06/12 and AU45) as review
measurements. SMAD uses its supplied multistage temporal model, rather than
the old mean-smile >= 0.45 threshold or a substitute AU12-only rule. EBMAD uses
head-motion-corrected landmark residuals, rather than the old brow >= 0.35
blendshape thresholds. Those three legacy face thresholds have been removed from the config. Do not add happiness, sadness, engagement or intention labels.

## Gaze and blink rules

OpenFace provides eye gaze vectors/angles; **BAGMAD** uses these and AU45 to
annotate blinks and reference-related gaze intervals. Head direction alone is
never substituted for eye gaze. A valid head turn can coexist with eyes looking
in the opposite direction.

| Behaviour / output | Rule |
| --- | --- |
| Blinking / `blinking` | Use accepted BAGMAD blink intervals, retaining native start/end times; valid non-blink coverage is false, invalid/uncovered coverage unknown. |
| Horizontal gaze / `gaze_left`, `gaze_right` | On usable non-blink observations, `gaze_right_deg < -15` or `> +15`, respectively. Both false inside the inclusive deadband. |
| Vertical gaze / `gaze_up`, `gaze_down` | On usable non-blink observations, `gaze_down_deg < -10` or `> +10`, respectively. Both false inside the inclusive deadband. |
| Camera-direction gaze / `gaze_toward_camera` | Usable non-blink gaze with `abs(gaze_right_deg) <= 15` AND `abs(gaze_down_deg) <= 10`. This is an angular proxy, not proven eye contact. |
| Native BAGMAD state / `bagmad_gaze_state` | Preserve its `eye contact` / `no eye contact` prediction as a model output, subject to tracking quality, with the model/reference provenance. |
| Reference gaze / `gaze_toward_reference`, `gaze_away_reference` | Map the two BAGMAD states only after the intended reference and model's inferred gaze region have been checked against annotated footage for that camera. Otherwise both semantic labels remain unknown. |

Gaze-direction thresholds are project settings:
`gaze.horizontal_deg = 15` and `gaze.vertical_deg = 10`; they are not BAGMAD
model thresholds. Horizontal and vertical labels can coexist for diagonal
looks. During BAGMAD blink intervals, coarse direction and reference-gaze
labels are unknown; preserve the blink event and any raw model output.

BAGMAD estimates a local gaze reference from the recording under assumptions
about conversational eye contact. That region is not automatically the camera,
students, board or a particular group in a classroom. Default
`gaze.reference_validated = false`; keep directional gaze available while
reference-gaze labels remain unknown. Save the native gaze-region parameters
and calibration description. No `looking_at_board`, `looking_at_class`, exact
gaze point or student/group target is assigned without the corresponding
camera/target calibration. Missing or degenerate reference statistics invalidate
reference predictions independently of usable raw directional gaze. Blink and
gaze-model statuses are exported separately so a failed reference fit does not
discard a successful blink solution or otherwise usable coarse directional gaze.

### Timing and output changes for review

- Body rules retain the existing 0.25 s confirmation. Apply it also to custom
  `turning*` and coarse directional/camera-gaze rules; negative/unknown clears
  confirmation immediately. Their confirmed events exclude the initial delay.
- SMAD, EBMAD and BAGMAD already return temporal annotations. Preserve their
  accepted intervals without a second 0.25 s filter that could erase short
  blinks/brow movements. For these labels, raw/confirmed frame states are the
  same after quality masking. Keep the full unmasked native annotations too.
- Completed nod/shake cycles are discrete events, with start `a`, peak `b`,
  completion `c` and duration recorded in seconds. Mark their frame label only
  on the source-frame interval containing `c`, without the 0.25 s filter.
  Store the cycle's true span separately from its completion-frame display.
- Frame CSV keeps existing body columns and adds native OpenFace frame/time,
  confidence/success, per-modality validity/reason, normalized head/gaze angles,
  AU measurements, SIS category, BAGMAD state and new raw/confirmed labels.
  Missing numeric values are blank/null, not zero. Export native radians and
  estimated translations with explicit units; do not claim calibrated room
  distances from default camera estimates.
- Event CSV distinguishes `confirmed_posture`, `hmad_interval` and
  `completed_cycle` timing, includes label source, and uses exclusive end
  timestamps. Preserve model interval boundaries even when the rendering frame
  grid is coarser; record completion time separately for cycles.
- Review video uses the existing body/mask/hand overlays, OpenFace facial
  landmarks and gaze arrows when valid, smile level and the new labels. It
  retains original source FPS/resolution and marks unavailable modalities as
  unknown. It must not draw stale face landmarks or gaze arrows.
- Metadata records OpenFace version/commit and binary/model hashes, HMAD
  commit/model hashes, R/package versions, actual thresholds, reference settings,
  source/analysis time mapping, processing status and valid-data coverage.
  Keep native OpenFace CSV, HMAD tables and available ELAN `.eaf` annotations.

Verification covers: OpenFace/R/HMAD availability and model files;
actual AU/landmark/gaze schema; head/gaze sign and degree conversion; the
29.97-to-25 fps mapping; absent faces/eyes, low confidence and rear views;
brief blinks and HMAD `X` regions; smile/brow interval mappings; sideways lean
without head rotation; held turns versus completed nod/shake cycles; independent
body operation; and synchronized CSV/events/video on the default 30 s clip.
Compare samples to manual labels before reporting tracking accuracy. These
checks are described with their execution status in README.md. A successful run
is execution evidence, not an accuracy benchmark.

### Upstream basis and implementation constraints

- [OpenFace 2.2 repository](https://github.com/TadasBaltrusaitis/OpenFace) and
  [output format](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format):
  landmarks, head pose, AU and gaze measurements with native units.
- [HMAD repository](https://github.com/srauzy/HMAD), reviewed at commit
  [`227a155b4fa20970ca559e019beb47b5c3f01d61`](https://github.com/srauzy/HMAD/commit/227a155b4fa20970ca559e019beb47b5c3f01d61):
  head model/residuals, SMAD, EBMAD and BAGMAD. Runs record the actual revision used and any compatibility fixes.
- [SMAD source](https://github.com/srauzy/HMAD/blob/227a155b4fa20970ca559e019beb47b5c3f01d61/sourceR/lpl.R.dev.smad.R):
  SIS `S0`-`S4`, unreliable `X` intervals, trained temporal models and 0.04 s
  interval construction. [EBMAD source](https://github.com/srauzy/HMAD/blob/227a155b4fa20970ca559e019beb47b5c3f01d61/sourceR/lpl.R.dev.ebmad.R):
  residual-based raise/frown detection and quality masks.
- [BAGMAD source](https://github.com/srauzy/HMAD/blob/227a155b4fa20970ca559e019beb47b5c3f01d61/sourceObjectR/lpl.R.dev.hmad.bagmadObject.R):
  blink handling and the two native gaze states;
  [local-reference model](https://github.com/srauzy/HMAD/blob/227a155b4fa20970ca559e019beb47b5c3f01d61/sourceObjectR/lpl.R.dev.hmad.bagmadModelObject.R).
- [HMAD OpenFace adapter](https://github.com/srauzy/HMAD/blob/227a155b4fa20970ca559e019beb47b5c3f01d61/sourceR/lpl.R.dev.openFaceOutputAnalysis.R):
  confidence filtering, head-model residuals and angle conversion.
- [HMAD command guide](https://github.com/srauzy/HMAD/wiki/Run-HMAD-commands):
  its legacy OpenFace launcher is Windows-only. On this macOS project, invoke
  the OpenFace executable independently and run R/HMAD on the resulting CSV.
  The local build helper records the macOS compiler compatibility adjustments.
- OpenFace's [research-use terms](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/Copyright.txt)
  and HMAD's [GPL license](https://github.com/srauzy/HMAD/blob/master/LICENSE)
  must be recorded with attribution in `THIRD_PARTY.md` during implementation.

## Moving toward or away from the camera

These labels describe camera-depth movement, independently of walking, facing
or head posture. OpenFace `pose_Tz` decreases toward the camera and increases
away. It estimates head depth, so head leaning alone is insufficient evidence
of teacher locomotion.

Use a trailing 1.5 s window with at least 1.0 s of consecutive valid head-depth
and torso observations. Both shoulders and hips must be observable. Require
positive finite head depth, usable torso yaw, and camera stability. A gap over
0.25 s or invalid required observation resets the history. Compute the medians
of depth, shoulder width and shoulder-midpoint-to-hip-midpoint length in the
first and last 0.2 s of the window. For each measure, divide the end-minus-start
difference by the window-wide median and by the time between endpoint medians.
This gives fractional change per second, not meters per second.

| Label | Raw rule |
| --- | --- |
| `moving_toward_camera` | Depth fractional rate < -0.03/s AND both torso-length and shoulder-width fractional rates > +0.02/s |
| `moving_away_from_camera` | Depth fractional rate > +0.03/s AND both body-scale rates < -0.02/s |
| Both false | All three absolute rates are within their respective thresholds |
| Both unknown | Missing/insufficient evidence, changing torso orientation, or conflicting/head-only movement |

Reject the window if torso yaw or image torso tilt changes by more than 10
degrees. Apply the common 0.25 s confirmation. All settings are in
`camera_motion`. This conservative scale corroboration can miss slow movement,
movement while turning, or movement with hidden hips. A walking label is not
required, and apparent depth changes do not establish calibrated displacement.

`CameraGuard` tracks up to 150 background features on 320 x 180 grayscale
frames, excluding a dilated person mask when available. Require at least 12
tracked points and a RANSAC affine inlier ratio >= 0.6. Camera-relative labels
are unknown if summed absolute log scale changes reach 0.01, or summed affine
translation reaches 0.02 image diagonals, over 1.5 s. A mean frame difference
>= 0.28 of the grayscale range marks a possible cut and resets histories/gates.
Low texture or moving backgrounds can cause conservative rejection; these are
heuristic camera checks, not guaranteed camera-motion segmentation.

## Teaching-aid detection and holding

Select `--teaching-aids on` or `--teaching-aids off` before analysis. The default
is **off**. Off mode skips object dependency checks, model loading and inference;
holding labels stay unknown, and the review video reports teaching aids as
disabled. Body, hand, head, face and gaze tracking remain enabled according to
their own settings. The selected mode is printed at startup and recorded in
metadata as `teaching_aid_detection`. `--no-objects` is a compatibility alias for
off. Combining on with `--no-hands` is rejected before processing.

Grounding DINO Tiny detects these configured object categories in crops around
the teacher's hands: pen, pencil, marker, chalk, pointer, remote, book, paper
and tablet. OpenFace/HMAD do not detect objects. The default model revision is
pinned; original text prompts, detector threshold (0.35), text threshold (0.25),
resolved model revision and library versions are recorded. Detection runs on
each processed source frame with usable teacher-hand landmarks. CPU inference
can be slow. The model is used without classroom-specific fine-tuning, and
small/hidden pens or chalk may be missed.

Holding is a **visual relationship heuristic**, using these requirements:

1. Both teacher shoulders and the corresponding Pose wrist must be observable.
   Match a complete, finite, in-frame 21-point Hand result to the closest Pose
   wrist, within 0.5 shoulder widths. Require Hand classification confidence
   >= 0.7, at least a 0.10-shoulder-width separation from the other wrist match,
   exactly one matching hand per side and palm width >= 8 pixels.
2. Detect an object inside a crop centered on the hand. Crop radius is the
   greater of 3 palm widths and 0.65 shoulder widths. Require the object category
   to be in `objects.classes` and its detector score >= 0.35.
3. Require grasp evidence: thumb/index tips closer than 0.65 palm widths OR at
   least one curled finger. A finger is curled when its MCP-PIP-tip angle is
   < 155 degrees or wrist-tip distance < 1.05 wrist-PIP distance. An empty fist
   alone cannot trigger holding.
4. At least two of the five fingertips must lie within the object box or within
   0.30 palm widths of it. Proximity alone without grasp evidence is insufficient.
   Two different plausible object types within 0.10 detector score of each other
   make that hand unknown.
5. Over successive observations, the selected category must remain the same and
   object-center displacement relative to hand center must change by <= 0.65
   palm widths. For a nearly stationary hand (movement <= 0.25 palm widths),
   also require consecutive object boxes to overlap with IoU > 0.1. This allows
   stationary holding and checks co-movement when the hand moves.
6. Confirm each hand after 0.4 s of consecutive candidate evidence. Missing,
   ambiguous or failed observations immediately clear that hand's confirmation;
   a sampling gap > 0.25 s resets tracking. Object/category changes restart it.
   The global 0.25 s posture gate is not applied again.

| Output | Meaning |
| --- | --- |
| `left_holding_object`, `right_holding_object` | Confirmed holding evidence for the teacher's anatomical hand, assigned by Pose-wrist association |
| `holding_pen`, `holding_pencil`, `holding_marker`, `holding_chalk`, `holding_pointer`, `holding_remote`, `holding_book`, `holding_paper`, `holding_tablet` | At least one confirmed hand holds a candidate of that configured type |
| `holding_teaching_aid` | At least one hand has confirmed holding evidence for a configured aid |
| `holding_status` | `holding`, `pending`, `no_target_object_observed`, or `unknown` |
| `held_objects` | Confirmed type, hand, detector score and bounding box; two hands may support the same object |

A missing/occluded/unassociated hand is unknown, including when `--no-hands`
is used. Negative status requires all relevant hands to be observable and no
configured target candidate to be observed; it does not prove empty hands or
exclude objects outside the configured vocabulary. Object detections without
holding evidence remain candidates in `measurements_json`. Left/right hands
are independent. Holding a marker does not establish writing, and a held book
does not establish instructional use. Cross-model teacher association still
assumes one selected teacher; use a fixed `--teacher-roi` crop when necessary.

## Configure your classroom

`classroom_config.example.json` contains the body, face, head, gaze, camera-motion
and teaching-aid settings. Facing defaults are set for the back-of-classroom
camera. Zones, groups and desks remain empty
until annotated; those location-dependent labels stay unknown.

Copy it to `classroom_config.json`. Coordinates are normalized to the **selected,
unmirrored teacher crop** (the full source image by default): x from left to right and y from top to bottom. Draw
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
  OpenFace/HMAD measurements, smile level, holding status and held-object types.
- `*_events.csv`: confirmed episodes with timestamps and durations.
- `*_windows.csv`: per-behaviour duration, coverage and counts in consecutive
  source-time windows (30 seconds by default).
- `*_metadata.json`: full effective settings, model hashes and detection counts.
- `*_preview.jpg`: a representative frame, preferring a visible skeleton and
  confirmed labels over opening titles.

The default test is the first 30 seconds of the Calculus WebM in the project
root. Override the source with `--input`, duration with `--seconds`, or use
`--full-video` to process the entire source. Default input/model/output paths
resolve relative to the project rather than your current directory. The Python rendering/merge stage streams frames with bounded landmark history.
HMAD is upstream R sequence analysis and loads its selected clip into memory;
use manageable source clips for long recordings. Use a new output directory for each run; existing
results are protected against replacement. Interrupted/failed runs can leave
partial MP4/CSV files; only a successful run writes the final metadata JSON.

### Consecutive gesture duration windows

`--window-seconds` defaults to 30 and must be positive and finite. Windows start
at source time zero, are nonoverlapping and half-open: `[0,30)`, `[30,60)`, etc.
The last window ends at the selected analysis duration. `--seconds` caps the
summary exactly, excluding an encoded fractional-frame tail; `--full-video`
uses the full decoded duration. Recognition histories and confirmation gates
continue across boundaries. Aggregation happens after recognition finishes;
OpenFace/HMAD remains offline sequence processing.

The window CSV has one row per window/behaviour. `active_seconds` is the union
of confirmed event intervals inside that window, intersected with observable
frame intervals `[frame_index/fps, (frame_index+1)/fps)`. Observability requires
a nonblank confirmed frame state. This retains native subframe facial timings
while excluding unknown periods. Completed nod/shake cycles contribute their
start-to-completion motion duration, not just the completion-frame display.
Cycles completing outside the selected half-open prefix contribute neither
duration nor count, even when imported native inference covers a longer clip.
Adjacent smile categories and duplicate/overlapping positives are merged.

`event_count` counts each observed posture/facial episode once, in the window
containing its start. Unknown gaps split observed episodes. Completed cycles
count individually in the window containing completion (`count_basis` is
`cycle_completion`; other labels use `episode_start`). `events_overlapping`
counts events with active observable time in each window, so a boundary-crossing
event can overlap both windows while contributing only one `event_count`.

`inactive_seconds = observed_seconds - active_seconds` and
`unknown_seconds = window_duration_s - observed_seconds`.
`active_percent_observed = 100 * active_seconds / observed_seconds`.
Activity and counts are blank for wholly unobserved behaviour/windows, including
disabled teaching-aid detection. Exported seconds use six decimal places. Labels
can co-occur; their durations are not mutually exclusive categories.

`window_summary.py --metadata PATH` can regenerate this CSV from existing frame
and event evidence without loading recognition models. It replaces only its
window CSV, and rejects incomplete/nonconsecutive frames or invalid event times.

API references: [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
for skeleton/world landmarks and segmentation;
[Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
for the retained standalone Face utility; the active face backend uses OpenFace/HMAD.
