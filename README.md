# Classroom behaviour extraction

The pipeline combines **OpenFace 2.2 + HMAD**, MediaPipe Pose/Hands and
**Grounding DINO** to annotate a selected teacher's visible behaviour:

- Head turns, completed nods and head shakes.
- Smile intensity, eyebrow raising/frowning, blinks and coarse gaze direction.
- Movement toward/away from a fixed camera, requiring corroborating torso scale.
- Holding configured teaching aids: pen, pencil, marker, chalk, pointer, remote,
  book, paper or tablet.
- Existing body, arm, finger, hand-motion and classroom-location behaviours.

The old face-to-neck head-turn heuristic and MediaPipe blendshape expression
rules are removed from `main.py`. Facial analysis is offline and uses the native
OpenFace executable and HMAD's R models. Teaching-aid holding and camera-relative
movement are project heuristics awaiting accuracy validation on annotated clips.
A smile does not establish happiness; holding a marker does not establish writing.

## Run

Activate the project environment, then check tools and models:

```sh
source .venv/bin/activate
python main.py --check-dependencies
python main.py --output-dir outputs/my_first_tracking_run
```

The default input is **Calculus 1 Lecture 0.1: Lines, Angle of Inclination, and
the Distance Formula**, the WebM in the project root. The default duration is
30 seconds. Use a new empty output directory for each run.

**Teaching-aid detection defaults to OFF.** Choose the mode before starting:

```sh
# Body, hands, head, face and gaze; object detector stays unloaded.
python main.py --teaching-aids off --output-dir outputs/body_face_run

# Enable the additional teaching-aid detector (slower).
python main.py --teaching-aids on --objects-device mps --output-dir outputs/with_teaching_aids
```

Omitting `--teaching-aids` also selects `off`. The selected mode is printed
before analysis, recorded in metadata, and shown as disabled in the review
video when off. Off mode does not check object-model dependencies, load its
weights, or run object inference. The existing `--no-objects` flag remains an
alias for `--teaching-aids off`. Body, hand and facial extraction still run.

```sh
python main.py --input "/path/to/classroom.mp4" --seconds 30 --output-dir outputs/classroom_30s
python main.py --full-video --output-dir outputs/full_lecture
python main.py --config classroom_config.example.json --show-scene --output-dir outputs/scene_run
python main.py --view mask --output-dir outputs/mask_run

# Fixed crop containing a single teacher; normalized x0 y0 x1 y1.
python main.py --teacher-roi 0.1 0.05 0.6 0.95 --output-dir outputs/teacher_crop

# Explicitly disable individual branches for a diagnostic run.
python main.py --no-objects --output-dir outputs/without_objects
python main.py --no-face --no-objects --output-dir outputs/body_only
```

`--no-face` makes head/face/gaze/depth-dependent labels unknown. `--no-hands`
also makes object holding unknown. No branch silently falls back to the old
facial mechanism. Missing required tools fail preflight; individual HMAD model
failures are reported in metadata and their dependent labels remain unknown.

When `--teaching-aids on` is selected, Grounding DINO runs on hand crops for every source frame with usable teacher
hands. CPU inference can be slow. `--objects-device mps` or `cuda` selects a
supported accelerator explicitly. An occluded/tiny pen can remain undetected;
`no_target_object_observed` does not prove the teacher has empty hands.
Enabling teaching aids together with `--no-hands` is rejected before analysis,
because holding detection requires hand tracking.

## Setup

Python body extraction was originally verified with Python 3.12 on macOS Apple
Silicon. The additional Python packages live in this project's isolated `.venv`;
native dependencies live in `.runtime/`, and upstream sources in `external/`.
These directories and downloaded model weights are excluded from Git.

```sh
conda activate ureca
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-objects.txt

# macOS arm64: build pinned OpenFace and install isolated R + matrixcalc.
python build_native.py
python setup_tracking.py --download-hmad --download-object-model
python main.py --check-dependencies
```

For body/facial runs only, `requirements.txt` is sufficient for the Python
dependencies; `requirements-objects.txt` and the Grounding DINO download are
needed when enabling teaching aids. OpenFace/R/HMAD are still required for faces.

`build_native.py` requires conda and Xcode Command Line Tools. It installs native
libraries locally, builds dlib/OpenFace, downloads the four CE-CLM landmark
experts, and repairs conda R's unquoted path assignments for this workspace's
spaces. The dlib source build includes one explicit template-argument adjustment
for modern Apple Clang. Native setup requires downloads and substantial disk
space; the build helper reuses existing matching checkouts/runtimes.

The default tools resolve to `external/OpenFace/build/bin/FeatureExtraction`
and `.runtime/r/bin/Rscript` when present. On another system, follow the
[official OpenFace installation guide](https://github.com/TadasBaltrusaitis/OpenFace/wiki),
install R with `matrixcalc`, and supply paths:

```sh
python main.py --openface-bin "/path/to/FeatureExtraction" \
  --hmad-dir "/path/to/HMAD" --rscript "/path/to/Rscript" \
  --output-dir outputs/native_custom_paths
```

MediaPipe Pose and Hand models are documented in [models/README.md](models/README.md).
The former MediaPipe Face model is not required by this pipeline. Attribution,
upstream revisions and license notes are in [THIRD_PARTY.md](THIRD_PARTY.md).
On macOS, MediaPipe can require terminal access to graphics services even for
CPU inference.

## Native processing and importing prior results

OpenFace and HMAD run on a **25 fps analysis copy**, preserving playback speed.
The source video retains its original FPS/resolution for body/hand/object
analysis and rendering. Results join by timestamp, with explicit source and
analysis frame numbers. This avoids HMAD's hard-coded 0.04 s interval assumption
misinterpreting the approximately 29.97 fps lecture video.

Native measurements and R sequence inference are retained under `native/`.
HMAD runs in a fresh copy of its sources/models under each output directory;
the pinned checkout is unchanged. The adapter normalizes the nonportable R
encoding alias `UTF8` to `UTF-8` in that copy, recording affected files. Empty
zero-duration EBMAD boundaries are counted and ignored; reversed or overlapping
intervals are rejected.

To reuse a matching zero-start 25 fps OpenFace CSV:

```sh
python main.py --openface-csv "/prior/native/processed/analysis.csv" \
  --no-objects --output-dir outputs/reuse_measurements
```

To reuse HMAD inference too, add `--hmad-results /prior/native/hmad_results`.
Those exports must include the adapter's `manifest.json` and canonical TSV
files; source hash, OpenFace CSV hash, crop and analysis FPS must match. Raw
HMAD tables are retained for review, but are not accepted as this canonical
import format without conversion. Importing prior results still requires the
matching source video. It does not install OpenFace or R on the importing host.
Provenance-matched native exports may cover a longer source prefix than the
requested render duration. Frame-level coverage in metadata then describes the
available native analysis context, while `frames_with_face` counts the rendered
source frames. Native event times are clipped to the rendered duration.

## Configuration and rules

[BEHAVIOUR_RULES.md](BEHAVIOUR_RULES.md) defines each rule, threshold, timing,
coordinate convention, missing-data policy and limitation. Copy
[classroom_config.example.json](classroom_config.example.json) to your own file
and adjust it. Its sections are `body`, `face`, `head`, `gaze`, `camera_motion`,
`objects`, `scene` and `confirmation_s`.

Head/gaze directions mean image left/right in unmirrored input. The default
head signs follow OpenFace's rotation matrix: positive raw yaw turns toward
image left, and positive raw pitch tilts downward. Validate direction settings
against your footage before drawing research conclusions. Torso facing uses
separate MediaPipe world geometry: the back-of-classroom camera defaults to
class-facing toward the camera (0 degrees), board-facing away (180 degrees).

BAGMAD's `eye contact` output is retained as a model prediction. Semantic
`gaze_toward_reference` / `gaze_away_reference` labels remain unknown until
`gaze.reference_validated` is true and a calibration description is supplied.
A statistical gaze region is not automatically the students, board or camera.

Use a fixed camera and a single selected teacher. The system does not perform
active-speaker selection or persistent teacher re-identification. A fixed
`--teacher-roi` can exclude other people. Scene polygons and exported object
boxes are relative to the selected crop, which is the full image by default;
metadata contains the crop offset for conversion to source-image coordinates.
Zones, groups and desks require annotations. Missing legs prevent rules such
as walking and standing. Python merging streams frames, while upstream HMAD
loads the selected sequence into R memory; use manageable clips for long videos.

## Outputs

Each successful run writes:

| Output | Contents |
| --- | --- |
| `*_behaviours.mp4` | Silent source-resolution/FPS video, body/mask/hand overlays, native facial landmarks/gaze arrows, confirmed objects and labels |
| `*_frames.csv` | Raw/confirmed states, smile category, holding status/types, per-modality evidence and native AU/pose/gaze measurements |
| `*_events.csv` | `confirmed_posture`, native `hmad_interval` and `completed_cycle` records, with source and exclusive end times |
| `*_metadata.json` | Effective settings, tool/model/source hashes, model revisions, native component status, coverage and crop/time mapping |
| `*_preview.jpg` | Representative rendered frame |
| `face_analysis.jsonl` | Native face measurements joined to HMAD labels on the 25 fps analysis timeline |
| `face_annotations.jsonl`, `head_cycles.jsonl` | Quality-masked native intervals and completed head-cycle details |
| `native/` | Analysis clip, original OpenFace CSV/logs, canonical HMAD exports, working/native tables and available ELAN annotations |

CSV uses `1` for true, `0` for false, blank for unknown. Native HMAD intervals
and completed cycles bypass the common 0.25 s posture gate. Holding uses its own
0.4 s confirmation. Event CSV rows are emitted as available, not globally sorted;
use timestamps when ordering them. Interrupted runs can leave partial outputs;
only a successful run writes the final metadata file.

## Verification

```sh
python -m unittest discover -s tests -v
python main.py --help
python main.py --check-dependencies
```

The tests cover existing body/hand geometry plus native timestamp joins, 25 fps
validation, HMAD quality/unknown masks, short blinks, head rotation and completed
cycles, camera zoom/cuts, head-only leaning, hand/object contact, ambiguity,
confirmation resets, and imported native results merged into an encoded video.
Runtime verification results are recorded in generated output directories.
Successful execution does not establish detection accuracy; annotated classroom
clips are still needed to measure errors and tune thresholds.

Verified on 12 November 2026:

| Check | Result |
| --- | --- |
| Automated tests | 45 passed, including default-off and explicit on/off detector selection |
| Native OpenFace extraction | 751 analysis frames at 25 fps covering the 30-second lecture selection |
| Current HMAD bridge | Head model, AUs, smiles, brows, blink model, gaze model and ELAN export all succeeded; see `outputs/native_current_verified/` |
| Native/body video run | 900 source frames decoded and matched 900 CSV rows, 30.03 s; see `outputs/openface_hmad_30s_final/verification.json` |
| All-branches video run | 90 frames / 3.003 s with real Grounding DINO inference; see `outputs/all_tracking_3s/verification.json` |

The all-branches sample produced object candidates but **no confirmed holding
episode**. Camera-relative movement stayed unknown because the footage did not
provide sufficient consistent torso evidence. These are execution/coverage
results, not accuracy claims. The 3-second all-branches CPU run took about
210 seconds on this workspace, so longer object-enabled runs can take much
longer than playback. Existing native measurements were reused for rendering;
the original OpenFace extraction and fresh current R bridge were also tested
separately. Historical MediaPipe-only outputs are not validation of this backend.
A real Grounding DINO inference on the Apple M4 MPS device also completed
successfully; that smoke check is in `outputs/object_model_smoke/mps_result.json`.
