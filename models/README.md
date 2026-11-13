# Tracking model assets

The active video pipeline needs the MediaPipe **Pose** and **Hand** models below,
the four OpenFace CE-CLM patch experts and bundled AU models, HMAD's model tables,
and Grounding DINO Tiny. The MediaPipe Face model is optional for the retained
standalone helper; it is not used for head/expression/gaze analysis in `classroom_behaviour/cli.py`.

Native/model setup is documented in [README.md](../README.md). On this macOS
arm64 workspace, `python -m scripts.build_native` builds OpenFace and downloads its patch
experts; `python -m scripts.setup_models --download-hmad --download-object-model`
downloads the pinned HMAD checkout and the object checkpoint. Models stay out of
Git. Grounding DINO's cache is `models/hf-cache/`; OpenFace/HMAD assets are under
their `external/` checkouts. Runs record the model revisions and available hashes.

## MediaPipe bundles

Official versioned model bundles downloaded from Google's model storage:

From the project root, download the files before running extraction:

```sh
curl -fL https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task -o models/pose_landmarker_full.task
curl -fL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o models/face_landmarker.task
curl -fL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o models/hand_landmarker.task
shasum -a 256 models/*.task
```

Model weights are ignored by Git and are not included in the repository.

| File | Source |
| --- | --- |
| `pose_landmarker_full.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task |
| `face_landmarker.task` | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task |
| `hand_landmarker.task` | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task |

SHA-256 checksums:

```text
5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1  pose_landmarker_full.task
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  face_landmarker.task
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1  hand_landmarker.task
```

Each extraction records checksums for the actual files used. The optional
standalone Face utility enables 478 landmarks, 52 blendshapes, and facial transformation matrices. Hand
extraction enables up to two sets of 21 landmarks, world coordinates, and
handedness classifications.
