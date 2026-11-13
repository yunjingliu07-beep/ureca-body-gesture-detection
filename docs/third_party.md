# Third-party code and models

## OpenFace

[TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace),
source revision `3d4b5cf8d96138be42bed229447f36cbb09a5a29`, supplies facial
landmarks, head pose, Action Units and eye gaze. Sources, compiled binaries and
model weights are local assets under `external/OpenFace`, excluded from Git.
Runs record binary/model hashes and the detected source revision.

OpenFace states academic/nonprofit noncommercial research-use terms; commercial
licensing is separate. Respect its code/model and training-dataset terms:
[Copyright.txt](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/Copyright.txt).
Cite Baltrusaitis et al., *OpenFace 2.0: Facial Behavior Analysis Toolkit*, FG 2018,
and the relevant submodule publications listed in its README.

The local build uses dlib `19.24.2`, source revision
`39749fba5163d4e25030358f6aa8d91617bfdca2`, and its original license. The build
helper adds an explicit `<T>` template argument at one dlib call for modern
Apple Clang. OpenCV, OpenBLAS and Boost retain their own license terms.

## HMAD

[srauzy/HMAD](https://github.com/srauzy/HMAD), pinned at
`227a155b4fa20970ca559e019beb47b5c3f01d61`, supplies head-model residuals, SMAD
smile intensity, EBMAD eyebrow movements and BAGMAD blink/gaze annotations.
Upstream R sources/models and their original GPL-3.0-or-later notices remain
in a separate local checkout. `classroom_behaviour/hmad_bridge.R` calls the upstream functions;
the Python adapter does not replace their learned models with AU thresholds.

HMAD is executed in a fresh working copy. The adapter changes the encoding alias
`UTF8` to `UTF-8` in that copy for compatibility with the tested R/iconv build;
affected paths are recorded in the native manifest. The pinned checkout stays
unchanged. Zero-duration quality boundaries are ignored only when joining
intervals, with counts in metadata; original native tables remain available.
The bridge also extends HMAD's MIME lookup with `video/webm` for ELAN media links
and keeps optional annotation export separate from blink/gaze inference status.
Redistribution of HMAD or derivatives must respect its GPL terms and notices.

Cite Rauzy and Goujon (2018), *Automatic annotation of facial actions from a
video record: The case of eyebrows raising and frowning*, and Rauzy and Amoyal
(2023), *Automatic tool to annotate smile intensities in conversational
face-to-face interactions*, along with the repository as requested by its wiki.

## Grounding DINO and Transformers

[IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO),
through Hugging Face Transformers, provides open-vocabulary object candidates.
The default checkpoint is `IDEA-Research/grounding-dino-tiny`, revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`. The repository and
[model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny) declare
Apache-2.0 terms; retain applicable notices for code and downloaded weights.
The cache is local under `models/hf-cache`, excluded from Git. Runs record the
resolved revision, prompts/settings and library versions. Holding inference
is this project's additional heuristic, not a native Grounding DINO output.

Cite Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training for
Open-Set Object Detection*. The project also uses PyTorch, torchvision,
Transformers, tokenizers and safetensors under their respective licenses.

## MediaPipe and other Python libraries

The body/hand extractor uses MediaPipe and Google's official Pose/Hand model
bundles. Model URLs and checksums are in [models/README.md](../models/README.md).
The former MediaPipe Face utility/model is retained for standalone helpers;
it is not the active head/expression/gaze mechanism in `classroom_behaviour/cli.py`.

The project also uses NumPy, OpenCV, Pillow and SciPy. Standalone gesture rules
in `classroom_behaviour/body_gestures.py` have no game-controller or keyboard-automation dependency.
FFmpeg creates the analysis copy; R and matrixcalc run HMAD. Respect each
package's license when redistributing environments or executables.
