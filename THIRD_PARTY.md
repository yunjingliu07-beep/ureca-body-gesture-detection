# Third-party code and models

## MediaPipe

The extractor uses the MediaPipe Python package and Google's official model
bundles. Model binaries are not committed here; download URLs and checksums
are documented in [models/README.md](models/README.md).

The project also uses NumPy and OpenCV. The standalone gesture rules in
`body_gestures.py` are implemented in this project and have no game-controller
or keyboard-automation dependency.
