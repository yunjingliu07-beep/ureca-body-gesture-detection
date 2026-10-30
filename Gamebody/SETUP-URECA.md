# Setup in the ureca environment

This checkout uses Python 3.12 on Apple Silicon. Install the compatible package
set instead of upstream's older `requirements.txt`:

```sh
conda activate ureca
cd Gamebody
python -m pip install -r requirements-ureca.txt
python -m pip check
```

The requirements use MediaPipe 0.10.21 for `mp.solutions`, one OpenCV package
(`opencv-contrib-python` also supplies `cv2`), and the author's macOS keyboard
fork. A separate `mediapipe_silicon` package is not needed. `play.py` imports the
installed keyboard package rather than expecting an untracked local clone.

Run from this directory so that the `maps/` files can be found:

```sh
python play.py --display 1 --landmarks 1
```

`--display 1` shows the mapped actions without sending keyboard presses. Add
`--input /path/to/video.mp4` to use a video instead of the webcam. Press `Esc` to
close the preview.

The original local setup saved package versions in
`../outputs/ureca-before-gamebody.txt`; local outputs are excluded from Git.
The existing `extract_speaker_pose.py` has a fallback for the landmark enum
location in this older MediaPipe version.

## Verification

Verified `play.py --help`, pose detection with 33 landmarks on a frame from the
existing video, Gamebody's frame processing in display-only mode, and a
one-second run of the original extraction script. These checks did not open
the webcam or send keyboard input. MediaPipe needs access to macOS graphics
services even for these CPU inference checks.

`pip check` reports `mediapipe 0.10.21 is not supported on this platform`.
The downloaded universal2 wheel has an Intel-only tag in its internal `WHEEL`
metadata. Its loaded native extension contains ARM64 code and inference passed
on this Apple M4. No missing dependencies or version conflicts were reported.
The wheel metadata has not been modified to suppress the warning.

The original local setup also saved final package versions in
`../outputs/ureca-after-gamebody.txt` (not committed).
