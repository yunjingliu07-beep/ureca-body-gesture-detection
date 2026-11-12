"""Run the OpenFace 2.2 executable and R/HMAD on a time-preserving 25 fps clip.

Commands use argument arrays, fresh work directories and captured logs. Existing
CSV/HMAD exports can be imported without installing native tools on this host.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2


HMAD_REVISION = "227a155b4fa20970ca559e019beb47b5c3f01d61"
ANALYSIS_FPS = 25.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command, log, cwd=None):
    print(f"Running {Path(str(command[0])).name}; log: {log}", flush=True)
    with Path(log).open("w") as output:
        result = subprocess.run([str(v) for v in command], cwd=cwd, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed (exit {result.returncode}); see {log}")


def resolve_executable(value, label):
    resolved = shutil.which(str(value))
    if resolved is None:
        raise ValueError(f"Missing {label}: {value}. See README.md native-backend setup, or supply existing exports.")
    return Path(resolved).resolve()


def validate_openface_csv(path, duration_s, fps=ANALYSIS_FPS, allow_longer=False):
    required = {"frame", "timestamp", "confidence", "success", "pose_Rx", "pose_Ry", "pose_Rz", "pose_Tz",
                "gaze_angle_x", "gaze_angle_y", "AU45_r"}
    count = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        reader.fieldnames = [s.strip() for s in reader.fieldnames or []]
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"OpenFace CSV missing columns: {sorted(missing)}")
        for row in reader:
            t = float(row["timestamp"])
            if not math.isfinite(t) or abs(t - count / fps) > 0.002:
                raise ValueError(f"OpenFace CSV must start at 0 and use 25 fps; invalid time at row {count+1}")
            if int(float(row["frame"])) != count + 1:
                raise ValueError("OpenFace frame indices must be sequential and one-based")
            count += 1
    mismatch = count / fps < duration_s - 1 / fps - 1e-5 if allow_longer else abs(count / fps - duration_s) > 1 / fps + 1e-5
    if count == 0 or mismatch:
        raise ValueError(f"OpenFace duration {count/fps:.3f}s does not match selected video {duration_s:.3f}s")
    return count


def prepare_native(args, work, duration_s, roi_px):
    """Return the original OpenFace CSV, canonical HMAD tables and provenance."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    manifest = {"backend": "OpenFace 2.2 + HMAD", "analysis_fps": ANALYSIS_FPS,
                "source": str(args.input.resolve()), "source_sha256": sha256(args.input),
                "duration_s": duration_s, "roi_px": list(roi_px), "source_start_s": 0.0,
                "camera_intrinsics": "Not supplied to this adapter; generated OpenFace runs use default estimates"}
    if args.openface_csv:
        csv_path = work / "openface.csv"
        shutil.copy2(args.openface_csv, csv_path)
        manifest["openface_input"] = str(args.openface_csv.resolve())
    else:
        executable = resolve_executable(args.openface_bin, "OpenFace FeatureExtraction")
        ffmpeg = resolve_executable(args.ffmpeg, "ffmpeg")
        x0, y0, x1, y1 = roi_px
        analysis = work / "analysis.avi"
        run_logged([ffmpeg, "-nostdin", "-v", "error", "-i", args.input.resolve(), "-t", duration_s,
                    "-vf", f"crop={x1-x0}:{y1-y0}:{x0}:{y0},fps=25", "-an", "-c:v", "mjpeg",
                    "-q:v", "2", analysis], work / "ffmpeg.log")
        cap = cv2.VideoCapture(str(analysis))
        analysis_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        analysis_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if abs(analysis_fps - ANALYSIS_FPS) > 1e-5 or abs(analysis_count / 25 - duration_s) > .04 + 1e-5:
            raise RuntimeError("Analysis-copy frame rate / duration validation failed")
        processed = work / "processed"
        run_logged([executable, "-f", analysis, "-out_dir", processed, "-of", "analysis.csv",
                    "-2Dfp", "-3Dfp", "-pdmparams", "-pose", "-aus", "-gaze"],
                   work / "openface.log", cwd=executable.parent)
        csv_path = processed / "analysis.csv"
        if not csv_path.is_file():
            raise RuntimeError(f"OpenFace did not produce {csv_path}; see {work / 'openface.log'}")
        manifest.update(openface_binary=str(executable), openface_binary_sha256=sha256(executable))
        checkout = executable.parents[2]
        if (checkout / ".git").is_dir():
            manifest["openface_revision"] = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        manifest["openface_asset_sha256"] = {
            str(p.relative_to(executable.parent)): sha256(p)
            for directory in ("model", "AU_predictors", "classifiers")
            for p in sorted((executable.parent / directory).rglob("*")) if p.is_file()}
    manifest["analysis_frames"] = validate_openface_csv(csv_path, duration_s, allow_longer=bool(args.hmad_results))
    manifest["analysis_available_s"] = manifest["analysis_frames"] / ANALYSIS_FPS
    manifest["openface_csv_sha256"] = sha256(csv_path)
    results = work / "hmad_results"
    if args.hmad_results:
        supplied = Path(args.hmad_results).resolve()
        prior = json.loads((supplied / "manifest.json").read_text())
        for field in ("openface_csv_sha256", "source_sha256", "roi_px", "analysis_fps"):
            if prior.get(field) != manifest[field]:
                raise ValueError(f"HMAD import provenance mismatch for {field}")
        if prior.get("analysis_available_s", prior.get("duration_s", manifest["analysis_available_s"])) < duration_s - .04 - 1e-5:
            raise ValueError("Imported HMAD analysis does not cover the selected source prefix")
        shutil.copytree(supplied, results)
        manifest["hmad_revision"] = prior.get("hmad_revision", "unreported")
        for field in ("openface_binary_sha256", "openface_revision", "openface_asset_sha256",
                      "hmad_asset_sha256", "hmad_compatibility", "hmad_bridge_sha256", "r_version"):
            if field in prior:
                manifest[field] = prior[field]
    else:
        rscript = resolve_executable(args.rscript, "Rscript")
        hmad = Path(args.hmad_dir).resolve()
        for name in ("sourceR", "sourceObjectR", "models"):
            if not (hmad / name).is_dir():
                raise ValueError(f"Missing HMAD {name} in {hmad}; run setup_tracking.py --download-hmad")
        root = work / "hmad_workspace"
        root.mkdir()
        for name in ("sourceR", "sourceObjectR", "models"):
            shutil.copytree(hmad / name, root / name)
        # Some upstream files use the nonportable iconv alias 'UTF8'. Keep the
        # pinned checkout intact and record this compatibility edit in the copy.
        encoding_fixes = []
        for directory in ("sourceR", "sourceObjectR"):
            for source in (root / directory).glob("*.R"):
                text = source.read_bytes()
                fixed = text.replace(b'"UTF8"', b'"UTF-8"')
                if fixed != text:
                    source.write_bytes(fixed)
                    encoding_fixes.append(str(source.relative_to(root)))
        manifest["hmad_compatibility"] = {"utf8_encoding_alias_files": encoding_fixes,
                                          "webm_mime_lookup_in_bridge": True}
        project = root / "projects" / "URECA" / "OPENFACE" / "processed"
        project.mkdir(parents=True)
        # HMAD's legacy helpers expect PROJECT_NAME + '_ofo.csv'.
        shutil.copy2(csv_path, project / "URECA_ofo.csv")
        analysis = work / "analysis.avi"
        if analysis.is_file():
            shutil.copy2(analysis, project.parent.parent / "analysis.avi")
        media = project.parent.parent / "analysis.avi" if analysis.is_file() else args.input.resolve()
        (project.parent.parent / "media_path.txt").write_text(str(media) + "\n")
        results.mkdir()
        run_logged([rscript, "--vanilla", Path(__file__).with_name("hmad_bridge.R"), root, "URECA", results],
                   work / "hmad.log")
        revision_file = hmad / "REVISION"
        revision = revision_file.read_text().strip() if revision_file.exists() else "unreported"
        if (hmad / ".git").exists():
            revision = subprocess.check_output(["git", "-C", str(hmad), "rev-parse", "HEAD"], text=True).strip()
        manifest.update(hmad_revision=revision, r_version=subprocess.check_output([str(rscript), "--version"],
                        text=True, stderr=subprocess.STDOUT).strip())
        manifest["hmad_bridge_sha256"] = sha256(Path(__file__).with_name("hmad_bridge.R"))
        manifest["hmad_asset_sha256"] = {str(p.relative_to(hmad)): sha256(p)
                                        for folder in ("sourceR", "sourceObjectR", "models")
                                        for p in sorted((hmad / folder).rglob("*")) if p.is_file()}
    if not (results / "status.tsv").is_file():
        raise ValueError("Missing HMAD status.tsv; use exports from this adapter")
    with (results / "status.tsv").open() as handle:
        manifest["hmad_components"] = list(csv.DictReader(handle, delimiter="\t"))
    if not {"smile", "brows", "bagmad"}.issubset({r["component"] for r in manifest["hmad_components"]}):
        raise ValueError("Incomplete HMAD component status report")
    manifest["status"] = "ok" if all(r["status"] == "ok" for r in manifest["hmad_components"]) else "partial"
    if manifest["status"] == "partial":
        print("HMAD completed partially; failed components remain unknown. See native/manifest.json.", flush=True)
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return csv_path, results, manifest
