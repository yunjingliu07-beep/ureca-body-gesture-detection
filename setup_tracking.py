"""Check runtime tools and download pinned HMAD sources or local object weights.

Run with the project's Python environment. Downloads stay under this project;
native OpenFace and R installation instructions are in README.md.
"""

import argparse
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

from native_tracking import HMAD_REVISION, sha256


ROOT = Path(__file__).resolve().parent


def download_openface_models(directory):
    """Download the four upstream CE-CLM experts, rejecting HTML download pages."""
    from concurrent.futures import ThreadPoolExecutor
    target = Path(directory) / "lib/local/LandmarkDetector/model/patch_experts"
    if not target.is_dir():
        raise ValueError("Clone OpenFace before downloading its landmark models")
    models = {"0.25": "7na5qsjzz8yfoer", "0.35": "k7bj804cyiu474t",
              "0.50": "ixt4vkbmxgab1iu", "1.00": "2t5t1sdpshzfhpj"}

    def download(item):
        scale, identifier = item
        path = target / f"cen_patches_{scale}_of.dat"
        if path.is_file() and path.stat().st_size > 1000000:
            return path.name, sha256(path)
        url = f"https://www.dropbox.com/s/{identifier}/{path.name}?dl=1"
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            with temporary.open("rb") as handle:
                beginning = handle.read(128).lstrip().lower()
            if temporary.stat().st_size < 1000000 or beginning.startswith((b"<!doctype", b"<html")):
                raise ValueError(f"Not a binary landmark model: {url}")
            temporary.replace(path)
            return path.name, sha256(path)
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(download, models.items()))
    (Path(directory) / "landmark_model_hashes.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print(json.dumps(hashes, indent=2))


def download_hmad(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://api.github.com/repos/srauzy/HMAD/tarball/{HMAD_REVISION}"
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        archive = Path(temporary) / "hmad.tar.gz"
        request = urllib.request.Request(url, headers={"User-Agent": "URECA-tracking-setup"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        extracted = Path(temporary) / "unpacked"
        extracted.mkdir()
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                target = (extracted / member.name).resolve()
                if not target.is_relative_to(extracted.resolve()) or member.issym() or member.islnk():
                    raise ValueError("Unsafe HMAD archive member")
            bundle.extractall(extracted, filter="data")
        roots = list(extracted.iterdir())
        if len(roots) != 1 or not (roots[0] / "sourceR").is_dir():
            raise ValueError("Unexpected HMAD source archive layout")
        (roots[0] / "REVISION").write_text(HMAD_REVISION + "\n")
        (roots[0] / "DOWNLOAD.json").write_text(json.dumps({"url": url, "revision": HMAD_REVISION,
                                                            "archive_sha256": sha256(archive)}, indent=2)+"\n")
        shutil.move(str(roots[0]), destination)
    print(f"Downloaded HMAD {HMAD_REVISION} to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-hmad", action="store_true")
    parser.add_argument("--hmad-dir", type=Path, default=ROOT / "external/HMAD")
    parser.add_argument("--download-object-model", action="store_true")
    parser.add_argument("--download-openface-models", action="store_true")
    parser.add_argument("--openface-dir", type=Path, default=ROOT / "external/OpenFace")
    parser.add_argument("--objects-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--objects-revision", default="a2bb814dd30d776dcf7e30523b00659f4f141c71")
    args = parser.parse_args()
    if args.download_hmad:
        download_hmad(args.hmad_dir)
    if args.download_openface_models:
        download_openface_models(args.openface_dir)
    if args.download_object_model:
        from teaching_aids import GroundingDinoDetector
        with GroundingDinoDetector(args.objects_model, args.objects_revision, cache_dir=str(ROOT / "models/hf-cache")) as detector:
            print(json.dumps(detector.metadata, indent=2))
    from main import build_parser, dependency_report
    options = build_parser().parse_args([])
    options.hmad_dir = args.hmad_dir
    print(json.dumps(dependency_report(options), indent=2))


if __name__ == "__main__":
    main()
