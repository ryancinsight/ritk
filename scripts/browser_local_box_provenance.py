"""Artifact and source provenance for the RITK browser regression."""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
from typing import Any


def file_digest(path: pathlib.Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def bundle_manifest(bundle: pathlib.Path) -> dict[str, dict[str, Any]]:
    names = (
        "index.html",
        "embed.html",
        "embed.js",
        "gallery.html",
        "gallery.css",
        "gallery.js",
    )
    paths = [bundle / name for name in names if (bundle / name).is_file()]
    consumer = bundle / "consumer"
    if consumer.is_dir():
        paths.extend(path for path in consumer.rglob("*") if path.is_file())
    manifest = {
        path.relative_to(bundle).as_posix(): file_digest(path)
        for path in sorted(paths)
    }
    for required in ("consumer/ritk_snap.js", "consumer/ritk_snap_bg.wasm"):
        if required not in manifest:
            raise RuntimeError(f"built browser bundle is missing {required}")
    return manifest


def copy_bundle(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.mkdir()
    shutil.copytree(source / "assets", destination / "assets")
    shutil.copytree(source / "consumer", destination / "consumer")
    # Inherit the shipped content policy when exercising the embeddable API in isolation.
    head = (source / "index.html").read_text(encoding="utf-8").split("</head>")[0]
    axes = ("axial", "coronal", "sagittal")
    canvases = "".join(f'<figure><canvas id="ritk-snap-{axis}"></canvas></figure>' for axis in axes)
    (destination / "embed.html").write_text(head + '</head><body><main id="metis-app" hidden></main>'
        + '<output id="ready"></output><section class="gallery-views">' + canvases
        + '</section><script type="module" src="embed.js"></script></body></html>', encoding="utf-8")
    (destination / "gallery.css").write_text(
        'body,figure{margin:0}.gallery-views{display:flex}.gallery-views canvas{width:1024px!important;height:1024px!important}',
        encoding="utf-8")
    (destination / "embed.js").write_text(
        'import init,{start_web_orthogonal_canvases} from "./consumer/ritk_snap.js";'
        'await init();start_web_orthogonal_canvases("ritk-snap-axial","ritk-snap-coronal","ritk-snap-sagittal");'
        'document.querySelector("#ready").textContent="Ready.";', encoding="utf-8")



def repository_state(path: pathlib.Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True, timeout=30
    ).strip()
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", "."], cwd=path, timeout=30
    )
    untracked_output = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=path,
        timeout=30,
    )
    untracked = {}
    for relative in filter(None, untracked_output.decode("utf-8").split("\0")):
        candidate = path / relative
        if candidate.is_file():
            untracked[relative] = file_digest(candidate)
    return {
        "head": head,
        "tracked_diff_bytes": len(diff),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked": untracked,
    }
