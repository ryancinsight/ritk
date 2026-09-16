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
        "gallery.html",
        "gallery.css",
        "gallery.js",
        "styles.css",
        "bootstrap.js",
        "metis_web.js",
        "metis_web_bg.wasm",
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
    for name in (
        "gallery.html",
        "gallery.css",
        "gallery.js",
        "styles.css",
        "bootstrap.js",
        "metis_web.js",
        "metis_web_bg.wasm",
    ):
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "assets", destination / "assets")
    shutil.copytree(source / "consumer", destination / "consumer")


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
