"""Resolve first-party stack tools from repository and linked-worktree layouts."""

from __future__ import annotations

from pathlib import Path


def metis_scripts(source_file: str) -> Path:
    """Find the Atlas Metis scripts directory from a RITK test module."""
    for root in Path(source_file).resolve().parents:
        candidates = (
            root / "repos" / "metis" / "scripts",
            root.parent / "metis" / "scripts",
        )
        for candidate in candidates:
            if (candidate / "browser_canvas.py").is_file():
                return candidate
    raise FileNotFoundError("Atlas Metis browser_canvas.py is not reachable from RITK")
