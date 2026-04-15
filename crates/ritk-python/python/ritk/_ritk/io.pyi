"""Type stubs for the ``_ritk.io`` submodule (image and transform I/O)."""

from __future__ import annotations

from typing import Any

from ritk._ritk.image import Image

def read_image(path: str) -> Image: ...
def write_image(image: Image, path: str) -> None: ...
def read_transform(path: str) -> dict[str, Any]: ...
def write_transform(
    path: str,
    dimensionality: int,
    transforms: list[dict[str, Any]],
    description: str = "",
) -> None: ...
