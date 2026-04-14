"""Type stub for the ``_ritk.io`` submodule (image I/O)."""

from __future__ import annotations

from ritk._ritk.image import Image

def read_image(path: str) -> Image: ...
def write_image(image: Image, path: str) -> None: ...
