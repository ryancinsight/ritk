"""Type stubs for the ``_ritk.segmentation`` submodule."""

from __future__ import annotations

from ritk._ritk.image import Image

def otsu_threshold(image: Image) -> tuple[float, Image]: ...
def connected_components(mask: Image, connectivity: int = 6) -> tuple[Image, int]: ...
