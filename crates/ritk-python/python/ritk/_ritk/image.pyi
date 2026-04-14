"""Type stub for the ``_ritk.image`` submodule (PyO3/maturin)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class Image:
    """Medical image with physical-space metadata (Z×Y×X, f32)."""

    def __init__(
        self,
        array: NDArray[np.float32],
        spacing: tuple[float, float, float] | None = None,
        origin: tuple[float, float, float] | None = None,
    ) -> None: ...
    def to_numpy(self) -> NDArray[np.float32]: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    @property
    def spacing(self) -> tuple[float, float, float]: ...
    @property
    def origin(self) -> tuple[float, float, float]: ...
