"""Type stubs for the ``_ritk.registration`` submodule."""

from __future__ import annotations

from ritk._ritk.image import Image

def demons_register(
    fixed: Image,
    moving: Image,
    max_iterations: int = 50,
    sigma_diffusion: float = 1.0,
) -> tuple[Image, Image]: ...
def diffeomorphic_demons_register(
    fixed: Image,
    moving: Image,
    max_iterations: int = 50,
    sigma_diffusion: float = 1.5,
    n_squarings: int = 6,
) -> tuple[Image, Image]: ...
def symmetric_demons_register(
    fixed: Image,
    moving: Image,
    max_iterations: int = 50,
    sigma_diffusion: float = 1.5,
) -> tuple[Image, Image]: ...
def syn_register(
    fixed: Image,
    moving: Image,
    max_iterations: int = 100,
    sigma_smooth: float = 3.0,
    cc_radius: int = 2,
) -> tuple[Image, Image]: ...
