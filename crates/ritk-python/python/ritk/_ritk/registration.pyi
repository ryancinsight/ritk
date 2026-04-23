"""Type stubs for the ``_ritk.registration`` submodule (PyO3/maturin).

All signatures derived from ``ritk-python/src/registration.rs``.
"""

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
def inverse_consistent_demons_register(
    fixed: Image,
    moving: Image,
    max_iterations: int = 50,
    sigma_diffusion: float = 1.5,
    inverse_consistency_weight: float = 0.5,
    n_squarings: int = 6,
) -> tuple[Image, Image]: ...
def multires_demons_register(
    fixed: Image,
    moving: Image,
    max_iterations: int = 50,
    sigma_diffusion: float = 1.0,
    levels: int = 3,
    use_diffeomorphic: bool = False,
    n_squarings: int = 6,
) -> tuple[Image, Image]: ...
def syn_register(
    fixed: Image,
    moving: Image,
    max_iterations: int = 100,
    sigma_smooth: float = 3.0,
    cc_radius: int = 2,
) -> tuple[Image, Image]: ...
def bspline_ffd_register(
    fixed: Image,
    moving: Image,
    initial_control_spacing: int = 8,
    num_levels: int = 3,
    max_iterations: int = 100,
    learning_rate: float = 0.01,
    regularization_weight: float = 0.001,
) -> Image: ...
def multires_syn_register(
    fixed: Image,
    moving: Image,
    num_levels: int = 3,
    iterations: list[int] | None = None,
    sigma_smooth: float = 3.0,
    cc_radius: int = 2,
    inverse_consistency: bool = True,
) -> tuple[Image, Image]: ...
def bspline_syn_register(
    fixed: Image,
    moving: Image,
    max_iterations: int = 100,
    control_spacing_z: int = 8,
    control_spacing_y: int = 8,
    control_spacing_x: int = 8,
    sigma_smooth: float = 1.0,
    cc_radius: int = 2,
    regularization_weight: float = 0.001,
) -> tuple[Image, Image]: ...
def lddmm_register(
    fixed: Image,
    moving: Image,
    max_iterations: int = 50,
    num_time_steps: int = 10,
    kernel_sigma: float = 2.0,
    learning_rate: float = 0.1,
    regularization_weight: float = 1.0,
) -> tuple[Image, Image]: ...
def build_atlas(
    subjects: list[Image],
    max_iterations: int = 5,
    convergence_threshold: float = 0.01,
    syn_iterations: list[int] | None = None,
    sigma_smooth: float = 3.0,
    cc_radius: int = 2,
) -> tuple[Image, list[float]]: ...
def majority_vote_fusion(
    atlas_labels: list[Image],
) -> tuple[Image, Image]: ...
def joint_label_fusion_py(
    target: Image,
    atlas_images: list[Image],
    atlas_labels: list[Image],
    patch_radius: int = 2,
    beta: float = 0.1,
) -> tuple[Image, Image]: ...
