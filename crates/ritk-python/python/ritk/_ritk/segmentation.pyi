"""Type stubs for the ``_ritk.segmentation`` submodule (PyO3/maturin).

All signatures are derived from the authoritative Rust source at
``ritk-python/src/segmentation.rs``.
"""

from __future__ import annotations

from ritk._ritk.image import Image

# ── Thresholding ────────────────────────────────────────────────────────────

def otsu_threshold(image: Image) -> tuple[float, Image]: ...
def li_threshold(image: Image) -> tuple[float, Image]: ...
def yen_threshold(image: Image) -> tuple[float, Image]: ...
def kapur_threshold(image: Image) -> tuple[float, Image]: ...
def triangle_threshold(image: Image) -> tuple[float, Image]: ...
def multi_otsu_threshold(
    image: Image, num_classes: int = 3
) -> tuple[list[float], Image]: ...
def binary_threshold_segment(
    image: Image,
    lower: float | None = None,
    upper: float | None = None,
    inside_value: float = 1.0,
    outside_value: float = 0.0,
) -> Image: ...

# ── Connected-component labeling ────────────────────────────────────────────

def connected_components(mask: Image, connectivity: int = 6) -> tuple[Image, int]: ...

# ── Region growing ──────────────────────────────────────────────────────────

def connected_threshold_segment(
    image: Image,
    seed: tuple[int, int, int],
    lower: float,
    upper: float,
) -> Image: ...

# ── Clustering ──────────────────────────────────────────────────────────────

def kmeans_segment(image: Image, k: int = 3) -> Image: ...

# ── Watershed ───────────────────────────────────────────────────────────────

def watershed_segment(image: Image) -> Image: ...
def marker_watershed_segment(gradient: Image, markers: Image) -> Image: ...

# ── Binary morphology ──────────────────────────────────────────────────────

def binary_erosion(image: Image, radius: int = 1) -> Image: ...
def binary_dilation(image: Image, radius: int = 1) -> Image: ...
def binary_opening(image: Image, radius: int = 1) -> Image: ...
def binary_closing(image: Image, radius: int = 1) -> Image: ...
def binary_fill_holes(image: Image) -> Image: ...
def morphological_gradient(image: Image, radius: int = 1) -> Image: ...

# ── Level-set segmentation ──────────────────────────────────────────────────

def chan_vese_segment(
    image: Image,
    mu: float = 0.25,
    nu: float = 0.0,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    max_iterations: int = 200,
    dt: float = 0.1,
    tolerance: float = 1e-3,
) -> Image: ...
def geodesic_active_contour_segment(
    image: Image,
    initial_phi: Image,
    propagation_weight: float = 1.0,
    curvature_weight: float = 1.0,
    advection_weight: float = 1.0,
    edge_k: float = 1.0,
    sigma: float = 1.0,
    dt: float = 0.05,
    max_iterations: int = 200,
) -> Image: ...
def shape_detection_segment(
    image: Image,
    initial_phi: Image,
    curvature_weight: float = 0.2,
    propagation_weight: float = 1.0,
    advection_weight: float = 1.0,
    edge_k: float = 1.0,
    sigma: float = 1.0,
    dt: float = 0.05,
    max_iterations: int = 200,
    tolerance: float = 1e-3,
) -> Image: ...
def threshold_level_set_segment(
    image: Image,
    initial_phi: Image,
    lower_threshold: float,
    upper_threshold: float,
    propagation_weight: float = 1.0,
    curvature_weight: float = 0.2,
    dt: float = 0.05,
    max_iterations: int = 200,
    tolerance: float = 1e-3,
) -> Image: ...
def laplacian_level_set_segment(
    image: Image,
    initial_phi: Image,
    propagation_weight: float = 1.0,
    curvature_weight: float = 0.2,
    sigma: float = 1.0,
    dt: float = 0.05,
    max_iterations: int = 200,
    tolerance: float = 1e-3,
) -> Image: ...

# -- Advanced region growing / topology -----------------------------------

def confidence_connected_segment(
    image: Image,
    seed: list[int],
    initial_lower: float,
    initial_upper: float,
    multiplier: float = 2.5,
    max_iterations: int = 15,
) -> Image: ...
def neighborhood_connected_segment(
    image: Image,
    seed: list[int],
    lower: float,
    upper: float,
    radius: int = 1,
) -> Image: ...
def skeletonization(image: Image) -> Image: ...
def label_shape_statistics(
    mask: Image,
    connectivity: int = 6,
) -> list[dict[str, object]]:
    """Per-label shape statistics for each connected component in a binary mask.

    Args:
        mask:         Binary mask image (foreground > 0.5).
        connectivity: Adjacency model (6 or 26; default 6).

    Returns:
        list of dicts, one per component, sorted by label ascending. Each dict has:
        label (int), voxel_count (int),
        centroid (list[float]: [z, y, x] in index coordinates),
        bounding_box_min (list[int]: [z, y, x]),
        bounding_box_max (list[int]: [z, y, x]).

    Raises:
        ValueError: if connectivity is not 6 or 26.
    """
    ...
