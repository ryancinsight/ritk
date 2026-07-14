"""Type stubs for the ``_ritk.segmentation`` submodule (PyO3/maturin).

All signatures are derived from the authoritative Rust source at
``ritk-python/src/segmentation.rs``.
"""

from __future__ import annotations

from typing import Any

from ritk._ritk.image import Image

# ── Thresholding ────────────────────────────────────────────────────────────

def otsu_threshold(image: Image) -> tuple[float, Image]: ...
def li_threshold(image: Image) -> tuple[float, Image]: ...
def yen_threshold(image: Image) -> tuple[float, Image]: ...
def kapur_threshold(image: Image) -> tuple[float, Image]: ...
def triangle_threshold(image: Image) -> tuple[float, Image]: ...
def huang_threshold(image: Image) -> tuple[float, Image]: ...
def intermodes_threshold(image: Image) -> tuple[float, Image]: ...
def isodata_threshold(image: Image) -> tuple[float, Image]: ...
def kittler_illingworth_threshold(image: Image) -> tuple[float, Image]: ...
def moments_threshold(image: Image) -> tuple[float, Image]: ...
def renyi_entropy_threshold(image: Image) -> tuple[float, Image]: ...
def shanbhag_threshold(image: Image) -> tuple[float, Image]: ...
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
def scalar_connected_component(
    image: Image, distance_threshold: float = 0.0, connectivity: int = 6
) -> Image:
    """Label scalar connected components (neighbours join if |delta value| <= threshold). ITK Parity: ScalarConnectedComponentImageFilter."""
    ...

def relabel_components(label_image: Image, minimum_object_size: int = 0) -> Image:
    """Relabel components by descending size. ITK Parity: RelabelComponentImageFilter."""
    ...

def relabel_label_map(label_image: Image) -> Image:
    """Relabel non-zero labels to consecutive 1..K in ascending original-label order. ITK Parity: RelabelLabelMapFilter (sitk.RelabelLabelMap)."""
    ...

def merge_label_map(label_images: list[Image], method: int = 0) -> Image:
    """Merge label images (0=Keep,1=Aggregate,2=Pack,3=Strict). ITK Parity: MergeLabelMapFilter (sitk.MergeLabelMap)."""
    ...

def label_set_dilate(
    label_image: Image, radius: list[float] = ..., use_image_spacing: bool = True
) -> Image:
    """Label-preserving Euclidean dilation. ITK Parity: LabelSetDilateImageFilter (sitk.LabelSetDilate)."""
    ...

def label_set_erode(
    label_image: Image, radius: list[float] = ..., use_image_spacing: bool = True
) -> Image:
    """Label-preserving Euclidean erosion. ITK Parity: LabelSetErodeImageFilter (sitk.LabelSetErode)."""
    ...

def change_label(label_image: Image, change_map: dict[int, int]) -> Image:
    """Remap label values per {old: new}; others unchanged. ITK Parity: ChangeLabelImageFilter."""
    ...

def threshold_maximum_connected_components(
    image: Image, minimum_object_size: int = 0, upper_boundary: int | None = None
) -> Image:
    """Threshold an image at the lower value that maximizes the number of connected components. ITK Parity: ThresholdMaximumConnectedComponentsImageFilter."""
    ...

# ── Region growing ──────────────────────────────────────────────────────────

def connected_threshold_segment(
    image: Image,
    seed: tuple[int, int, int],
    lower: float,
    upper: float,
) -> Image: ...

# ── Clustering ──────────────────────────────────────────────────────────────

def kmeans_segment(
    image: Image,
    k: int = 3,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    seed: int | None = None,
) -> Image: ...

# ── Watershed ────────────────────────────────────────────────────

def watershed_segment(image: Image) -> Image: ...
def marker_watershed_segment(gradient: Image, markers: Image) -> Image: ...
def morphological_watershed(image: Image, level: float = 0.0) -> Image:
    """Marker-less morphological watershed. ITK Parity: MorphologicalWatershedImageFilter."""
    ...

def isolated_watershed_segment(
    image: Image,
    seed1: list[int],
    seed2: list[int],
    threshold: float = 0.0,
    isolated_value_tolerance: float = 0.001,
    upper_value_limit: float = 1.0,
) -> Image:
    """Isolated watershed: finds T* separating two seeds. Labels 1, 2, 3.
    ITK Parity: IsolatedWatershedImageFilter."""
    ...

def toboggan(image: Image) -> Image:
    """Toboggan steepest-descent basin labeling (labels >= 2). ITK Parity: TobogganImageFilter (sitk.Toboggan)."""
    ...

def slic(
    image: Image,
    super_grid_size: int,
    spatial_proximity_weight: float = 10.0,
    maximum_number_of_iterations: int = 5,
    enforce_connectivity: bool = True,
    initialization_perturbation: bool = True,
) -> Image:
    """SLIC super-pixel segmentation. ITK Parity: SLICImageFilter (sitk.SLIC); label-exact at the default config (z==1 treated as 2-D)."""
    ...

def vector_connected_component(
    channels: list[Image],
    distance_threshold: float = 1.0,
    fully_connected: bool = False,
) -> Image:
    """Connected components of a vector image (join if 1-|a.b|<=threshold). ITK Parity: VectorConnectedComponentImageFilter (sitk.VectorConnectedComponent); partition-exact."""
    ...

def vector_confidence_connected_segment(
    channels: list[Image],
    seeds: list[tuple[int, int, int]],
    multiplier: float = 2.5,
    number_of_iterations: int = 4,
    initial_neighborhood_radius: int = 1,
    replace_value: float = 1.0,
) -> Image:
    """Vector confidence-connected region growing (Mahalanobis). ITK Parity: VectorConfidenceConnectedImageFilter (sitk.VectorConfidenceConnected); region-exact for well-conditioned inputs."""
    ...

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
def slic_superpixel(
    image: Image,
    n_superpixels: int = 100,
    compactness: float = 10.0,
    max_iterations: int = 10,
    tolerance: float = 1e-3,
    min_component_size: int = 5,
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
    opts: ShapeDetectionOptions | None = None,
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
def isolated_connected_segment(
    image: Image,
    seed1: list[int],
    seed2: list[int],
    lower: float = 0.0,
    upper: float = 1.0,
    replace_value: float = 1.0,
    isolated_value_tolerance: float = 1.0,
    find_upper_threshold: bool = True,
) -> Image:
    """Isolated-connected segmentation. ITK Parity: IsolatedConnectedImageFilter."""
    ...

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

# -- GrowCut / STAPLE ---------------------------------------------------------

def growcut_segment(image: Image, seeds: Image, max_iter: int = 200) -> Image: ...
def staple_ensemble(
    raters: list[Image], max_iter: int = 100, tol: float = 1e-06
) -> dict[str, Any]: ...
def multi_label_staple(
    raters: list[Image],
    max_iter: int = 0,
    termination_threshold: float = 1e-5,
    label_for_undecided: float | None = None,
) -> Image:
    """Multi-label STAPLE EM consensus of K integer label maps.

    Args:
        raters:                list of Image, each a label map (f32). All must
                               have the same shape.
        max_iter:              Maximum EM iterations; 0 iterates to convergence.
        termination_threshold: Stop when max confusion-matrix change < threshold.
        label_for_undecided:   Label for tie voxels; None uses L (max_label+1,
                               ITK default).

    Returns:
        Hard consensus label image (f32), same shape/spacing/origin as raters[0].

    ITK Parity: MultiLabelSTAPLEImageFilter.
    """
    ...

def label_set_dilate(
    label_image: Image,
    radius: list[float] = [1.0, 1.0, 1.0],
    use_image_spacing: bool = True,
) -> Image:
    """Label-preserving Euclidean dilation, matching sitk.LabelSetDilate.
    Expands every label region by a per-axis Euclidean structuring element.
    ITK Parity: LabelSetDilateImageFilter."""
    ...

def label_set_erode(
    label_image: Image,
    radius: list[float] = [1.0, 1.0, 1.0],
    use_image_spacing: bool = True,
) -> Image:
    """Label-preserving Euclidean erosion, matching sitk.LabelSetErode.
    Shrinks every label region by a per-axis Euclidean structuring element.
    ITK Parity: LabelSetErodeImageFilter."""
    ...
