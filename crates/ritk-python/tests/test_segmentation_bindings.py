import numpy as np
import pytest
import ritk


def _image(values: np.ndarray) -> ritk.Image:
    return ritk.Image(np.asarray(values, dtype=np.float32))


def test_connected_components_returns_two_components_for_separated_voxels() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.float32)
    mask[1, 1, 1] = 1.0
    mask[3, 3, 3] = 1.0

    labeled, count = ritk.segmentation.connected_components(
        _image(mask), connectivity=6
    )
    labeled_np = labeled.to_numpy()

    assert count == 2
    assert labeled_np.shape == mask.shape
    assert labeled_np[1, 1, 1] > 0.0
    assert labeled_np[3, 3, 3] > 0.0
    assert labeled_np[1, 1, 1] != labeled_np[3, 3, 3]
    assert labeled_np.sum() > 0.0


def test_connected_components_26_connectivity_merges_diagonal_voxels() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.float32)
    mask[0, 0, 0] = 1.0
    mask[1, 1, 1] = 1.0

    labeled_6, count_6 = ritk.segmentation.connected_components(
        _image(mask), connectivity=6
    )
    labeled_26, count_26 = ritk.segmentation.connected_components(
        _image(mask), connectivity=26
    )

    labeled_6_np = labeled_6.to_numpy()
    labeled_26_np = labeled_26.to_numpy()

    assert count_6 == 2
    assert count_26 == 1
    assert labeled_6_np[0, 0, 0] != labeled_6_np[1, 1, 1]
    assert labeled_26_np[0, 0, 0] == labeled_26_np[1, 1, 1]


def test_connected_components_rejects_invalid_connectivity() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="connectivity must be 6 or 26"):
        ritk.segmentation.connected_components(_image(mask), connectivity=18)


def test_chan_vese_segment_preserves_shape_and_finite_values() -> None:
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[2:7, 2:7, 2:7] = 1.0

    result = ritk.segmentation.chan_vese_segment(
        _image(image),
        mu=0.25,
        nu=0.0,
        lambda1=1.0,
        lambda2=1.0,
        max_iterations=3,
        dt=0.1,
        tolerance=1e-6,
    )
    result_np = result.to_numpy()

    assert result_np.shape == image.shape
    assert np.isfinite(result_np).all()
    assert np.var(result_np) > 0.0


def test_geodesic_active_contour_segment_preserves_shape_and_finite_values() -> None:
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[3:6, 3:6, 3:6] = 2.0

    initial_phi = np.ones((9, 9, 9), dtype=np.float32)
    initial_phi[2:7, 2:7, 2:7] = -1.0

    result = ritk.segmentation.geodesic_active_contour_segment(
        _image(image),
        _image(initial_phi),
        propagation_weight=1.0,
        curvature_weight=0.5,
        advection_weight=1.0,
        edge_k=1.0,
        sigma=1.0,
        dt=0.05,
        max_iterations=2,
    )
    result_np = result.to_numpy()

    assert result_np.shape == image.shape
    assert np.isfinite(result_np).all()
    assert np.var(result_np) > 0.0


def test_shape_detection_segment_preserves_shape_and_finite_values() -> None:
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[2:5, 2:5, 2:5] = 1.5

    initial_phi = np.ones((7, 7, 7), dtype=np.float32)
    initial_phi[1:6, 1:6, 1:6] = -1.0

    result = ritk.segmentation.shape_detection_segment(
        _image(image),
        _image(initial_phi),
        curvature_weight=0.2,
        propagation_weight=1.0,
        advection_weight=1.0,
        edge_k=1.0,
        sigma=1.0,
        dt=0.05,
        max_iterations=2,
        tolerance=1e-6,
    )
    result_np = result.to_numpy()

    assert result_np.shape == image.shape
    assert np.isfinite(result_np).all()
    assert np.var(result_np) > 0.0


def test_threshold_level_set_segment_preserves_shape_and_finite_values() -> None:
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[2:5, 2:5, 2:5] = 3.0

    initial_phi = np.ones((7, 7, 7), dtype=np.float32)
    initial_phi[1:6, 1:6, 1:6] = -1.0

    result = ritk.segmentation.threshold_level_set_segment(
        _image(image),
        _image(initial_phi),
        lower_threshold=1.0,
        upper_threshold=4.0,
        propagation_weight=1.0,
        curvature_weight=0.2,
        dt=0.05,
        max_iterations=2,
        tolerance=1e-6,
    )
    result_np = result.to_numpy()

    assert result_np.shape == image.shape
    assert np.isfinite(result_np).all()
    assert np.var(result_np) > 0.0


def test_laplacian_level_set_segment_preserves_shape_and_finite_values() -> None:
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, 3, 3] = 5.0

    initial_phi = np.ones((7, 7, 7), dtype=np.float32)
    initial_phi[2:5, 2:5, 2:5] = -1.0

    result = ritk.segmentation.laplacian_level_set_segment(
        _image(image),
        _image(initial_phi),
        propagation_weight=1.0,
        curvature_weight=0.2,
        sigma=1.0,
        dt=0.05,
        max_iterations=2,
        tolerance=1e-6,
    )
    result_np = result.to_numpy()

    assert result_np.shape == image.shape
    assert np.isfinite(result_np).all()
    assert np.var(result_np) > 0.0
