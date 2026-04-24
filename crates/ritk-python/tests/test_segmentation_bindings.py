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


# -- distance_transform -----------------------------------------------------------------------------------------

def test_distance_transform_all_foreground_returns_zeros() -> None:
    # All-foreground image: every voxel is at distance 0 from foreground
    mask = np.ones((5, 5, 5), dtype=np.float32)
    result = ritk.filter.distance_transform(_image(mask), foreground_threshold=0.5)
    result_np = result.to_numpy()
    assert result_np.shape == mask.shape
    assert np.allclose(result_np, 0.0, atol=1e-5)


def test_distance_transform_single_foreground_voxel_background_nonzero() -> None:
    # Single foreground voxel at [2,2,2] in a 5x5x5 image with unit spacing.
    # Voxel [2,2,2] must be 0.0; adjacent voxel [2,2,3] is 1 unit away.
    mask = np.zeros((5, 5, 5), dtype=np.float32)
    mask[2, 2, 2] = 1.0
    result = ritk.filter.distance_transform(_image(mask), foreground_threshold=0.5)
    result_np = result.to_numpy()
    assert result_np.shape == mask.shape
    assert result_np[2, 2, 2] == pytest.approx(0.0, abs=1e-5)
    assert result_np[2, 2, 3] == pytest.approx(1.0, abs=1e-4)
    mask_bool = mask < 0.5
    assert np.all(result_np[mask_bool] > 0.0)


def test_distance_transform_squared_equals_euclidean_squared() -> None:
    # squared=True output must equal element-wise square of squared=False output
    mask = np.zeros((5, 5, 5), dtype=np.float32)
    mask[2, 2, 2] = 1.0
    img = _image(mask)
    dist_np = ritk.filter.distance_transform(img, foreground_threshold=0.5, squared=False).to_numpy()
    sq_np = ritk.filter.distance_transform(img, foreground_threshold=0.5, squared=True).to_numpy()
    assert np.allclose(sq_np, dist_np ** 2, atol=1e-4)


# -- label_shape_statistics -------------------------------------------------------------------------------

def test_label_shape_stats_single_voxel_known_centroid() -> None:
    # Single foreground voxel at [2, 1, 3] -- centroid must equal [2.0, 1.0, 3.0]
    mask = np.zeros((5, 5, 5), dtype=np.float32)
    mask[2, 1, 3] = 1.0
    result = ritk.segmentation.label_shape_statistics(_image(mask), connectivity=6)
    assert len(result) == 1
    s = result[0]
    assert s["label"] == 1
    assert s["voxel_count"] == 1
    assert s["centroid"] == pytest.approx([2.0, 1.0, 3.0], abs=1e-5)
    assert s["bounding_box_min"] == [2, 1, 3]
    assert s["bounding_box_max"] == [2, 1, 3]


def test_label_shape_stats_two_components_sorted() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.float32)
    mask[0, 0, 0] = 1.0
    mask[4, 4, 4] = 1.0
    result = ritk.segmentation.label_shape_statistics(_image(mask), connectivity=6)
    assert len(result) == 2
    labels = [s["label"] for s in result]
    assert labels == sorted(labels)
    for s in result:
        assert s["voxel_count"] == 1


def test_label_shape_stats_rejects_invalid_connectivity() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="connectivity must be 6 or 26"):
        ritk.segmentation.label_shape_statistics(_image(mask), connectivity=18)
