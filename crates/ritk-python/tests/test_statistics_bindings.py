import math

import numpy as np
import pytest
import ritk


def _image(values: np.ndarray) -> ritk.Image:
    return ritk.Image(np.asarray(values, dtype=np.float32))


def test_label_intensity_stats_single_label_two_voxels_known_values() -> None:
    # label 1 covers voxels with intensity 3.0 and 5.0
    # population mean = 4.0, population std = sqrt(1.0) = 1.0
    label = np.zeros((3, 3, 3), dtype=np.float32)
    label[1, 1, 1] = 1.0
    label[1, 1, 2] = 1.0
    intensity = np.zeros((3, 3, 3), dtype=np.float32)
    intensity[1, 1, 1] = 3.0
    intensity[1, 1, 2] = 5.0

    result = ritk.statistics.compute_label_intensity_statistics(
        _image(label), _image(intensity)
    )

    assert len(result) == 1
    s = result[0]
    assert s["label"] == 1
    assert s["count"] == 2
    assert s["min"] == pytest.approx(3.0, abs=1e-5)
    assert s["max"] == pytest.approx(5.0, abs=1e-5)
    assert s["mean"] == pytest.approx(4.0, abs=1e-5)
    assert s["std"] == pytest.approx(1.0, abs=1e-5)


def test_label_intensity_stats_background_excluded() -> None:
    # Only background (label 0) voxels -- result must be empty
    label = np.zeros((3, 3, 3), dtype=np.float32)
    intensity = np.ones((3, 3, 3), dtype=np.float32)

    result = ritk.statistics.compute_label_intensity_statistics(
        _image(label), _image(intensity)
    )

    assert len(result) == 0


def test_label_intensity_stats_multiple_labels_sorted() -> None:
    # Label 2 has intensity 10.0 (single voxel); label 1 has intensity 2.0 (single voxel)
    # Results must be sorted ascending: label 1 first, label 2 second
    label = np.zeros((3, 3, 3), dtype=np.float32)
    label[0, 0, 0] = 2.0
    label[2, 2, 2] = 1.0
    intensity = np.zeros((3, 3, 3), dtype=np.float32)
    intensity[0, 0, 0] = 10.0
    intensity[2, 2, 2] = 2.0

    result = ritk.statistics.compute_label_intensity_statistics(
        _image(label), _image(intensity)
    )

    assert len(result) == 2
    assert result[0]["label"] == 1
    assert result[0]["count"] == 1
    assert result[0]["min"] == pytest.approx(2.0, abs=1e-5)
    assert result[0]["max"] == pytest.approx(2.0, abs=1e-5)
    assert result[0]["mean"] == pytest.approx(2.0, abs=1e-5)
    assert result[0]["std"] == pytest.approx(0.0, abs=1e-5)
    assert result[1]["label"] == 2
    assert result[1]["count"] == 1
    assert result[1]["min"] == pytest.approx(10.0, abs=1e-5)
    assert result[1]["max"] == pytest.approx(10.0, abs=1e-5)
    assert result[1]["mean"] == pytest.approx(10.0, abs=1e-5)
    assert result[1]["std"] == pytest.approx(0.0, abs=1e-5)


def test_label_intensity_stats_four_voxels_known_mean_and_std() -> None:
    # Label 1 covers 4 voxels with values [1, 2, 3, 4]
    # population mean = 2.5, population variance = E[X^2] - E[X]^2
    # E[X^2] = (1+4+9+16)/4 = 7.5, E[X]^2 = 6.25 => var = 1.25 => std = sqrt(1.25)
    expected_std = math.sqrt(1.25)
    label = np.zeros((2, 2, 1), dtype=np.float32)
    label[:, :, 0] = 1.0
    intensity = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)

    result = ritk.statistics.compute_label_intensity_statistics(
        _image(label), _image(intensity)
    )

    assert len(result) == 1
    s = result[0]
    assert s["label"] == 1
    assert s["count"] == 4
    assert s["min"] == pytest.approx(1.0, abs=1e-5)
    assert s["max"] == pytest.approx(4.0, abs=1e-5)
    assert s["mean"] == pytest.approx(2.5, abs=1e-5)
    assert s["std"] == pytest.approx(expected_std, abs=1e-5)


def test_minmax_normalize_range_inverted_bounds_raises() -> None:
    # ritk.Image requires 3-D arrays; reshape scalar sequence to (1, 1, 3).
    image = _image(np.array([[[0.0, 1.0, 2.0]]], dtype=np.float32))

    with pytest.raises(ValueError, match="strictly less than"):
        ritk.statistics.minmax_normalize_range(image, 1.0, 0.0)


def test_zscore_normalize_masked_matches_foreground_shape() -> None:
    image = _image(np.arange(8, dtype=np.float32).reshape(2, 2, 2))
    mask = _image(
        np.array([[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    )

    result = ritk.statistics.zscore_normalize(image, mask=mask)
    result_np = result.to_numpy()

    assert result_np.shape == image.to_numpy().shape
    assert np.isfinite(result_np).all()

    foreground = result_np[mask.to_numpy() > 0.5]
    assert foreground.size == 4
    assert foreground.mean() == pytest.approx(0.0, abs=1e-5)


def test_zscore_normalize_mask_shape_mismatch_raises() -> None:
    image = _image(np.arange(8, dtype=np.float32).reshape(2, 2, 2))
    mask = _image(np.ones((2, 2, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="same shape as image"):
        ritk.statistics.zscore_normalize(image, mask=mask)


def test_minmax_normalize_range_and_zscore_bindings_are_available() -> None:
    # ritk.Image requires 3-D arrays; reshape to (1, 2, 2) so shape is valid.
    image = _image(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
    ranged = ritk.statistics.minmax_normalize_range(image, 0.0, 1.0)
    assert isinstance(ranged, ritk.Image)
    # Verify the range: min should be 0.0 and max should be 1.0 after normalization.
    ranged_np = ranged.to_numpy()
    assert ranged_np.min() == pytest.approx(0.0, abs=1e-6)
    assert ranged_np.max() == pytest.approx(1.0, abs=1e-6)

    zscore = ritk.statistics.zscore_normalize(image)
    assert isinstance(zscore, ritk.Image)
    # Z-score output must have zero mean and unit std.
    zscore_np = zscore.to_numpy()
    assert zscore_np.mean() == pytest.approx(0.0, abs=1e-5)
    assert zscore_np.std() == pytest.approx(1.0, abs=1e-4)
