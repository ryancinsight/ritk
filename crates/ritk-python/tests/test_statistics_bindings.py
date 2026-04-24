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
