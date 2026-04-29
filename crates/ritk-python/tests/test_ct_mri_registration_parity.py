"""CT/MRI cross-modal registration parity tests.

Tests in this module use the MRI-DIR porcine phantom DICOM datasets:
- CT:  test_data/3_head_ct_mridir/DICOM/ (409 slices, 512×512, 0.625 mm slice thickness)
- MRI: test_data/2_head_mri_t2/DICOM/   (94 slices, T2-weighted, same phantom)

Source: Ger et al., Medical Physics 2018, DOI: 10.1002/mp.13090 (CC BY 4.0)

All tests are skipped when the DICOM directories are absent.  To enable them:
  1. Download from TCIA MRI-DIR collection.
  2. Extract to test_data/3_head_ct_mridir/DICOM/ and test_data/2_head_mri_t2/DICOM/.
  3. Run: pytest crates/ritk-python/tests/test_ct_mri_registration_parity.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

# ---------------------------------------------------------------------------
# DICOM data discovery
# ---------------------------------------------------------------------------


def _find_test_data() -> str | None:
    """Walk up from this file's directory for up to 3 levels seeking a test_data folder."""
    here = Path(__file__).resolve().parent
    for _ in range(4):  # current directory + 3 ancestor levels
        candidate = here / "test_data"
        if candidate.is_dir():
            return str(candidate)
        here = here.parent
    return None


def _ct_dicom_dir() -> str | None:
    """Return the CT DICOM directory path or None when absent."""
    td = _find_test_data()
    if td is None:
        return None
    p = os.path.join(td, "3_head_ct_mridir", "DICOM")
    return p if os.path.isdir(p) else None


def _mri_dicom_dir() -> str | None:
    """Return the MRI DICOM directory path or None when absent."""
    td = _find_test_data()
    if td is None:
        return None
    p = os.path.join(td, "2_head_mri_t2", "DICOM")
    return p if os.path.isdir(p) else None


# Module-level presence flag evaluated once at import time.
_DATA_PRESENT: bool = _ct_dicom_dir() is not None and _mri_dicom_dir() is not None


# ---------------------------------------------------------------------------
# DICOM I/O helpers
# ---------------------------------------------------------------------------


def _load_sitk_dicom(dicom_dir: str) -> "sitk.Image":
    """Load the first DICOM series from dicom_dir using SimpleITK's GDCM reader."""
    reader = sitk.ImageSeriesReader()
    series_uids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_uids:
        raise RuntimeError(f"No DICOM series found in: {dicom_dir}")
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dicom_dir, series_uids[0]))
    return reader.Execute()


def _ritk(
    arr: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> "ritk.Image":
    """Wrap a 3-D numpy array in a ritk.Image with the given voxel spacing."""
    return ritk.Image(
        np.ascontiguousarray(arr, dtype=np.float32), spacing=list(spacing)
    )


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------


def _central_crop(arr: np.ndarray, size: int = 32) -> np.ndarray:
    """Extract a size³ central crop from a 3-D numpy array (shape z,y,x).

    Precondition: each spatial dimension must be >= size.
    """
    nz, ny, nx = arr.shape
    if nz < size or ny < size or nx < size:
        raise ValueError(f"Array shape {arr.shape} is too small for a {size}³ crop.")
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    half = size // 2
    return arr[
        cz - half : cz + half, cy - half : cy + half, cx - half : cx + half
    ].astype(np.float32)


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Minmax normalise arr to [0, 1]; returns zeros when the range is degenerate."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# NCC helper
# ---------------------------------------------------------------------------


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation: cov(a,b) / (||a-ā|| * ||b-b̄||).

    Returns 0.0 when either signal has zero variance (degenerate case).
    """
    ma = float(a.mean())
    mb = float(b.mean())
    da = float(np.sqrt(((a - ma) ** 2).sum()))
    db = float(np.sqrt(((b - mb) ** 2).sum()))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(((a - ma) * (b - mb)).sum()) / (da * db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_skip = pytest.mark.skipif(not _DATA_PRESENT, reason="MRI-DIR DICOM data absent")


@_skip
def test_ct_statistics_agree_with_sitk():
    """CT volume statistics computed by RITK must agree with SimpleITK within 5%.

    Mathematical basis: both implementations compute min/max/mean over identical
    voxel data loaded from the same DICOM series.  A 5% relative tolerance absorbs
    any floating-point conversion differences between f32 (RITK) and f64 (ITK).

    Sanity bounds:
      - CT air ≈ -1000 HU → minimum must be below -500.
      - CT bone/metal → maximum must exceed 200 HU.
    """
    ct_dir = _ct_dicom_dir()
    assert ct_dir is not None  # guarded by _DATA_PRESENT

    # SimpleITK reference
    ct_img = _load_sitk_dicom(ct_dir)
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(ct_img)
    sitk_min = stats_filter.GetMinimum()
    sitk_max = stats_filter.GetMaximum()
    sitk_mean = stats_filter.GetMean()

    # Domain sanity: CT HU range
    assert sitk_min < -500, (
        f"CT minimum {sitk_min:.1f} HU is unexpectedly high; expected air < -500 HU"
    )
    assert sitk_max > 200, (
        f"CT maximum {sitk_max:.1f} HU is unexpectedly low; expected bone/implant > 200 HU"
    )

    # RITK path: read_image dispatches to DICOM when path is a directory
    ct_ritk = ritk.io.read_image(ct_dir)
    ritk_stats = ritk.statistics.compute_statistics(ct_ritk)

    tol = 0.05  # 5% relative tolerance
    assert abs(ritk_stats["min"] - sitk_min) / max(abs(sitk_min), 1.0) < tol, (
        f"min mismatch: ritk={ritk_stats['min']:.2f}, sitk={sitk_min:.2f}"
    )
    assert abs(ritk_stats["max"] - sitk_max) / max(abs(sitk_max), 1.0) < tol, (
        f"max mismatch: ritk={ritk_stats['max']:.2f}, sitk={sitk_max:.2f}"
    )
    assert abs(ritk_stats["mean"] - sitk_mean) / max(abs(sitk_mean), 1.0) < tol, (
        f"mean mismatch: ritk={ritk_stats['mean']:.4f}, sitk={sitk_mean:.4f}"
    )


@_skip
def test_mri_statistics_agree_with_sitk():
    """MRI volume statistics computed by RITK must agree with SimpleITK within 5%.

    Mathematical basis: identical to CT statistics test but for T2-weighted MRI.
    MRI intensities are unsigned (minimum >= 0) and non-trivial (maximum > 0,
    mean > 0) for a valid T2-weighted head volume.
    """
    mri_dir = _mri_dicom_dir()
    assert mri_dir is not None

    # SimpleITK reference
    mri_img = _load_sitk_dicom(mri_dir)
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(mri_img)
    sitk_min = stats_filter.GetMinimum()
    sitk_max = stats_filter.GetMaximum()
    sitk_mean = stats_filter.GetMean()

    # Domain sanity: MRI is unsigned, non-trivial
    assert sitk_min >= 0, (
        f"MRI minimum {sitk_min:.2f} is negative; MRI must be unsigned"
    )
    assert sitk_max > 0, (
        f"MRI maximum {sitk_max:.2f} is zero; volume must be non-trivial"
    )
    assert sitk_mean > 0, f"MRI mean {sitk_mean:.4f} is non-positive"

    # RITK path
    mri_ritk = ritk.io.read_image(mri_dir)
    ritk_stats = ritk.statistics.compute_statistics(mri_ritk)

    tol = 0.05
    assert abs(ritk_stats["min"] - sitk_min) / max(abs(sitk_min), 1.0) < tol, (
        f"min mismatch: ritk={ritk_stats['min']:.2f}, sitk={sitk_min:.2f}"
    )
    assert abs(ritk_stats["max"] - sitk_max) / max(abs(sitk_max), 1.0) < tol, (
        f"max mismatch: ritk={ritk_stats['max']:.2f}, sitk={sitk_max:.2f}"
    )
    assert abs(ritk_stats["mean"] - sitk_mean) / max(abs(sitk_mean), 1.0) < tol, (
        f"mean mismatch: ritk={ritk_stats['mean']:.4f}, sitk={sitk_mean:.4f}"
    )


@_skip
def test_ct_mri_ncc_is_low_before_registration():
    """NCC between a central 32³ CT crop and MRI crop must be below 0.5 before registration.

    Mathematical basis: CT encodes tissue X-ray attenuation (Hounsfield units),
    T2-weighted MRI encodes transverse magnetisation decay.  The two intensity
    distributions are fundamentally different; their NCC before registration and
    intensity normalisation must be low (|NCC| < 0.5).  This validates the
    cross-modal registration premise: direct NCC alignment is insufficient and
    motivates mutual-information-based or SyN-based registration.

    The test extracts a 32³ central crop from each modality after loading via
    SimpleITK (numpy array) to isolate the signal comparison from I/O paths.
    """
    ct_img = _load_sitk_dicom(_ct_dicom_dir())
    mri_img = _load_sitk_dicom(_mri_dicom_dir())

    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # shape (nz, ny, nx)
    mri_arr = sitk.GetArrayFromImage(mri_img).astype(np.float32)

    ct_crop = _central_crop(ct_arr, size=32)
    mri_crop = _central_crop(mri_arr, size=32)

    ncc = _ncc(ct_crop, mri_crop)
    assert abs(ncc) < 0.5, (
        f"Pre-registration CT/MRI NCC {ncc:.4f} is unexpectedly high (>= 0.5); "
        "the two modalities should have dissimilar intensity distributions"
    )


@_skip
def test_histogram_match_ct_to_mri_reduces_distribution_gap():
    """Histogram matching CT to MRI must reduce the median distribution gap.

    Mathematical basis: histogram matching implements piecewise-linear CDF-quantile
    mapping from the source (CT) distribution to the reference (MRI) distribution.
    After matching, the p50 of the matched CT crop must be strictly closer to the
    p50 of the normalised MRI crop than the p50 of the normalised CT crop was.

    Both modalities are minmax-normalised to [0, 1] before matching so that the
    distribution gap is comparable on a common scale.  The assertion is:
        |median(matched_ct) - median(mri_norm)| < |median(ct_norm) - median(mri_norm)|
    """
    ct_img = _load_sitk_dicom(_ct_dicom_dir())
    mri_img = _load_sitk_dicom(_mri_dicom_dir())

    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mri_arr = sitk.GetArrayFromImage(mri_img).astype(np.float32)

    ct_crop = _central_crop(ct_arr, size=32)
    mri_crop = _central_crop(mri_arr, size=32)

    ct_norm = _minmax(ct_crop)
    mri_norm = _minmax(mri_crop)

    gap_before = abs(float(np.median(ct_norm)) - float(np.median(mri_norm)))

    matched = ritk.statistics.histogram_match(_ritk(ct_norm), _ritk(mri_norm))
    matched_arr = matched.to_numpy()

    gap_after = abs(float(np.median(matched_arr)) - float(np.median(mri_norm)))

    assert gap_after < gap_before, (
        f"Histogram matching did not reduce the median distribution gap: "
        f"before={gap_before:.4f}, after={gap_after:.4f}; "
        f"ct_norm p50={np.median(ct_norm):.4f}, mri_norm p50={np.median(mri_norm):.4f}, "
        f"matched p50={np.median(matched_arr):.4f}"
    )
