"""Systematic ritk-vs-SimpleITK filter parity corpus on the canonical SimpleITK
test image (``cthead1``).

This is a single data-driven table mirroring the SimpleITK cmake filter-test
pattern: each row applies the *same* operation with ritk and with SimpleITK on
identical real-image bytes (the ``cthead1`` CT head SimpleITK ships for its own
regression suite) and asserts interior agreement to a tolerance derived from the
operation. It consolidates the per-op differential checks established across the
parity campaign into a permanent, exhaustive regression battery.

Each row pins the exact SimpleITK calling convention (these are easy to get
wrong — see the module-level notes in ``test_sitk_filter_corpus_parity``):

* float-exact (rel ``< 1e-6``): the recursive/discrete Gaussian family, the
  intensity transforms, the discrete Laplacian / gradient magnitude.
* bit-exact (rel ``== 0``): image arithmetic, median, grayscale box morphology.
* small derived tolerances for the analytically-different but convergent ops
  (anisotropic diffusion, which depends on per-iteration f32 accumulation).

Skips cleanly when SimpleITK or the object store is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

from _sitk_data import fetch  # noqa: E402

IMAGE = "cthead1-Float.mha"  # 256×256 float CT head, spacing 0.352778 mm (isotropic)


@pytest.fixture(scope="module")
def images():
    path = fetch(IMAGE)
    ri = ritk.io.read_image(path)
    si = sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)
    return ri, si


def _np3(a) -> np.ndarray:
    """SimpleITK array → 3-D float64 (promote a 2-D slice to z=1)."""
    arr = sitk.GetArrayFromImage(a).astype(np.float64) if not isinstance(a, np.ndarray) else a.astype(np.float64)
    return arr[None] if arr.ndim == 2 else arr


def _r3(a) -> np.ndarray:
    arr = np.asarray(a.to_numpy(), np.float64)
    return arr[None] if arr.ndim == 2 else arr


def _rel_interior(ra: np.ndarray, sa: np.ndarray, margin: int) -> float:
    ra, sa = _r3(ra) if not isinstance(ra, np.ndarray) else ra, _np3(sa)
    assert ra.shape == sa.shape, f"shape {ra.shape} != {sa.shape}"
    m = margin
    sl = (slice(None), slice(m, -m), slice(m, -m)) if ra.shape[0] == 1 else (slice(m, -m),) * 3
    return float(np.abs(ra[sl] - sa[sl]).max() / max(np.abs(sa[sl]).max(), 1e-9))


# Each case: (id, ritk(ri) -> PyImage, sitk(si) -> sitk.Image, rel_tol, margin)
# `ri`/`si` are the cthead1 ritk/sitk images from the `images` fixture.
CASES = [
    # ── Image arithmetic — bit-exact ───────────────────────────────────────────
    ("add_images", lambda ri, si: ritk.filter.add_images(ri, ri), lambda si: sitk.Add(si, si), 0.0, 2),
    ("subtract_images", lambda ri, si: ritk.filter.subtract_images(ri, ri), lambda si: sitk.Subtract(si, si), 0.0, 2),
    ("multiply_images", lambda ri, si: ritk.filter.multiply_images(ri, ri), lambda si: sitk.Multiply(si, si), 0.0, 2),
    ("maximum_images", lambda ri, si: ritk.filter.maximum_images(ri, ri), lambda si: sitk.Maximum(si, si), 0.0, 2),
    ("minimum_images", lambda ri, si: ritk.filter.minimum_images(ri, ri), lambda si: sitk.Minimum(si, si), 0.0, 2),
    # ── Intensity transforms — float-exact ─────────────────────────────────────
    ("sigmoid", lambda ri, si: ritk.filter.sigmoid_filter(ri, alpha=50.0, beta=100.0),
     lambda si: sitk.Sigmoid(si, alpha=50.0, beta=100.0, outputMaximum=1.0, outputMinimum=0.0), 1e-6, 4),
    ("rescale_intensity", lambda ri, si: ritk.filter.rescale_intensity(ri, 0.0, 255.0),
     lambda si: sitk.RescaleIntensity(si, 0.0, 255.0), 1e-6, 2),
    ("intensity_windowing", lambda ri, si: ritk.filter.intensity_windowing(ri, 50.0, 150.0, 0.0, 255.0),
     lambda si: sitk.IntensityWindowing(si, 50.0, 150.0, 0.0, 255.0), 1e-6, 2),
    # ── Discrete edge operators — float-exact ──────────────────────────────────
    ("laplacian", lambda ri, si: ritk.filter.laplacian(ri), lambda si: sitk.Laplacian(si), 1e-6, 4),
    ("gradient_magnitude", lambda ri, si: ritk.filter.gradient_magnitude(ri), lambda si: sitk.GradientMagnitude(si), 1e-6, 4),
    # ── Gaussian family — float-exact ──────────────────────────────────────────
    ("discrete_gaussian", lambda ri, si: ritk.filter.discrete_gaussian(ri, 4.0),
     lambda si: sitk.DiscreteGaussian(si, 4.0), 1e-6, 4),
    ("smoothing_recursive_gaussian", lambda ri, si: ritk.filter.recursive_gaussian(ri, sigma=2.0, order=0),
     lambda si: sitk.SmoothingRecursiveGaussian(si, sigma=2.0), 1e-6, 4),
    ("gradient_magnitude_recursive_gaussian", lambda ri, si: ritk.filter.recursive_gaussian(ri, sigma=2.0, order=1),
     lambda si: sitk.GradientMagnitudeRecursiveGaussian(si, sigma=2.0), 1e-6, 4),
    ("laplacian_recursive_gaussian", lambda ri, si: ritk.filter.laplacian_of_gaussian(ri, sigma=2.0),
     lambda si: sitk.LaplacianRecursiveGaussian(si, sigma=2.0), 1e-6, 4),
    ("recursive_gaussian_directional_o1_x", lambda ri, si: ritk.filter.recursive_gaussian_directional(ri, 2.0, 1, 2),
     lambda si: sitk.RecursiveGaussian(si, 2.0, False, 1, 0), 1e-6, 5),
    ("recursive_gaussian_directional_o2_y", lambda ri, si: ritk.filter.recursive_gaussian_directional(ri, 2.0, 2, 1),
     lambda si: sitk.RecursiveGaussian(si, 2.0, False, 2, 1), 1e-6, 5),
    ("unsharp_mask", lambda ri, si: ritk.filter.unsharp_mask(ri, 1.0, 0.5, 0.0),
     lambda si: sitk.UnsharpMask(si, [1.0, 1.0], 0.5, 0.0), 1e-6, 4),
    # ── Anisotropic diffusion — convergent f32 accumulation ────────────────────
    ("gradient_anisotropic_diffusion", lambda ri, si: ritk.filter.anisotropic_diffusion(ri, 5, 3.0, 0.0625),
     lambda si: sitk.GradientAnisotropicDiffusion(si, timeStep=0.0625, conductanceParameter=3.0, numberOfIterations=5), 1e-5, 4),
    ("curvature_anisotropic_diffusion", lambda ri, si: ritk.filter.curvature_anisotropic_diffusion(ri, 5, 0.0625, 3.0),
     lambda si: sitk.CurvatureAnisotropicDiffusion(si, timeStep=0.0625, conductanceParameter=3.0, numberOfIterations=5), 1e-4, 4),
    # ── Grayscale box morphology — bit-exact ───────────────────────────────────
    ("grayscale_dilation", lambda ri, si: ritk.filter.grayscale_dilation(ri, 1), lambda si: sitk.GrayscaleDilate(si, [1, 1]), 0.0, 2),
    ("grayscale_erosion", lambda ri, si: ritk.filter.grayscale_erosion(ri, 1), lambda si: sitk.GrayscaleErode(si, [1, 1]), 0.0, 2),
    ("median", lambda ri, si: ritk.filter.median_filter(ri, 1), lambda si: sitk.Median(si, [1, 1]), 0.0, 2),
]


@pytest.mark.parametrize("name,rfn,sfn,tol,margin", CASES, ids=[c[0] for c in CASES])
def test_filter_matches_sitk(images, name, rfn, sfn, tol, margin):
    ri, si = images
    rel = _rel_interior(rfn(ri, si), sfn(si), margin)
    if tol == 0.0:
        assert rel == 0.0, f"{name}: expected bit-exact, got rel {rel:.2e}"
    else:
        assert rel < tol, f"{name}: rel {rel:.2e} >= {tol:.0e}"
