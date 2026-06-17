"""ritk parity against SimpleITK's *own cmake test cases*.

SimpleITK generates its BasicFilters regression suite from per-filter YAML
descriptions under ``Code/BasicFilters/yaml/<Filter>.yaml``; each file carries a
``tests:`` array whose entries pin an input image, a concrete ``settings``
parameter set, and an ``md5hash`` of the ITK-computed output (the cmake test
downloads the baseline and compares the hash).

This module mirrors those upstream test cases for the filters ritk implements:
it drives ritk **and** live SimpleITK with the *exact same* parameter values the
upstream YAML pins, and asserts agreement. SimpleITK computing the output here is
equivalent to the upstream baseline (same ITK code path that produced the md5),
so this is a faithful re-implementation of the cmake cases that does not require
fetching the content-addressed ``.nrrd``/``.png`` baselines.

Each case is annotated with the upstream ``<Filter>.yaml`` test ``tag`` it
mirrors. Input images that ritk cannot represent (RGB / multi-component) are
exercised on the scalar ``cthead1`` the suite also ships, with the upstream
parameter values preserved.

Skips cleanly when SimpleITK or the object store is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

from test_sitk_filter_corpus_parity import fetch, IMAGE  # noqa: E402


@pytest.fixture(scope="module")
def cthead():
    """cthead1 as (ritk, sitk-float) at native spacing, plus a spacing-1.0 pair.

    Some upstream cases (e.g. DiscreteGaussian ``bigG``) use unit-spacing PNG
    inputs; large physical variances on the native (0.35 mm) spacing overflow the
    ITK Bessel kernel identically in ITK and ritk, so those cases are run on the
    unit-spacing view to match the upstream input geometry.
    """
    path = fetch(IMAGE)
    si = sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)
    ri = ritk.io.read_image(path)
    arr = sitk.GetArrayFromImage(si).astype(np.float32)
    si1 = sitk.GetImageFromArray(arr)  # spacing defaults to 1.0
    ri1 = ritk.Image(np.ascontiguousarray(arr[None]))
    return ri, si, ri1, si1


def _rel(r, s, m=8):
    r = np.asarray((r[0] if isinstance(r, tuple) else r).to_numpy(), np.float64)
    s = sitk.GetArrayFromImage(s).astype(np.float64)
    if r.ndim == 2:
        r = r[None]
    if s.ndim == 2:
        s = s[None]
    return np.abs(r[:, m:-m, m:-m] - s[:, m:-m, m:-m]).max() / max(np.abs(s).max(), 1e-9)


# ── BinaryThresholdImageFilter.yaml :: tag "NarrowThreshold" ───────────────────
# settings: LowerThreshold=10, UpperThreshold=100, InsideValue=255, OutsideValue=0
def test_binary_threshold_narrow(cthead):
    ri, si, *_ = cthead
    r = ritk.filter.binary_threshold(ri, 10.0, 100.0, 255.0, 0.0)
    s = sitk.BinaryThreshold(si, 10.0, 100.0, 255, 0)
    assert _rel(r, s, m=2) == 0.0


# ── SmoothingRecursiveGaussianImageFilter.yaml :: tag "rgb_image" (Sigma=5) ────
# upstream input is RGB; the Sigma=5 setting is exercised on scalar cthead.
def test_smoothing_recursive_gaussian_sigma5(cthead):
    ri, si, *_ = cthead
    r = ritk.filter.recursive_gaussian(ri, sigma=5.0, order=0)
    s = sitk.SmoothingRecursiveGaussian(si, 5.0)
    assert _rel(r, s) < 1e-6


# ── DiscreteGaussianImageFilter.yaml :: tag "bigG" ─────────────────────────────
# settings: Variance=[100,100,100], MaximumKernelWidth=64 (upstream input is a
# unit-spacing PNG, so run on the spacing-1.0 view).
def test_discrete_gaussian_bigG(cthead):
    _, _, ri1, si1 = cthead
    r = ritk.filter.discrete_gaussian(ri1, 100.0)
    f = sitk.DiscreteGaussianImageFilter()
    f.SetVariance(100.0)
    f.SetMaximumKernelWidth(64)
    s = f.Execute(si1)
    assert _rel(r, s) < 1e-6


# ── MedianImageFilter.yaml :: tag "defaults" (Radius=1) ────────────────────────
# upstream "by23" uses an anisotropic Radius=[2,3] on an RGB input; ritk's median
# takes a scalar radius, so the default (radius-1 box) case is mirrored.
def test_median_default(cthead):
    ri, si, *_ = cthead
    r = ritk.filter.median_filter(ri, 1)
    s = sitk.Median(si, [1, 1])
    assert _rel(r, s, m=2) == 0.0


# ── OtsuThresholdImageFilter.yaml :: tag "default_on_float" (Ramp-Zero-One) ────
# upstream baseline threshold ≈ 0.5 on a 0→1 ramp. ritk and ITK agree on the
# (synthesised) ramp; the exact published 0.50002 depends on the upstream ramp's
# size/histogram, so the assertion is ritk≈ITK on the same synthesised input.
def test_otsu_ramp_zero_one():
    ramp = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (64, 1))
    ri = ritk.Image(np.ascontiguousarray(ramp[None]))
    rt = ritk.segmentation.otsu_threshold(ri)[0]
    f = sitk.OtsuThresholdImageFilter()
    f.SetNumberOfHistogramBins(128)
    f.Execute(sitk.GetImageFromArray(ramp))
    st = f.GetThreshold()
    assert abs(rt - st) < 1e-3, f"otsu ramp ritk={rt} sitk={st}"
    assert 0.45 < rt < 0.55, f"otsu ramp threshold {rt} not near 0.5"
