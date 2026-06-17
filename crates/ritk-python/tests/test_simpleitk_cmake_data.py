"""ritk parity on SimpleITK's cmake test cases using the **real upstream input
data**.

This is the high-fidelity companion to ``test_simpleitk_cmake_cases.py``. Where
that module reuses the locally-available ``cthead1`` with upstream parameter
values, this one fetches the *exact* input images the upstream cmake tests use
(``RA-Float.nrrd``, ``WhiteDots.png``, ``Ramp-Zero-One-Float.nrrd``, …) from
SimpleITK's content-addressed ExternalData store, then drives ritk and live
SimpleITK with the parameters pinned in each filter's
``Code/BasicFilters/yaml/<Filter>.yaml`` ``tests:`` entry.

Inputs are resolved by SHA-512 from the committed manifest
``externals/sitk_input_manifest.json`` (harvested from the upstream
``Testing/Data/Input/<name>.sha512`` files) and downloaded once into a local
cache (``externals/sitk_data``, git-ignored). The download is hash-verified;
tests skip cleanly if the object store is unreachable, so the suite stays
hermetic offline.

Because SimpleITK computes the comparison output via the same ITK code path that
produced each upstream md5 baseline, asserting ritk == SimpleITK here is a
faithful re-implementation of the cmake regression — the same input bytes, the
same parameters, the same oracle.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# Manifest (input name -> sha512) is committed next to this test; the fetched
# data is cached under the git-ignored externals/ tree.
_MANIFEST = os.path.join(_HERE, "sitk_input_manifest.json")
_CACHE = os.path.join(_REPO_ROOT, "externals", "sitk_data")
_CDN = "https://data.kitware.com/api/v1/file/hashsum/sha512/{}/download"


def _manifest():
    if not os.path.exists(_MANIFEST):
        pytest.skip("SimpleITK input manifest not present")
    with open(_MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def fetch_input(name: str) -> str:
    """Return a local path to the upstream input `name`, downloading + verifying
    it from ExternalData on first use. Skips the test if unavailable."""
    sha = _manifest().get(name)
    if sha is None:
        pytest.skip(f"{name} not in manifest")
    os.makedirs(_CACHE, exist_ok=True)
    dst = os.path.join(_CACHE, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    try:
        urllib.request.urlretrieve(_CDN.format(sha), dst)
    except Exception as exc:  # offline / store down
        pytest.skip(f"could not fetch {name}: {exc}")
    if hashlib.sha512(open(dst, "rb").read()).hexdigest() != sha:
        os.remove(dst)
        pytest.fail(f"{name}: sha512 mismatch after download")
    return dst


def _pair(name):
    path = fetch_input(name)
    return ritk.io.read_image(path), sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)


def _rel(r, s, m=3):
    r = np.asarray((r[0] if isinstance(r, tuple) else r).to_numpy(), np.float64)
    s = sitk.GetArrayFromImage(s).astype(np.float64)
    if r.ndim == 2:
        r = r[None]
    if s.ndim == 2:
        s = s[None]
    return np.abs(r[:, m:-m, m:-m] - s[:, m:-m, m:-m]).max() / max(np.abs(s).max(), 1e-9)


# Each case: (id mirroring <Filter>.yaml::tag, input name, ritk fn, sitk fn, tol).
_CASES = [
    ("DiscreteGaussian/bigG", "WhiteDots.png",
     lambda ri: ritk.filter.discrete_gaussian(ri, 100.0),
     lambda si: _dg(si, 100.0, 64), 1e-6),
    ("GradientMagnitude/default", "RA-Float.nrrd",
     lambda ri: ritk.filter.gradient_magnitude(ri),
     lambda si: sitk.GradientMagnitude(si), 1e-6),
    ("Laplacian/default", "RA-Float.nrrd",
     lambda ri: ritk.filter.laplacian(ri),
     lambda si: sitk.Laplacian(si), 1e-6),
    ("SmoothingRecursiveGaussian/default", "RA-Float.nrrd",
     lambda ri: ritk.filter.recursive_gaussian(ri, sigma=1.0, order=0),
     lambda si: sitk.SmoothingRecursiveGaussian(si, 1.0), 1e-6),
    ("RescaleIntensity/3d", "RA-Float.nrrd",
     lambda ri: ritk.filter.rescale_intensity(ri, 0.0, 255.0),
     lambda si: sitk.RescaleIntensity(si, 0.0, 255.0), 1e-6),
    ("Normalize/defaults", "Ramp-Up-Short.nrrd",
     lambda ri: ritk.filter.normalize_image(ri),
     lambda si: sitk.Normalize(si), 1e-6),
    ("Sigmoid/defaults", "Ramp-Zero-One-Float.nrrd",
     lambda ri: ritk.filter.sigmoid_filter(ri, alpha=1.0, beta=0.0, min_output=0.0, max_output=1.0),
     lambda si: sitk.Sigmoid(si, 1.0, 0.0, 1.0, 0.0), 1e-6),
    # bit-exact:
    ("BinaryThreshold/NarrowThreshold", "RA-Short.nrrd",
     lambda ri: ritk.filter.binary_threshold(ri, 10.0, 100.0, 255.0, 0.0),
     lambda si: sitk.BinaryThreshold(si, 10.0, 100.0, 255, 0), 0.0),
    ("Threshold/Threshold1", "RA-Slice-Short.nrrd",
     lambda ri: ritk.filter.threshold_outside(ri, 25000.0, 65535.0),
     lambda si: sitk.Threshold(si, 25000.0, 65535.0, 0.0), 0.0),
]


def _dg(si, variance, mkw):
    f = sitk.DiscreteGaussianImageFilter()
    f.SetVariance(variance)
    f.SetMaximumKernelWidth(mkw)
    return f.Execute(si)


@pytest.mark.parametrize("tag,name,rfn,sfn,tol", _CASES, ids=[c[0] for c in _CASES])
def test_cmake_case_on_upstream_data(tag, name, rfn, sfn, tol):
    ri, si = _pair(name)
    rel = _rel(rfn(ri), sfn(si))
    if tol == 0.0:
        assert rel == 0.0, f"{tag}: expected bit-exact on {name}, got rel {rel:.2e}"
    else:
        assert rel < tol, f"{tag}: rel {rel:.2e} >= {tol:.0e} on {name}"
