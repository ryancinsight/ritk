"""Fetch canonical SimpleITK test images for accurate ritk-vs-SimpleITK comparison.

SimpleITK stores its test corpus via CMake ExternalData: the repository holds
``<name>.sha512`` content-link files and the actual bytes live in an object store
keyed by SHA-512.  This module downloads a curated subset on demand (verifying
the hash) into ``externals/sitk_data`` (git-ignored) and exposes helpers to load
an image either through ritk's own reader or — for formats/dimensionalities ritk
cannot yet read directly — through SimpleITK, promoting 2-D images to a degenerate
``[1, Y, X]`` ritk volume.

Tests that use this module skip cleanly when SimpleITK is unavailable or the
object store cannot be reached.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
import ritk  # noqa: E402

# Object-store URL templates (ExternalData), tried in order.
_STORES = (
    "https://simpleitk.s3.amazonaws.com/public/SHA512/{}",
    "https://data.kitware.com/api/v1/file/hashsum/SHA512/{}/download",
    "https://simpleitk.org/SimpleITKExternalData/SHA512/{}",
)

# Curated name -> SHA-512 map (from SimpleITK/Testing/Data/Input/*.sha512).
SHA512: dict[str, str] = {
    "RA-Short.nrrd": "a8d0a877d0994f199542ed01200e68089f144cea86a97fedd744b969ddbc9957265a17b510425ca074340966eb69768ae71d7a00fc60ecb495517b1583681950",
    "RA-Float.nrrd": "7d642e9ac12a1d4e7e73aaceed13dc38baa676535b589ee62546ac8beecbc1e0dd38b8c5625c6c842413812229f3f1f337d8f6a1dd4b6b2fde7f51b31a9cfdeb",
    "OAS1_0001_MR1_mpr-1_anon.nrrd": "875ba2928c00e4b37fe21de08077e9c015f75ae6134e5fd97d55011d3e04d13e32253f37d9474f00429da44d0aedd86cdb1accb0edb4ac56799035b0774f4c85",
    "OAS1_0002_MR1_mpr-1_anon.nrrd": "9fd6d5a2c4df51e6b36704f2c91c52fc6069859b25ff3f6a69c0c5b2f5d1727c716c8a781004bb3c0e717756b954f568013a3d1a7a97448c118c1176fe33587a",
    "cthead1-Float.mha": "6a99d4f3edaf5238ab585f9b8445720a8d71157157acb0a461ca568e10ece671fb8f2af3d348d6cbe5b43560536f149cc1a4102ce9a4b8b4936ae23e3d9101b8",
    "BrainProtonDensitySlice.png": "db3a15caea5f4d5fd8632116d0ab1dbf02a140b019594411537f77a3ffe2340972436b4f82684a6170b0624601182749b6c45f7a46f7dd32616674f57ec874cd",
    "Gaussian_1.5.nrrd": "6476cb9bd1580474dab9db623e242785ba4c6d4d4da49616209130734ddf0e08ca841536b7bf4d49ec6dc25df0d79492079bf1ecd767d9134b137bdb1871fc2c",
    "Ramp-Up-Short.nrrd": "7c4ba39e08f0cabd60208b37f464a56d872f5ac02ef2d581948f89a79aa49cf65b4e94314c977ac295ce164096190ff77ea57638f41633e35a2a28d58dcc30d3",
    # ── Golden ITK baselines (precomputed filter outputs from SimpleITK/Testing) ──
    "cthead1.png": "07fde89ac38926e4cd1746282071c8d3fb0b6222aed9d70f5fdcabb10a8caee3ad8c040d103924e55aa484d6f8ab5310cde52a6a0ecf83a9d87f0501f23c0cc1",
    "cthead1-grad-mag.nrrd": "4ea626bdb5618e1da0e91997800a5ef9d88af0c7106ee540c9197d8f2a4a3253ed1581c99d5921b06dbe173a93ba7e780503e0b88f781283fa2534e573bfd1bb",
    "2th_cthead1.png": "d6a59581d34f463cf23eefed40f0ccbd43092d59601d1473290bd212053acca6b00640984a648cf8ab7dbb0ba04057fe5c003695b32026651bd486edd350562a",
    "2th_cthead1_distance.nrrd": "8697e9938c6d583777e848e37cf44419916703f44546dbedbf72b34ef2c80eda42a44e72e73d46e02cb4fa7f71232e97729632dae00aeb379a95c024d7171304",
    "cthead1-mask.png": "5f130baea5187237996ed413c456d60da7bdb34da282829ff83a5949a29b044e33af73948beebac4999ed5028de211a8983be0d576f5f35da1d68b2c36628861",
}

_CACHE = Path(__file__).resolve().parents[3] / "externals" / "sitk_data"


def fetch(name: str) -> str:
    """Return the local path to test image `name`, downloading + verifying once.

    Skips the test if the object store is unreachable.
    """
    if name not in SHA512:
        raise KeyError(f"unknown SimpleITK test image: {name}")
    digest = SHA512[name]
    _CACHE.mkdir(parents=True, exist_ok=True)
    out = _CACHE / name
    if out.exists() and hashlib.sha512(out.read_bytes()).hexdigest() == digest:
        return str(out)
    last = ""
    for tmpl in _STORES:
        try:
            data = urllib.request.urlopen(tmpl.format(digest), timeout=60).read()
            if hashlib.sha512(data).hexdigest() == digest:
                out.write_bytes(data)
                return str(out)
            last = "hash mismatch"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
    pytest.skip(f"could not fetch SimpleITK test data {name}: {last}")
    raise AssertionError("unreachable")  # pragma: no cover


def load_sitk(name: str):
    """Read `name` as a SimpleITK image (cast to float32)."""
    return sitk.Cast(sitk.ReadImage(fetch(name)), sitk.sitkFloat32)


def load_ritk(name: str):
    """Read `name` as a ritk 3-D `Image` via SimpleITK (the universal loader),
    promoting 2-D images to a `[1, Y, X]` volume.

    Using SimpleITK to load means the comparison is driven by identical input
    bytes regardless of whether ritk's own reader supports the format yet.
    """
    img = load_sitk(name)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [Z,Y,X] or [Y,X]
    if arr.ndim == 2:
        arr = arr[None, :, :]
    spacing = list(img.GetSpacing())[::-1]  # sitk [x,y,z] -> ritk [z,y,x]
    if len(spacing) == 2:
        spacing = [1.0, *spacing]
    return ritk.Image(np.ascontiguousarray(arr), spacing=spacing)


def sitk_to_zyx(img) -> np.ndarray:
    """SimpleITK image -> float32 `[Z,Y,X]` numpy array (2-D promoted to z=1)."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr[None, :, :] if arr.ndim == 2 else arr
