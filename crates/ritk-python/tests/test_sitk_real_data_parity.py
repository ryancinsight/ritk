"""ritk vs SimpleITK parity on the canonical SimpleITK test corpus.

Unlike the synthetic-image parity suite, these tests run on the *real* medical
images SimpleITK ships for its own regression tests (brain MRI, CT head, ramps),
fetched on demand from the ExternalData object store (see ``_sitk_data``).  Each
test applies the same operation with ritk and with SimpleITK on identical input
bytes and asserts agreement to a tolerance derived from the signal range.

Skips cleanly when SimpleITK or the object store is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

import ritk  # noqa: E402

from _sitk_data import fetch, load_ritk, load_sitk, sitk_to_zyx  # noqa: E402


def _interior(a: np.ndarray, m: int = 3) -> np.ndarray:
    return a[m:-m, m:-m, m:-m] if a.shape[0] > 2 * m else a[:, m:-m, m:-m]


def _rel_max(ritk_arr: np.ndarray, sitk_arr: np.ndarray) -> float:
    """Max abs difference on the interior, relative to the input dynamic range."""
    rng = max(float(sitk_arr.max() - sitk_arr.min()), 1e-6)
    d = np.abs(_interior(ritk_arr).astype(np.float64) - _interior(sitk_arr).astype(np.float64))
    return float(d.max()) / rng


# Real 3-D volumes ritk can drive end-to-end.
VOLUMES_3D = ["RA-Float.nrrd", "RA-Short.nrrd", "OAS1_0001_MR1_mpr-1_anon.nrrd"]
IMAGES_ANY = VOLUMES_3D + ["cthead1-Float.mha", "BrainProtonDensitySlice.png"]


# ── Filter parity on real data ────────────────────────────────────────────────


@pytest.mark.parametrize("name", IMAGES_ANY)
def test_discrete_gaussian_matches_sitk_on_real_image(name):
    rimg = load_ritk(name)
    simg = load_sitk(name)
    ra = ritk.filter.discrete_gaussian(
        rimg, variance=4.0, maximum_error=0.01, spacing_mode=ritk.filter.PySpacingMode.Voxel
    ).to_numpy()
    sa = sitk_to_zyx(sitk.DiscreteGaussian(simg, variance=4.0, maximumError=0.01, useImageSpacing=False))
    rel = _rel_max(ra, sa)
    # ritk and ITK build the discrete Gaussian kernel with slightly different
    # radius/boundary handling, so at the high-contrast edges of real anatomy the
    # responses differ by up to a few percent of the dynamic range; smooth inputs
    # agree far more tightly.
    assert rel < 0.03, f"{name}: discrete_gaussian rel max diff {rel:.4f}"


@pytest.mark.parametrize("name", IMAGES_ANY)
def test_median_matches_sitk_on_real_image(name):
    rimg = load_ritk(name)
    simg = load_sitk(name)
    ra = ritk.filter.median_filter(rimg, radius=1).to_numpy()
    # For a z=1 promoted 2-D image use an in-plane radius (radius 0 in z).
    radius = [1, 1, 0] if rimg.shape[0] == 1 else [1, 1, 1]
    sa = sitk_to_zyx(sitk.Median(simg, radius))
    rel = _rel_max(ra, sa)
    # Median is exact where the neighbourhoods match; z=1 promotion of a 2-D
    # image makes ritk's 3-D 6-neighbour median differ from a 2-D median only at
    # nothing (z has one slice), so agreement is tight.
    assert rel < 1e-6 if rimg.shape[0] == 1 else rel < 0.05, f"{name}: median rel diff {rel:.5f}"


@pytest.mark.parametrize("name", VOLUMES_3D)
def test_gradient_magnitude_matches_sitk_on_real_volume(name):
    rimg = load_ritk(name)
    simg = load_sitk(name)
    ra = ritk.filter.gradient_magnitude(rimg).to_numpy()
    sa = sitk_to_zyx(sitk.GradientMagnitude(simg))
    rel = _rel_max(ra, sa)
    assert rel < 0.02, f"{name}: gradient_magnitude rel diff {rel:.4f}"


# ── Statistics parity on real data ────────────────────────────────────────────


@pytest.mark.parametrize("name", IMAGES_ANY)
def test_statistics_match_sitk_on_real_image(name):
    rimg = load_ritk(name)
    simg = load_sitk(name)
    rs = ritk.statistics.compute_statistics(rimg)
    f = sitk.StatisticsImageFilter()
    f.Execute(simg)
    rng = max(f.GetMaximum() - f.GetMinimum(), 1e-6)
    assert abs(rs["min"] - f.GetMinimum()) / rng < 1e-5, f"{name}: min"
    assert abs(rs["max"] - f.GetMaximum()) / rng < 1e-5, f"{name}: max"
    assert abs(rs["mean"] - f.GetMean()) / rng < 1e-4, f"{name}: mean"
    # SimpleITK GetSigma is the population (ddof=0) std; allow ddof difference.
    assert abs(rs["std"] - f.GetSigma()) / rng < 1e-2, f"{name}: std"


def test_histogram_matching_matches_sitk_on_real_pair():
    """ritk.histogram_match follows ITK's HistogramMatchingImageFilter (quantile
    landmarks + threshold-at-mean), matching sitk on the cthead1 → RA-slice pair.

    Regression: ritk used an exact full-CDF specification, which ignored ITK's
    mean threshold and K-landmark mapping (its output min did not reach the
    reference min and it diverged ~15 % in the tails). The landmark algorithm now
    agrees with sitk to well under 1 % across match-point / threshold settings.
    """
    src_s = load_sitk("cthead1-Float.mha")
    src_r = load_ritk("cthead1-Float.mha")
    ra = load_sitk("RA-Short.nrrd")
    ref_s = sitk.Cast(ra[:, :, 32], sitk.sitkFloat32)
    ref_r = ritk.Image(
        np.ascontiguousarray(sitk_to_zyx(ref_s).astype(np.float32)), spacing=[1, 1, 1]
    )

    for match_points, threshold in [(7, True), (7, False), (20, True), (1, True)]:
        rk = np.asarray(
            ritk.statistics.histogram_match(
                src_r,
                ref_r,
                num_bins=256,
                num_match_points=match_points,
                threshold_at_mean=threshold,
            ).to_numpy(),
            dtype=np.float64,
        )
        sk = sitk_to_zyx(
            sitk.HistogramMatching(
                src_s,
                ref_s,
                numberOfHistogramLevels=256,
                numberOfMatchPoints=match_points,
                thresholdAtMeanIntensity=threshold,
            )
        ).astype(np.float64)
        rng = max(float(np.abs(sk).max()), 1e-9)
        rel = float(np.abs(rk - sk).max()) / rng
        assert rel < 1e-2, f"histmatch mp={match_points} thr={threshold}: rel {rel:.2e}"


# ── Segmentation parity on real data ──────────────────────────────────────────


def test_confidence_connected_matches_sitk_on_cthead():
    """ritk.confidence_connected_segment matches sitk.ConfidenceConnected.

    Regression: ritk advanced one BFS ring per iteration, so it under-segmented
    ~180x (41 px vs sitk's 7387 on this seed). It now re-floods the whole region
    each iteration like ITK. ritk's API takes explicit initial bounds where ITK
    derives them from the seed neighbourhood, so to compare like-for-like the
    initial interval is set to ITK's neighbourhood (radius 1) mean ± k·σ; from the
    second iteration on both algorithms use identical region statistics and
    converge to the same region.
    """
    simg = load_sitk("cthead1-Float.mha")
    rimg = load_ritk("cthead1-Float.mha")
    arr = sitk_to_zyx(simg)[0].astype(np.float64)  # [y, x]

    sx, sy, radius, mult, iters = 128, 128, 1, 2.5, 8

    seg_s = sitk.ConfidenceConnected(
        simg,
        seedList=[(sx, sy)],
        numberOfIterations=iters,
        multiplier=mult,
        initialNeighborhoodRadius=radius,
        replaceValue=1,
    )
    fg_s = sitk_to_zyx(seg_s)[0] > 0.5

    # ITK's initial interval: mean ± k·σ over the (2r+1)² seed neighbourhood (N-1 σ).
    nb = arr[sy - radius : sy + radius + 1, sx - radius : sx + radius + 1].ravel()
    mean, std = nb.mean(), nb.std(ddof=1)
    seg_r = ritk.segmentation.confidence_connected_segment(
        rimg,
        seed=[0, sy, sx],
        initial_lower=float(mean - mult * std),
        initial_upper=float(mean + mult * std),
        multiplier=mult,
        max_iterations=iters,
    )
    fg_r = np.asarray(seg_r.to_numpy())[0] > 0.5

    # The grown region must be substantial (guards against the ring-growth defect).
    assert fg_s.sum() > 1000, f"sitk region unexpectedly small: {fg_s.sum()}"
    inter = np.logical_and(fg_r, fg_s).sum()
    dice = 2.0 * inter / (fg_r.sum() + fg_s.sum())
    assert dice > 0.99, f"Dice {dice:.4f} (ritk {fg_r.sum()} vs sitk {fg_s.sum()})"


# ── Registration on a real brain pair ─────────────────────────────────────────


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64) - a.mean()
    b = b.ravel().astype(np.float64) - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def test_demons_improves_ncc_on_real_brain_pair():
    fixed = load_ritk("OAS1_0001_MR1_mpr-1_anon.nrrd")
    moving = load_ritk("OAS1_0002_MR1_mpr-1_anon.nrrd")
    fa, ma = fixed.to_numpy(), moving.to_numpy()
    # Min-max normalize both to [0,1] so Demons (intensity-based) is well-posed
    # across the two subjects.
    fn = ritk.statistics.minmax_normalize(fixed)
    mn = ritk.statistics.minmax_normalize(moving)
    ncc_before = _ncc(fa, ma)
    warped, _ = ritk.registration.demons_register(fn, mn, max_iterations=40, sigma_diffusion=1.5)
    ncc_after = _ncc(fn.to_numpy(), warped.to_numpy())
    assert ncc_after >= ncc_before, (
        f"Demons must not reduce NCC on real brain pair: {ncc_before:.4f} -> {ncc_after:.4f}"
    )


# ── ritk native I/O on the real corpus ────────────────────────────────────────


@pytest.mark.parametrize("name", ["RA-Float.nrrd", "RA-Short.nrrd", "Ramp-Up-Short.nrrd"])
def test_ritk_reads_raw_3d_image_identically_to_sitk(name):
    """ritk's own reader matches SimpleITK byte-for-byte on raw-encoded 3-D files."""
    path = fetch(name)
    ra = ritk.io.read_image(path).to_numpy()
    sa = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
    assert ra.shape == sa.shape, f"{name}: shape {ra.shape} != {sa.shape}"
    assert float(np.max(np.abs(ra - sa))) == 0.0, f"{name}: ritk read differs from sitk"


def test_ritk_reads_gzip_nrrd_brain_mri():
    """ritk reads gzip-encoded NRRD (the OAS1 brain MRIs) identically to sitk."""
    path = fetch("OAS1_0001_MR1_mpr-1_anon.nrrd")
    ra = ritk.io.read_image(path).to_numpy()
    sa = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
    assert ra.shape == sa.shape
    assert float(np.max(np.abs(ra - sa))) == 0.0


# ── Write round-trips: ritk writes → SimpleITK reads back, metadata preserved ──


@pytest.mark.parametrize("ext", [".mha", ".nrrd"])
def test_ritk_write_roundtrips_metadata_through_sitk(ext, tmp_path):
    """ritk writing a volume with an anisotropic origin and an axis-permuted
    direction (OAS1) must produce a file SimpleITK reads back identically.

    Guards two I/O conventions: the NRRD writer's space declaration (LPS, not RAS
    — RAS made sitk negate the x/y origin) and the MetaImage TransformMatrix
    layout (ITK row-major axis directions, i.e. the transpose of the column-major
    direction matrix — writing columns transposed an axis-permuted direction).
    """
    path = fetch("OAS1_0001_MR1_mpr-1_anon.nrrd")
    orig = sitk.ReadImage(path)
    ri = ritk.io.read_image(path)

    out = str(tmp_path / f"roundtrip{ext}")
    ritk.io.write_image(ri, out)
    back = sitk.ReadImage(out)

    # Pixels byte-exact.
    assert float(np.max(np.abs(
        sitk.GetArrayFromImage(orig).astype(np.float64)
        - sitk.GetArrayFromImage(back).astype(np.float64)
    ))) == 0.0
    # Geometry preserved.
    assert np.allclose(orig.GetSpacing(), back.GetSpacing(), atol=1e-4), "spacing"
    assert np.allclose(orig.GetOrigin(), back.GetOrigin(), atol=1e-3), (
        f"origin {orig.GetOrigin()} vs {back.GetOrigin()}"
    )
    assert np.allclose(orig.GetDirection(), back.GetDirection(), atol=1e-4), (
        f"direction {orig.GetDirection()} vs {back.GetDirection()}"
    )


@pytest.mark.parametrize("ext", [".mha", ".nrrd"])
def test_ritk_reads_sitk_written_oblique_direction(ext, tmp_path):
    """ritk reads a SimpleITK-written volume with an axis-permuted direction and
    anisotropic origin/spacing, preserving geometry on a sitk→ritk→sitk loop.

    Validates the MetaImage TransformMatrix *reader* (ITK row-major axis
    directions) — distinct from the OAS1 test, which reads from NRRD.
    """
    img = sitk.GetImageFromArray(np.arange(4 * 5 * 6, dtype=np.float32).reshape(6, 5, 4))
    img.SetDirection((0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0))
    img.SetSpacing((2.0, 3.0, 4.0))
    img.SetOrigin((10.0, 20.0, 30.0))
    src = str(tmp_path / f"src{ext}")
    sitk.WriteImage(img, src)

    ri = ritk.io.read_image(src)
    out = str(tmp_path / f"rt{ext}")
    ritk.io.write_image(ri, out)
    back = sitk.ReadImage(out)

    assert float(np.max(np.abs(
        sitk.GetArrayFromImage(img).astype(np.float64)
        - sitk.GetArrayFromImage(back).astype(np.float64)
    ))) == 0.0
    assert np.allclose(img.GetDirection(), back.GetDirection(), atol=1e-4)
    assert np.allclose(img.GetOrigin(), back.GetOrigin(), atol=1e-3)
    assert np.allclose(img.GetSpacing(), back.GetSpacing(), atol=1e-4)


# ── ritk native 2-D reads (promoted to z=1) ───────────────────────────────────
#
# ritk's `Image` is intrinsically 3-D; its MetaImage, NRRD, and PNG readers
# promote a 2-D file to a degenerate `[1, Y, X]` (z=1) volume at read time.
# Each case reads the file natively with ritk and asserts byte-exact agreement
# with SimpleITK's `GetArrayFromImage` after adding the singleton z-axis.


@pytest.mark.parametrize(
    "name",
    [
        "cthead1-Float.mha",  # 2-D MetaImage, MET_FLOAT
        "BrainProtonDensitySlice.png",  # 2-D PNG, 8-bit grayscale
        "Gaussian_1.5.nrrd",  # 2-D gzip-encoded NRRD
    ],
)
def test_ritk_reads_2d_image_as_z1_identically_to_sitk(name):
    """ritk reads a 2-D file as `[1, Y, X]`, matching sitk's array with z=1 added."""
    path = fetch(name)
    ra = ritk.io.read_image(path).to_numpy()
    sa = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
    assert sa.ndim == 2, f"{name}: expected a 2-D SimpleITK image, got ndim {sa.ndim}"
    sa = sa[None, :, :]
    assert ra.shape == sa.shape, f"{name}: shape {ra.shape} != {sa.shape}"
    assert ra.shape[0] == 1, f"{name}: 2-D file must promote to a single z-slice"
    assert float(np.max(np.abs(ra - sa))) == 0.0, f"{name}: ritk 2-D read differs from sitk"
