"""Performance + result audit: ritk (PyO3) vs SimpleITK.

Times image I/O, filters, statistics, and registration on both stacks and
reports, per operation: median wall time for each, the ritk/sitk speedup, and a
result-agreement metric (max abs difference on the interior, or NCC for
registration).  Run directly:

    PYTHONIOENCODING=utf-8 py crates/ritk-python/tests/audit_sitk_performance.py

This is an audit script, not a pytest module (no ``test_`` prefix).
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time

import numpy as np
import SimpleITK as sitk

import ritk


# ── timing helpers ────────────────────────────────────────────────────────────
def _bench(fn, runs=5, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _interior(a, m=4):
    if m <= 0:
        return a
    s = tuple(slice(m, -m) for _ in range(a.ndim))
    return a[s]


def _max_diff(a, b, crop=4):
    a, b = _interior(a, crop).astype(np.float64), _interior(b, crop).astype(np.float64)
    return float(np.max(np.abs(a - b)))


def _ncc(a, b):
    a = a.ravel().astype(np.float64) - a.mean()
    b = b.ravel().astype(np.float64) - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


ROWS = []


def _row(op, t_ritk, t_sitk, agree, note=""):
    speed = t_sitk / t_ritk if t_ritk > 0 else float("inf")
    ROWS.append((op, t_ritk * 1e3, t_sitk * 1e3, speed, agree, note))


def _to_sitk(arr, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr, dtype=np.float32))
    img.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
    return img


def _ritk(arr, spacing=(1.0, 1.0, 1.0)):
    return ritk.Image(np.ascontiguousarray(arr, dtype=np.float32), spacing=list(spacing))


# ── workloads ─────────────────────────────────────────────────────────────────
def _volume(shape=(64, 128, 128), seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1000, shape[0] * shape[1] * shape[2], dtype=np.float32).reshape(shape)
    return (base + rng.standard_normal(shape).astype(np.float32) * 50).astype(np.float32)


def audit_io():
    arr = _volume()
    rimg = _ritk(arr)
    simg = _to_sitk(arr)
    d = tempfile.mkdtemp()
    for ext in [".nii.gz", ".nrrd", ".mha"]:
        rp = os.path.join(d, "r" + ext)
        sp = os.path.join(d, "s" + ext)

        def w_ritk():
            ritk.io.write_image(rimg, rp)

        def w_sitk():
            sitk.WriteImage(simg, sp)

        tw_r, tw_s = _bench(w_ritk, runs=3), _bench(w_sitk, runs=3)
        # cross-read agreement: ritk reads sitk's file
        cross = ritk.io.read_image(sp).to_numpy()
        agree = _max_diff(cross, arr, crop=0)

        def r_ritk():
            ritk.io.read_image(rp)

        def r_sitk():
            sitk.GetArrayFromImage(sitk.ReadImage(sp))

        tr_r, tr_s = _bench(r_ritk, runs=3), _bench(r_sitk, runs=3)
        _row(f"write {ext}", tw_r, tw_s, "—")
        _row(f"read {ext}", tr_r, tr_s, agree, "ritk reads sitk file")


def audit_filters():
    arr = _volume()
    rimg, simg = _ritk(arr), _to_sitk(arr)

    cases = [
        (
            "discrete_gaussian(var=4)",
            lambda: ritk.filter.discrete_gaussian(rimg, variance=4.0, maximum_error=0.01),
            lambda: sitk.DiscreteGaussian(simg, variance=4.0, maximumError=0.01),
        ),
        (
            "median(radius=1)",
            lambda: ritk.filter.median_filter(rimg, radius=1),
            lambda: sitk.Median(simg, [1, 1, 1]),
        ),
        (
            "gradient_magnitude",
            lambda: ritk.filter.gradient_magnitude(rimg),
            lambda: sitk.GradientMagnitude(simg),
        ),
        (
            "rescale_intensity(0,255)",
            lambda: ritk.filter.rescale_intensity(rimg, 0.0, 255.0),
            lambda: sitk.RescaleIntensity(simg, 0.0, 255.0),
        ),
        (
            "recursive_gaussian(s=2)",
            lambda: ritk.filter.recursive_gaussian(rimg, sigma=2.0),
            lambda: sitk.SmoothingRecursiveGaussian(simg, 2.0),
        ),
    ]
    for name, rf, sf in cases:
        try:
            ra = rf().to_numpy()
            sa = sitk.GetArrayFromImage(sf())
            agree = _max_diff(ra, sa)
            t_r, t_s = _bench(rf, runs=5), _bench(sf, runs=5)
            _row(name, t_r, t_s, f"{agree:.3e}")
        except Exception as e:  # noqa: BLE001
            _row(name, float("nan"), float("nan"), "ERR", str(e)[:60])

    # sobel_gradient is a *normalized physical-gradient* operator (intensity/mm),
    # not sitk's raw SobelEdgeDetection kernel sum, so they are not directly
    # comparable; time both and note the definitional difference.
    try:
        t_r = _bench(lambda: ritk.filter.sobel_gradient(rimg), runs=5)
        t_s = _bench(lambda: sitk.SobelEdgeDetection(simg), runs=5)
        _row("sobel_gradient", t_r, t_s, "n/a", "ritk=physical grad; sitk=raw kernel")
    except Exception as e:  # noqa: BLE001
        _row("sobel_gradient", float("nan"), float("nan"), "ERR", str(e)[:60])


def audit_statistics():
    arr = _volume()
    rimg, simg = _ritk(arr), _to_sitk(arr)

    def r_stats():
        return ritk.statistics.compute_statistics(rimg)

    def s_stats():
        f = sitk.StatisticsImageFilter()
        f.Execute(simg)
        return f

    rs = r_stats()
    sf = s_stats()
    diffs = {
        "mean": abs(rs["mean"] - sf.GetMean()),
        "std": abs(rs["std"] - sf.GetSigma()),
        "min": abs(rs["min"] - sf.GetMinimum()),
        "max": abs(rs["max"] - sf.GetMaximum()),
    }
    worst = max(diffs.values())
    t_r, t_s = _bench(r_stats, runs=5), _bench(s_stats, runs=5)
    _row("statistics(mean/std/min/max)", t_r, t_s, f"{worst:.3e}", "ddof handled separately")


def _sphere(size=32, radius=7):
    c = size // 2
    z, y, x = np.mgrid[:size, :size, :size]
    return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= radius**2).astype(np.float32)


def audit_registration():
    fixed = _sphere()
    moving = np.roll(fixed, 3, axis=2).astype(np.float32)
    rf, rm = _ritk(fixed), _ritk(moving)
    sf, sm = _to_sitk(fixed), _to_sitk(moving)
    ncc_before = _ncc(fixed, moving)

    # ritk Thirion demons
    def r_demons():
        return ritk.registration.demons_register(rf, rm, max_iterations=50, sigma_diffusion=1.0)

    warped, _disp = r_demons()
    ncc_r = _ncc(fixed, warped.to_numpy())
    t_r = _bench(r_demons, runs=3, warmup=0)

    # sitk Demons
    def s_demons():
        dem = sitk.DemonsRegistrationFilter()
        dem.SetNumberOfIterations(50)
        dem.SetStandardDeviations(1.0)
        field = dem.Execute(sf, sm)
        tx = sitk.DisplacementFieldTransform(sitk.Image(field))
        return sitk.Resample(sm, sf, tx, sitk.sitkLinear)

    try:
        sw = s_demons()
        ncc_s = _ncc(fixed, sitk.GetArrayFromImage(sw))
        t_s = _bench(s_demons, runs=3, warmup=0)
        note = f"NCC {ncc_before:.3f}→ ritk {ncc_r:.3f} / sitk {ncc_s:.3f}"
    except Exception as e:  # noqa: BLE001
        t_s = float("nan")
        note = f"sitk demons err: {str(e)[:40]}; ritk NCC {ncc_before:.3f}→{ncc_r:.3f}"
    _row("demons_register(50it, 32³)", t_r, t_s, f"{ncc_r:.3f}", note)


def main():
    print(f"ritk @ {ritk.__file__}")
    print(f"SimpleITK {sitk.Version_VersionString()}\n")
    audit_io()
    audit_filters()
    audit_statistics()
    audit_registration()

    hdr = f"{'operation':<32}{'ritk ms':>10}{'sitk ms':>10}{'speedup':>9}{'agree':>14}  note"
    print(hdr)
    print("-" * len(hdr))
    for op, tr, ts, sp, agree, note in ROWS:
        spc = f"{sp:.2f}x" if sp == sp and sp != float("inf") else "—"
        trs = f"{tr:.2f}" if tr == tr else "ERR"
        tss = f"{ts:.2f}" if ts == ts else "ERR"
        print(f"{op:<32}{trs:>10}{tss:>10}{spc:>9}{str(agree):>14}  {note}")


if __name__ == "__main__":
    main()
