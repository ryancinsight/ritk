use super::*;

/// Partition of unity: if all control points = 1.0, every evaluated voxel = 1.0.
///
/// Proof: s = Σ_{a,b,c} Bz[a]·By[b]·Bx[c]·1 = (ΣBz)·(ΣBy)·(ΣBx) = 1³ = 1. ∎
#[test]
fn constant_control_points_partition_of_unity() {
    let ctrl = vec![1.0f64; 4 * 4 * 4];
    let cg = [4usize, 4, 4];
    let dims = [10usize, 10, 10];

    let result = bspline_evaluate(&ctrl, cg, dims);

    assert_eq!(result.len(), 10 * 10 * 10);
    for (vi, &v) in result.iter().enumerate() {
        assert!(
            (v - 1.0_f32).abs() < 1e-5,
            "voxel {vi}: expected 1.0, got {v:.8}"
        );
    }
}

/// Partition of unity holds with larger control grid and non-cubic dimensions.
#[test]
fn partition_of_unity_larger_grid() {
    let cg = [6usize, 5, 7];
    let dims = [15usize, 12, 20];
    let ctrl = vec![1.0f64; 6 * 5 * 7];

    let result = bspline_evaluate(&ctrl, cg, dims);

    assert_eq!(result.len(), 15 * 12 * 20);
    for (vi, &v) in result.iter().enumerate() {
        assert!(
            (v - 1.0_f32).abs() < 1e-5,
            "voxel {vi}: expected 1.0, got {v:.8}"
        );
    }
}

/// Basis values sum to 1 for a dense sweep of u ∈ [0, 1].
#[test]
fn basis_sums_to_one_over_unit_interval() {
    for i in 0..=1000 {
        let u = i as f64 / 1000.0;
        let b = cubic_bspline_basis(u);
        let sum = b[0] + b[1] + b[2] + b[3];
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "u={u:.4}: basis sum = {sum:.15} ≠ 1"
        );
    }
}

/// Single-level MBA preserves the low-frequency structure of a smooth field.
///
/// `bspline_fit` is the Lee–Wolberg–Shin single-level (scattered-data)
/// approximation — the kernel ITK's `BSplineScatteredDataPointSetToImageFilter`
/// runs with `NumberOfLevels = 1`, exactly as N4 configures it. It is a smoother,
/// not an interpolator: it does not reproduce a field pointwise (nor preserve its
/// mean), but it tracks the spatial gradient with high fidelity, which is what a
/// low-frequency bias estimator needs. A trilinear ramp reconstructs with
/// correlation ≳ 0.96 (measured 0.968).
#[test]
fn mba_preserves_low_frequency_ramp() {
    let dims = [10usize, 10, 10];
    let cg = [6usize, 6, 6];
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    // Trilinear field: f(iz,iy,ix) ∈ [0, 0.65].
    let field: Vec<f32> = (0..n)
        .map(|vi| {
            let iz = vi / (ny * nx);
            let iy = (vi % (ny * nx)) / nx;
            let ix = vi % nx;
            0.30 * (iz as f32 / (nz - 1) as f32)
                + 0.20 * (iy as f32 / (ny - 1) as f32)
                + 0.15 * (ix as f32 / (nx - 1) as f32)
        })
        .collect();

    let ctrl = bspline_fit(&field, dims, cg).expect("bspline_fit failed");
    let approx = bspline_evaluate(&ctrl, cg, dims);

    // Pearson correlation between the field and its reconstruction.
    let mf = field.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let ma = approx.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let (mut sff, mut saa, mut sfa) = (0.0f64, 0.0f64, 0.0f64);
    for (&f, &a) in field.iter().zip(approx.iter()) {
        let (df, da) = (f as f64 - mf, a as f64 - ma);
        sff += df * df;
        saa += da * da;
        sfa += df * da;
    }
    let corr = sfa / (sff.sqrt() * saa.sqrt());
    assert!(corr > 0.96, "ramp reconstruction correlation {corr:.4} ≤ 0.96");
}

/// MBA of a zero residual is exactly zero (mock-detection: the output must be a
/// genuine function of the input, not a fixed lattice).
#[test]
fn mba_fit_of_zero_is_zero() {
    let dims = [8usize, 8, 8];
    let cg = [4usize, 4, 4];
    let ctrl = bspline_fit(&vec![0.0f32; 8 * 8 * 8], dims, cg).expect("bspline_fit failed");
    let approx = bspline_evaluate(&ctrl, cg, dims);
    let max_abs = approx.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(max_abs < 1e-6, "fit of zero is non-zero: max |v| = {max_abs}");
}

/// Single-level MBA of a constant produces a smooth, bounded, real (non-degenerate)
/// field. It over-shoots the constant in the interior (mean is not preserved — a
/// known single-level property), but stays in a bounded band and varies smoothly.
/// In N4 this is harmless: a constant intensity has a ≈zero sharpening residual,
/// so this fit is never applied to a literal constant in the EM loop.
#[test]
fn mba_constant_is_smooth_and_bounded() {
    let dims = [8usize, 8, 8];
    let cg = [4usize, 4, 4];
    let field = vec![2.5f32; 8 * 8 * 8];

    let ctrl = bspline_fit(&field, dims, cg).expect("bspline_fit failed");
    let approx = bspline_evaluate(&ctrl, cg, dims);

    // Bounded band: measured reconstruction ∈ [2.648, 3.591].
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &approx {
        lo = lo.min(v);
        hi = hi.max(v);
        assert!(v.is_finite() && v > 0.0, "non-finite/non-positive value {v}");
    }
    assert!(lo > 2.0 && hi < 4.0, "constant fit band [{lo:.3}, {hi:.3}] outside [2.0, 4.0]");
    // Real work: the fit is not a degenerate flat lattice.
    assert!(hi - lo > 1e-3, "constant fit is degenerate (flat): span {:.5}", hi - lo);

    // Smoothness: adjacent voxels along z differ by a small amount (≤ 0.25).
    let mut max_adj = 0.0f32;
    for z in 0..7 {
        for y in 0..8 {
            for x in 0..8 {
                let a = approx[(z * 8 + y) * 8 + x];
                let b = approx[((z + 1) * 8 + y) * 8 + x];
                max_adj = max_adj.max((a - b).abs());
            }
        }
    }
    assert!(max_adj < 0.25, "constant fit not smooth: max adjacent Δz = {max_adj:.4}");
}
