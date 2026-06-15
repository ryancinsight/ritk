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

/// Round-trip: fit a trilinear field (in the B-spline span) then re-evaluate.
/// A trilinear function is in the span of the cubic B-spline basis, so the
/// Tikhonov-regularised fit (λ=1e-6) must recover it to RMS < 0.1.
#[test]
fn round_trip_linear_field_rms_below_threshold() {
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

    let ctrl = bspline_fit(&field, dims, cg, 10_000).expect("bspline_fit failed");
    let approx = bspline_evaluate(&ctrl, cg, dims);

    let rms = (field
        .iter()
        .zip(approx.iter())
        .map(|(&r, &a)| ((r - a) as f64).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();

    assert!(rms < 0.1, "round-trip RMS {rms:.6} ≥ 0.1");
}

/// Fit to a constant field yields control points that reconstruct to ≈ constant.
#[test]
fn round_trip_constant_field() {
    let dims = [8usize, 8, 8];
    let cg = [4usize, 4, 4];
    let n = 8 * 8 * 8;
    let field = vec![2.5f32; n];

    let ctrl = bspline_fit(&field, dims, cg, 10_000).expect("bspline_fit failed");
    let approx = bspline_evaluate(&ctrl, cg, dims);

    for (vi, &v) in approx.iter().enumerate() {
        assert!(
            (v - 2.5_f32).abs() < 0.05,
            "voxel {vi}: expected ~2.5, got {v:.6}"
        );
    }
}
