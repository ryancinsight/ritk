use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
    let (vals, _) = ritk_tensor_ops::extract_vec(img).unwrap();
    vals
}

/// Build a spherical signed-distance level set with φ < 0 inside.
fn sphere_phi(nz: usize, ny: usize, nx: usize, radius: f64) -> Vec<f32> {
    let n = nz * ny * nx;
    let (cz, cy, cx) = (
        (nz as f64 - 1.0) * 0.5,
        (ny as f64 - 1.0) * 0.5,
        (nx as f64 - 1.0) * 0.5,
    );
    (0..n)
        .map(|i| {
            let iz = i / (ny * nx);
            let iy = (i / nx) % ny;
            let ix = i % nx;
            let dz = iz as f64 - cz;
            let dy = iy as f64 - cy;
            let dx = ix as f64 - cx;
            ((dz * dz + dy * dy + dx * dx).sqrt() - radius) as f32
        })
        .collect()
}

// ── 1. Level set moves when feature has spatial gradient ─────────────────

/// A linear z-ramp feature produces a spatially varying edge potential F.
/// The level set evolution ∂φ/∂t = F(curvature − propagation)|∇φ| is therefore
/// non-uniform.  With `curvature_scaling = 1.0` and a convex sphere (κ = 2/R > 0)
/// the inside region contracts.
///
/// Verification: after 20 iterations the total absolute change
/// Σ|φ_out − φ_in| must be non-trivially large (> 0.5 per voxel on average
/// near the zero crossing is not required — we only need Σ > 1e-3).
#[test]
fn test_canny_level_set_moves_with_gradient_feature() {
    let [nz, ny, nx] = [20usize, 20, 20];
    let n = nz * ny * nx;
    let r_initial = 7.0_f64;

    let phi_vals = sphere_phi(nz, ny, nx, r_initial);

    // Linear ramp along z: value in [0, 1].
    let feat_vals: Vec<f32> = (0..n)
        .map(|i| {
            let iz = i / (ny * nx);
            iz as f32 / (nz - 1) as f32
        })
        .collect();

    let phi_img = make_image(phi_vals.clone(), [nz, ny, nx]);
    let feat_img = make_image(feat_vals, [nz, ny, nx]);

    let filter = CannySegmentationLevelSet {
        number_of_iterations: 20,
        canny_threshold: 0.5,
        canny_variance: 1.0,
        curvature_scaling: 1.0,
        propagation_scaling: 0.0,
        dt: 0.05,
        max_rms_error: 1e-6, // don't stop early
    };

    let out = filter.apply(&phi_img, &feat_img).unwrap();
    let result = extract_vals(&out);

    let total_change: f32 = phi_vals
        .iter()
        .zip(result.iter())
        .map(|(&a, &b)| (b - a).abs())
        .sum();

    assert!(
        total_change > 1e-3,
        "level set should move with gradient feature: total_change={total_change:.4e}"
    );

    // All output values must be finite.
    assert!(
        result.iter().all(|v| v.is_finite()),
        "level set contains non-finite values"
    );
}

// ── 2. Level set stays finite with constant feature ───────────────────────

/// A constant feature image has zero gradient magnitude everywhere: F = 1.
/// The only active force is the curvature term.  For a convex sphere with
/// curvature_scaling = 1.0 and very small dt, the contour barely moves —
/// verified by confirming all output values are finite and that the shape
/// is preserved (same voxel count).
///
/// This also exercises the code path where `gaussian_smooth` is called with a
/// non-trivial sigma and the resulting gradient magnitude is all zero.
#[test]
fn test_canny_level_set_finite_with_constant_feature() {
    let [nz, ny, nx] = [16usize, 16, 16];
    let n = nz * ny * nx;

    let phi_vals = sphere_phi(nz, ny, nx, 5.0);
    let feat_vals = vec![0.5_f32; n];

    let phi_img = make_image(phi_vals, [nz, ny, nx]);
    let feat_img = make_image(feat_vals, [nz, ny, nx]);

    let filter = CannySegmentationLevelSet {
        number_of_iterations: 50,
        ..Default::default()
    };

    let out = filter.apply(&phi_img, &feat_img).unwrap();
    let result = extract_vals(&out);

    assert_eq!(result.len(), n, "output size must match input");
    assert_eq!(out.shape(), [nz, ny, nx], "output shape must match input");

    let any_non_finite = result.iter().any(|v| !v.is_finite());
    assert!(
        !any_non_finite,
        "level set contains non-finite values after 50 iterations with constant feature"
    );
}
