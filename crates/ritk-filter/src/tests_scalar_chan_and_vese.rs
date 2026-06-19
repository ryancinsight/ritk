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
///
/// `phi(x) = distance_from_centre - radius` — negative inside the sphere.
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

/// Build a binary sphere feature image: 1.0 inside, 0.0 outside.
fn sphere_feature(nz: usize, ny: usize, nx: usize, radius: f64) -> Vec<f32> {
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
            let d = (dz * dz + dy * dy + dx * dx).sqrt();
            if d <= radius {
                1.0_f32
            } else {
                0.0
            }
        })
        .collect()
}

// ── 1. Chan-Vese contracts a large initial LS toward the feature boundary ─

/// Setup: 20×20×20, feature sphere of radius 6, initial level set sphere of
/// radius 9 (over-estimates the feature region by ~3 voxels).
///
/// Chan-Vese data-fidelity forces: in the annular band 6 < d < 9 the feature
/// is 0 but those voxels are classified inside (φ < 0).  The inside mean
/// c₁ ≈ 0.30 (fraction of feature inside the large sphere), outside mean
/// c₂ ≈ 0.  At the outer boundary (u₀ = 0):
///   diff1² = (0 − 0.30)² ≈ 0.09 → positive force → φ increases → voxels
///   move toward outside.
///
/// After 50 iterations with dt = 0.1 the contraction should reduce the
/// inside voxel count by at least 5% from the initial ~3054 to ≤ 2900.
#[test]
fn test_chan_and_vese_contracts_toward_feature_boundary() {
    let [nz, ny, nx] = [20usize, 20, 20];
    let r_feature = 6.0_f64;
    let r_initial = 9.0_f64;

    let feature_vals = sphere_feature(nz, ny, nx, r_feature);
    let phi_init_vals = sphere_phi(nz, ny, nx, r_initial);

    let initial_inside = phi_init_vals.iter().filter(|&&v| v < 0.0).count();

    let feature_img = make_image(feature_vals, [nz, ny, nx]);
    let phi_img = make_image(phi_init_vals, [nz, ny, nx]);

    let filter = ScalarChanAndVeseDenseLevelSet {
        number_of_iterations: 50,
        lambda1: 1.0,
        lambda2: 1.0,
        mu: 0.5,
        nu: 0.0,
        dt: 0.1,
        epsilon: 1.0,
    };

    let out = filter.apply(&phi_img, &feature_img).unwrap();
    let result = extract_vals(&out);

    let final_inside = result.iter().filter(|&&v| v < 0.0).count();

    assert!(
        final_inside < initial_inside,
        "Chan-Vese should contract initial_inside={initial_inside} → final_inside={final_inside}; \
         expected a decrease"
    );
    // Require at least 5% contraction (~152 voxels for initial ~3054).
    let contraction_frac = (initial_inside - final_inside) as f64 / initial_inside as f64;
    assert!(
        contraction_frac >= 0.05,
        "contraction fraction {:.2}% < 5% threshold (initial={initial_inside}, final={final_inside})",
        contraction_frac * 100.0
    );
}

// ── 2. Level set stays finite after 100 iterations ───────────────────────

/// Numerical stability: 100 PDE steps of the Chan-Vese evolution must not
/// produce any NaN or ±Inf in the output level set.
///
/// Verified by checking that every output value is finite.  This exercises the
/// Dirac (denominator ε² + φ²) and curvature (denominator |∇φ|² + ε_curv)
/// guard paths.
#[test]
fn test_chan_and_vese_stays_finite() {
    let [nz, ny, nx] = [16usize, 16, 16];
    let phi_vals = sphere_phi(nz, ny, nx, 5.0);
    let feat_vals = sphere_feature(nz, ny, nx, 5.0);

    let phi_img = make_image(phi_vals, [nz, ny, nx]);
    let feat_img = make_image(feat_vals, [nz, ny, nx]);

    let filter = ScalarChanAndVeseDenseLevelSet {
        number_of_iterations: 100,
        ..Default::default()
    };

    let out = filter.apply(&phi_img, &feat_img).unwrap();
    let result = extract_vals(&out);

    let any_non_finite = result.iter().any(|v| !v.is_finite());
    assert!(
        !any_non_finite,
        "level set contains non-finite values after 100 iterations"
    );
    // Output shape must be preserved.
    assert_eq!(out.shape(), [nz, ny, nx]);
}
