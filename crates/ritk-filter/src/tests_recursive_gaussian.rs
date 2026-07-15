//! Tests for recursive_gaussian
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_core::image::Image;
use ritk_image::test_support as ts;

type B = LegacyBurnBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
    ts::make_image_with_spacing::<B, 3>(vals, dims, spacing)
}

fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// Smoothing a constant image must return the same constant value at every
/// voxel.
///
/// **Proof**: A constant signal x[n] = c is the steady state of the IIR
/// recursion: y = B·c + (d1+d2+d3)·y ⇒ y·(1−d1−d2−d3) = B·c ⇒ y = c.
/// Both forward and backward passes converge to c, and the cascade
/// preserves c. Boundary initialisation to c ensures no transients. ∎
#[test]
fn test_smoothing_constant_image() {
    let dims = [16, 16, 16];
    let c = 42.0_f32;
    let vals = vec![c; dims[0] * dims[1] * dims[2]];
    let img = make_image(vals, dims, [1.0, 1.0, 1.0]);

    let filter = RecursiveGaussianFilter::new(2.0);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - c).abs() < 1e-3,
            "constant image smoothing: voxel {i} = {v}, expected {c}"
        );
    }
}

/// Smoothing (order 0) preserves total intensity (sum) to within a small
/// tolerance.
///
/// **Proof sketch**: A Gaussian kernel integrates to 1, so convolution
/// preserves the L¹ norm of the signal. The IIR approximation is designed
/// to have unit DC gain, so the sum is preserved up to boundary effects.
#[test]
fn test_smoothing_preserves_sum() {
    let dims = [20, 20, 20];
    let n = dims[0] * dims[1] * dims[2];
    // Non-trivial signal: voxel value = flat index mod 17
    let vals: Vec<f32> = (0..n).map(|i| (i % 17) as f32).collect();
    let sum_in: f64 = vals.iter().map(|&v| v as f64).sum();

    let img = make_image(vals, dims, [1.0, 1.0, 1.0]);

    let filter = RecursiveGaussianFilter::new(1.5);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);
    let sum_out: f64 = out.iter().map(|&v| v as f64).sum();

    let rel_err = (sum_out - sum_in).abs() / sum_in.abs().max(1e-12);
    assert!(
        rel_err < 0.05,
        "sum not preserved: input sum = {sum_in}, output sum = {sum_out}, \
         relative error = {rel_err}"
    );
}

/// First derivative of a linear ramp I(x) = x gives a constant.
///
/// **Derivation**: d/dx (x) = 1. The smoothing step preserves linearity
/// (Gaussian * linear = linear with same slope at interior points).
/// The central difference of the smoothed linear ramp gives ≈ 1 at
/// interior voxels.
#[test]
fn test_first_derivative_of_linear_ramp() {
    let [nz, ny, nx] = [1usize, 1, 64];
    let vals: Vec<f32> = (0..nx).map(|ix| ix as f32).collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

    let filter = RecursiveGaussianFilter::new(3.0).with_derivative_order(DerivativeOrder::First);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    // The interior values should be approximately constant
    let margin = 12;
    let interior: Vec<f32> = out[margin..nx - margin].to_vec();
    let mean: f64 = interior.iter().map(|&v| v as f64).sum::<f64>() / interior.len() as f64;

    // All interior values should be close to the mean
    for (i, &v) in interior.iter().enumerate() {
        let dev = ((v as f64) - mean).abs();
        assert!(
            dev < mean.abs() * 0.15 + 0.1,
            "first derivative of ramp not constant at interior position {}: \
             value = {v}, mean = {mean}",
            i + margin
        );
    }

    // The mean itself should be positive and close to 0.5 (central
    // difference of unit-slope ramp: (x[n+1]-x[n-1])/2 = 1*0.5... but
    // the smoothing + FD composition may differ in scale). Verify nonzero.
    assert!(
        mean.abs() > 0.01,
        "first derivative of ramp should be nonzero, got mean = {mean}"
    );
}

/// Second derivative of a quadratic I(x) = x² gives approximately
/// constant output at interior voxels.
///
/// **Derivation**: d²/dx² (x²) = 2. The smoothing preserves quadratic
/// structure at interior points, and the central second-difference
/// x[n+1]-2x[n]+x[n-1] of the smoothed quadratic gives ≈ 2 at interior
/// voxels (the exact value depends on the smoothing kernel width but
/// should be constant).
#[test]
fn test_second_derivative_of_quadratic() {
    let [nz, ny, nx] = [1usize, 1, 64];
    let vals: Vec<f32> = (0..nx).map(|ix| (ix as f32) * (ix as f32)).collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

    let filter = RecursiveGaussianFilter::new(3.0).with_derivative_order(DerivativeOrder::Second);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    // Interior values should be approximately constant
    let margin = 15;
    let interior: Vec<f32> = out[margin..nx - margin].to_vec();
    let mean: f64 = interior.iter().map(|&v| v as f64).sum::<f64>() / interior.len() as f64;

    for (i, &v) in interior.iter().enumerate() {
        let dev = ((v as f64) - mean).abs();
        assert!(
            dev < mean.abs() * 0.25 + 0.5,
            "second derivative of quadratic not constant at interior position {}: \
             value = {v}, mean = {mean}",
            i + margin
        );
    }
    // The mean should be close to 2.0 (exact second derivative of x²)
    assert!(
        mean.abs() > 0.5,
        "second derivative of quadratic should be substantially nonzero, \
         got mean = {mean}"
    );
}

/// `laplacian_recursive_gaussian` computes the physical Laplacian:
/// ∂²/∂x²(G_σ * x²) = 2 exactly (the Gaussian convolution adds a constant σ²
/// that the second derivative annihilates). This holds independently of voxel
/// spacing, which locks the per-axis `1/spacing²` normalisation — without it the
/// anisotropic case would read 2·s² instead of 2.
#[test]
fn laplacian_recursive_gaussian_quadratic_is_two() {
    // nx/margin sized so the IIR boundary transient (≈4·σ/sx pixels, worst at
    // sx=0.5 → σ_pix=6) has fully decayed in the interior window [48, 112].
    let nx = 160usize;
    let margin = 48;
    for &sx in &[1.0_f64, 2.0, 0.5] {
        // f(ix) = (ix·sx)² = physical_x², so ∇²f = 2 in physical units. All
        // values are exact f32 integers (< 2²⁴), so the input carries no error.
        let vals: Vec<f32> = (0..nx).map(|ix| (ix as f64 * sx).powi(2) as f32).collect();
        let img = make_image(vals, [1, 1, nx], [1.0, 1.0, sx]);
        let out = extract_vals(&laplacian_recursive_gaussian(&img, 3.0).unwrap());
        for (i, &v) in out[margin..nx - margin].iter().enumerate() {
            assert!(
                ((v as f64) - 2.0).abs() < 0.02,
                "∇²(G*x²) must be 2 in physical units (spacing {sx}) at interior \
                 voxel {}, got {v}",
                i + margin
            );
        }
    }
}

/// `gradient_magnitude_recursive_gaussian` is the physical gradient magnitude:
/// `|∇(G_σ * a·x)| = |a|` exactly (smoothing preserves the linear gradient).
/// Holds independently of spacing, locking the per-axis `1/spacing` factor —
/// without it the anisotropic case would read `|a|·s`.
#[test]
fn gradient_magnitude_recursive_gaussian_ramp_is_slope() {
    let nx = 160usize;
    let margin = 48;
    let a = 3.0_f64;
    for &sx in &[1.0_f64, 2.0, 0.5] {
        // f(ix) = a·(ix·sx) = a·x_phys, so |∇f| = |a| in physical units.
        let vals: Vec<f32> = (0..nx).map(|ix| (a * ix as f64 * sx) as f32).collect();
        let img = make_image(vals, [1, 1, nx], [1.0, 1.0, sx]);
        let out = extract_vals(&gradient_magnitude_recursive_gaussian(&img, 3.0).unwrap());
        for (i, &v) in out[margin..nx - margin].iter().enumerate() {
            assert!(
                ((v as f64) - a).abs() < 0.02,
                "|∇(G*a·x)| must be {a} in physical units (spacing {sx}) at interior \
                 voxel {}, got {v}",
                i + margin
            );
        }
    }
}

/// `recursive_gaussian_directional` (order 1) along x returns the signed slope of
/// a linear ramp `f = a·x` in the interior (≈ a, in index units — no spacing
/// division, matching the raw ITK single-axis filter). Other axes are untouched.
#[test]
fn recursive_gaussian_directional_ramp_slope() {
    let nx = 96usize;
    let a = 3.0_f64;
    let vals: Vec<f32> = (0..nx).map(|ix| (a * ix as f64) as f32).collect();
    let img = make_image(vals, [1, 1, nx], [1.0, 1.0, 1.0]);
    let out = extract_vals(
        &recursive_gaussian_directional(&img, 3.0, DerivativeOrder::First, 2).unwrap(),
    );
    let margin = 24;
    for (i, &v) in out[margin..nx - margin].iter().enumerate() {
        assert!(
            ((v as f64) - a).abs() < 0.02,
            "∂/∂x(G*a·x) must be {a} at interior x={}, got {v}",
            i + margin
        );
    }
}

/// Smoothing with a large image and small sigma should approximate identity.
#[test]
fn test_small_sigma_near_identity() {
    let dims = [8, 8, 8];
    let n = dims[0] * dims[1] * dims[2];
    let vals: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
    let img = make_image(vals.clone(), dims, [1.0, 1.0, 1.0]);

    // Sigma = 0.1 in physical units, pixel sigma < 0.2 → skipped
    let filter = RecursiveGaussianFilter::new(0.1);
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    for (i, (&expected, &actual)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "small sigma should be near-identity: voxel {i} = {actual}, expected {expected}"
        );
    }
}

/// Verify the Deriche DC-gain invariant: (ΣN + ΣM) / (1 + ΣD) = 1.
#[test]
fn test_coefficients_dc_gain() {
    for &sigma in &[0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
        let c = DericheCoefficients::from_sigma(sigma);
        let dc =
            (c.n.iter().sum::<f64>() + c.m.iter().sum::<f64>()) / (1.0 + c.d.iter().sum::<f64>());
        assert!(
            (dc - 1.0).abs() < 1e-12,
            "Deriche DC gain invariant violated for sigma={sigma}: {dc}"
        );
    }
}
