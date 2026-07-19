use super::*;
use coeus_core::MoiraiBackend;
use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

type B = MoiraiBackend;

fn make_test_image(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    let device = Default::default();
    let t = Tensor::<f32, B>::from_slice_on(shape, &data, &device);
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

// ── extract_vec ────────────────────────────────────────────────────────

/// Round-trip: extract_vec then rebuild must reproduce the original image.
///
/// # Derivation
/// extract_vec(I) = (v, d)  and  rebuild(v, d, I) = I' must satisfy:
///   ∀ i: I'(i) = I(i)    (element-wise equality within f32 precision)
///   shape(I') = shape(I)
#[test]
fn extract_and_rebuild_roundtrip() {
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
    let img = make_test_image(data.clone(), [2, 3, 4]);

    let (vals, dims) = extract_vec(&img).unwrap();
    assert_eq!(vals, data, "extracted values must equal original data");
    assert_eq!(dims, [2, 3, 4], "extracted dims must match image shape");

    let rebuilt = rebuild(vals, dims, &img);
    let got = rebuilt.data().to_vec();
    assert_eq!(got, data, "rebuilt image must reproduce original data");
}

/// Spatial metadata is preserved through extract → rebuild.
#[test]
fn rebuild_preserves_metadata() {
    let sp = Spacing::new([2.5, 1.0, 0.5]);
    let orig = Point::new([10.0, 20.0, 30.0]);
    let device = Default::default();
    let t = Tensor::<f32, B>::from_slice_on([1usize, 2, 3], &[1.0_f32; 6], &device);
    let img = Image::new(t, orig, sp, Direction::identity())
        .expect("invariant: fixture tensor has the declared rank");

    let (vals, dims) = extract_vec(&img).unwrap();
    let rebuilt = rebuild(vals, dims, &img);

    assert_eq!(rebuilt.origin(), img.origin(), "origin must be preserved");
    assert_eq!(
        rebuilt.spacing(),
        img.spacing(),
        "spacing must be preserved"
    );
}

/// extract_vec_infallible produces the same result as extract_vec.
#[test]
fn extract_vec_infallible_matches_fallible() {
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let img = make_test_image(data.clone(), [2, 2, 2]);

    let (v1, d1) = extract_vec(&img).unwrap();
    let (v2, d2) = extract_vec_infallible(&img);

    assert_eq!(v1, v2, "infallible variant must return same values");
    assert_eq!(d1, d2, "infallible variant must return same dims");
}

/// rebuild constructs an image with the expected shape.
#[test]
fn rebuild_output_has_correct_shape() {
    let data = vec![1.0_f32; 3 * 4 * 5];
    let img = make_test_image(data.clone(), [3, 4, 5]);
    let (vals, dims) = extract_vec(&img).unwrap();
    let out = rebuild(vals, dims, &img);
    assert_eq!(
        out.shape(),
        img.shape(),
        "rebuilt shape must match source shape"
    );
}

// ── gaussian_kernel ────────────────────────────────────────────────

/// Kernel sums to 1.0 (wide-precision f64 variant).
#[test]
fn gaussian_kernel_sums_to_one_wide_precision() {
    let kernel = gaussian_kernel(2.0_f64, None);
    let sum: f64 = kernel.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12, "kernel sum = {sum}");
}

/// Kernel sums to 1.0 (single-precision f32 variant).
#[test]
fn gaussian_kernel_sums_to_one_single_precision() {
    let kernel = gaussian_kernel(2.0_f32, None);
    let sum: f32 = kernel.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "kernel sum = {sum}");
}

/// Zero sigma returns identity kernel.
#[test]
fn gaussian_kernel_zero_sigma_is_identity() {
    let kernel = gaussian_kernel(0.0_f64, None);
    assert_eq!(kernel, vec![1.0_f64]);
}

/// Explicit radius overrides the default.
#[test]
fn gaussian_kernel_explicit_radius() {
    let kernel = gaussian_kernel(1.0_f64, Some(5));
    assert_eq!(kernel.len(), 11); // 2 * 5 + 1
}

/// Centre-to-adjacent ratio verifies the exponent denominator is exactly 2σ².
///
/// # Derivation
/// For d=1 from centre: w₁/w₀ = exp(-1 / (2σ²)).
/// With σ=2.0: expected = exp(-1/8) ≈ 0.882497.
/// The previous defect (`1 + σ²` = 5) would produce exp(-1/5) ≈ 0.818731 — a ~7% error.
#[test]
fn gaussian_kernel_exponent_denominator_is_two_sigma_squared() {
    let sigma = 2.0_f64;
    let kernel = gaussian_kernel(sigma, Some(4));
    let centre = 4_usize; // r = 4, centre = index 4
    let expected_ratio = (-1.0_f64 / (2.0 * sigma * sigma)).exp();
    let actual_ratio = kernel[centre - 1] / kernel[centre];
    assert!(
        (actual_ratio - expected_ratio).abs() < 1e-12,
        "ratio kernel[r-1]/kernel[r] = {actual_ratio:.9}, expected exp(-1/(2σ²)) = {expected_ratio:.9}"
    );
}

/// Kernel is symmetric.
#[test]
fn gaussian_kernel_is_symmetric() {
    let kernel = gaussian_kernel(2.0_f64, None);
    let n = kernel.len();
    for i in 0..n {
        assert!(
            (kernel[i] - kernel[n - 1 - i]).abs() < 1e-15,
            "asymmetry at i={i}: {} vs {}",
            kernel[i],
            kernel[n - 1 - i]
        );
    }
}

/// Peak is at the centre.
#[test]
fn gaussian_kernel_peak_at_centre() {
    let kernel = gaussian_kernel(1.0_f64, Some(3));
    let center = kernel.len() / 2;
    for (i, &w) in kernel.iter().enumerate() {
        if i != center {
            assert!(
                kernel[center] >= w,
                "centre {} < kernel[{i}] = {w}",
                kernel[center]
            );
        }
    }
}
