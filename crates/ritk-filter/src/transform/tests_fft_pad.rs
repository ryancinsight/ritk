use super::*;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, shape)
}

fn voxels(img: &Image<f32, B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

// ── good-size arithmetic ───────────────────────────────────────────────────

/// Greatest prime factor matches first-principles factorization on known values.
#[test]
fn greatest_prime_factor_known() {
    assert_eq!(greatest_prime_factor(1), 1);
    assert_eq!(greatest_prime_factor(2), 2);
    assert_eq!(greatest_prime_factor(240), 5); // 2^4 * 3 * 5
    assert_eq!(greatest_prime_factor(230), 23); // 2 * 5 * 23
    assert_eq!(greatest_prime_factor(226), 113); // 2 * 113
    assert_eq!(greatest_prime_factor(243), 3); // 3^5
    assert_eq!(greatest_prime_factor(251), 251); // prime
}

/// Next 2-3-5-smooth size matches the sitk-verified targets (probed against
/// `itk::FFTPadImageFilter` with `SizeGreatestPrimeFactor = 5`).
#[test]
fn next_smooth_size_matches_itk() {
    let cases = [
        (230, 240),
        (226, 240),
        (199, 200),
        (127, 128),
        (251, 256),
        (243, 243), // already smooth
        (210, 216),
        (101, 108),
        (97, 100),
        (1, 1), // unit axis untouched
    ];
    for (n, want) in cases {
        assert_eq!(next_smooth_size(n, 5), want, "next_smooth_size({n})");
    }
}

/// Pad extents split symmetrically with the smaller half on the lower side, and
/// leave already-smooth axes unpadded.
#[test]
fn pad_extents_symmetric_split() {
    let f = FftPadImageFilter::default();
    // shape [z=1, y=226, x=230] -> [1, 240, 240]: low=(14/2,10/2)=(7,5), hi=(7,5).
    let (lo, hi) = f.pad_extents([1, 226, 230]);
    assert_eq!(*lo.as_array(), [0, 7, 5]);
    assert_eq!(*hi.as_array(), [0, 7, 5]);
    // odd total pad: 101 -> 108, total 7, low=3, high=4.
    let (lo2, hi2) = f.pad_extents([1, 1, 101]);
    assert_eq!(*lo2.as_array(), [0, 0, 3]);
    assert_eq!(*hi2.as_array(), [0, 0, 4]);
}

// ── apply correctness ──────────────────────────────────────────────────────

/// ZeroFluxNeumann (default) replicates edge voxels into the padded region and
/// enlarges X from 7 (gpf 7 > 5) to 8.
#[test]
fn apply_zero_flux_neumann_default() {
    // 1x1x7 ramp; 7 -> next smooth = 8, total pad 1 -> low 0, high 1.
    let img = make_image(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [1, 1, 7]);
    let out = FftPadImageFilter::default().apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 8]);
    // low pad 0, high pad 1 replicates the last edge voxel 7.0.
    assert_eq!(voxels(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0]);
}

/// Zero boundary fills the padded region with 0; symmetric split for total 3
/// (X: 11 -> 12, low 0 high 1; here use 13 -> 15? choose 11 -> 12).
#[test]
fn apply_zero_boundary_constant_fill() {
    // 1x1x11 ones; 11 (prime) -> 12, total pad 1 -> low 0, high 1, fill 0.
    let img = make_image(vec![1.0; 11], [1, 1, 11]);
    let out = FftPadImageFilter::new(5, FftPadBoundary::Zero)
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 12]);
    let v = voxels(&out);
    assert_eq!(v[..11], [1.0; 11]);
    assert_eq!(v[11], 0.0);
}

/// Periodic boundary wraps around; 13 (prime) -> 15, total pad 2 -> low 1 high 1.
#[test]
fn apply_periodic_boundary_wraps() {
    let data: Vec<f32> = (0..13).map(|i| i as f32).collect();
    let img = make_image(data, [1, 1, 13]);
    let out = FftPadImageFilter::new(5, FftPadBoundary::Periodic)
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 15]);
    let v = voxels(&out);
    // low pad 1 wraps the last voxel (12); high pad 1 wraps the first (0).
    assert_eq!(v[0], 12.0);
    assert_eq!(v[1], 0.0);
    assert_eq!(v[14], 0.0);
}

/// Already-smooth shapes are returned unchanged (no padding).
#[test]
fn apply_smooth_shape_unchanged() {
    let img = make_image(vec![9.0; 8], [1, 1, 8]); // 8 = 2^3, smooth
    let out = FftPadImageFilter::default().apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 8]);
    assert_eq!(voxels(&out), vec![9.0; 8]);
}
