use super::*;
use coeus_core::SequentialBackend;
use ritk_image::tensor::Tensor;
use ritk_image::test_support as ts;
use ritk_image::Image as NativeImage;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = coeus_core::SequentialBackend;

fn make_image(vals: Vec<f32>, depth: usize, rows: usize, cols: usize) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, [depth, rows, cols])
}

fn image_vals(img: &Image<f32, B, 3>) -> Vec<f32> {
    img.data().to_vec()
}

/// Invariant: uniform input → output = input.
///
/// Proof: G_σ * constant = constant (Gaussian kernel sums to 1),
/// so mask = 0 everywhere, |mask| = 0 < threshold for any threshold ≥ 0,
/// output = input.
#[test]
fn uniform_input_is_identity() {
    let img = make_image(vec![3.0_f32; 2 * 4 * 4], 2, 4, 4);
    let filter = UnsharpMaskFilter::new(
        vec![GaussianSigma::new_unchecked(1.0)],
        2.0,
        0.0,
        ClampPolicy::ClampToInputRange,
    );
    let out = filter.apply::<B>(&img).expect("apply failed");
    let vals = image_vals(&out);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 3.0_f32).abs() < 1e-4,
            "voxel {i}: expected 3.0, got {v} (uniform input violated identity)"
        );
    }
}

/// Invariant: amount = 0 → output = input identically.
///
/// Proof: output = I + 0 · (...) = I for all p.
#[test]
fn amount_zero_is_exact_identity() {
    // Non-trivial image with gradient values.
    let vals: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let img = make_image(vals.clone(), 2, 4, 4);
    let filter = UnsharpMaskFilter::new(
        vec![GaussianSigma::new_unchecked(1.0)],
        0.0,
        0.0,
        ClampPolicy::NoClamp,
    );
    let out = filter.apply::<B>(&img).expect("apply failed");
    let out_vals = image_vals(&out);
    for (i, (&expected, &got)) in vals.iter().zip(out_vals.iter()).enumerate() {
        assert!(
            (expected - got).abs() < 1e-5,
            "voxel {i}: expected {expected}, got {got} (amount=0 identity violated)"
        );
    }
}

/// Invariant: threshold > all |mask| values → output = input.
///
/// Construction: use a constant image (mask = 0 everywhere) with threshold = 100.0;
/// since |mask| = 0 < 100.0 for all voxels, sharpening is never triggered.
#[test]
fn threshold_suppresses_all_sharpening() {
    // Constant image → mask = 0 < threshold = 100.0 everywhere.
    let img = make_image(vec![42.0_f32; 2 * 3 * 3], 2, 3, 3);
    let filter = UnsharpMaskFilter::new(
        vec![GaussianSigma::new_unchecked(1.0)],
        5.0,
        100.0,
        ClampPolicy::NoClamp,
    );
    let out = filter.apply::<B>(&img).expect("apply failed");
    let out_vals = image_vals(&out);
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            (v - 42.0_f32).abs() < 1e-4,
            "voxel {i}: expected 42.0, got {v} (threshold suppression violated)"
        );
    }
}

/// Invariant: clamp=true → output(p) ≤ max(input) for all p.
///
/// Construction: step edge [0, 0, ..., 1, 1, ...] with large amount (5.0);
/// without clamping, edge voxels would exceed 1.0. Clamping enforces ≤ 1.0.
#[test]
fn clamp_enforces_upper_bound() {
    // 1×1×8 step edge: [0,0,0,0,1,1,1,1]
    let vals: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let img = make_image(vals.clone(), 1, 1, 8);
    // Large amount to ensure edge overshoot without clamping.
    let filter = UnsharpMaskFilter::new(
        vec![GaussianSigma::new_unchecked(0.5)],
        5.0,
        0.0,
        ClampPolicy::ClampToInputRange,
    );
    let out = filter.apply::<B>(&img).expect("apply failed");
    let out_vals = image_vals(&out);
    let input_max = 1.0_f32;
    let input_min = 0.0_f32;
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            v <= input_max + 1e-5,
            "voxel {i}: clamp=true violated upper bound: {v} > {input_max}"
        );
        assert!(
            v >= input_min - 1e-5,
            "voxel {i}: clamp=true violated lower bound: {v} < {input_min}"
        );
    }
}

/// Invariant: clamp=false → sharpening can produce values outside [min(I), max(I)].
///
/// Construction: same step edge with large amount; at least one edge voxel must
/// exceed 1.0 (the input maximum), proving the unsharp mask is genuinely applied.
#[test]
fn no_clamp_allows_overshoot() {
    // 1×1×8 step edge: [0,0,0,0,1,1,1,1]
    let vals: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let img = make_image(vals.clone(), 1, 1, 8);
    let filter = UnsharpMaskFilter::new(
        vec![GaussianSigma::new_unchecked(0.5)],
        5.0,
        0.0,
        ClampPolicy::NoClamp,
    );
    let out = filter.apply::<B>(&img).expect("apply failed");
    let out_vals = image_vals(&out);
    // At the step boundary (positions 4–5), the sharpened output must exceed 1.0.
    let any_above_max = out_vals.iter().any(|&v| v > 1.0 + 1e-5);
    assert!(
        any_above_max,
        "no_clamp_allows_overshoot: expected at least one voxel > 1.0 at step edge, \
         got max = {:.6}. The unsharp mask formula is not applying sharpening.",
        out_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
}

/// Invariant: spatial metadata (origin, spacing, direction) is preserved.
#[test]
fn spatial_metadata_preserved() {
    let tensor = Tensor::<f32, B>::from_slice([2, 3, 3], &[1.0_f32; 2 * 3 * 3]);
    let origin = Point::new([10.0, 20.0, 30.0]);
    let spacing = Spacing::new([1.5, 2.0, 0.75]);
    let dir = Direction::<3>::identity();
    let img = Image::new(tensor, origin, spacing, dir)
        .expect("invariant: fixture tensor has the declared rank");
    let filter = UnsharpMaskFilter::default();
    let out = filter.apply::<B>(&img).expect("apply failed");
    assert_eq!(out.origin(), img.origin(), "origin changed");
    assert_eq!(out.spacing(), img.spacing(), "spacing changed");
    assert_eq!(out.direction(), img.direction(), "direction changed");
}

/// Invariant: sharpening genuinely increases contrast near step edges.
///
/// For a step edge [0.0, 1.0] embedded in a 1×1×4 image, after sharpening with
/// threshold=0 and amount>0, the difference between the edge voxels must be
/// strictly greater than in the input.
#[test]
fn sharpening_increases_edge_contrast() {
    // 1×1×4 step: [0, 0, 1, 1]
    let vals: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0];
    let img = make_image(vals, 1, 1, 4);
    // amount=2.0, no clamping so we can observe the actual sharpened values.
    let filter = UnsharpMaskFilter::new(
        vec![GaussianSigma::new_unchecked(0.5)],
        2.0,
        0.0,
        ClampPolicy::NoClamp,
    );
    let out = filter.apply::<B>(&img).expect("apply failed");
    let out_vals = image_vals(&out);
    // The output step contrast (max − min) must be > input contrast (1.0 − 0.0 = 1.0).
    let out_min = out_vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let out_max = out_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let out_contrast = out_max - out_min;
    assert!(
        out_contrast > 1.0,
        "sharpening did not increase edge contrast: \
         output contrast = {out_contrast:.4} (expected > 1.0, input contrast = 1.0)"
    );
}

/// The native boundary shares the Deriche value kernel, so a constant volume
/// has a zero high-frequency mask and remains unchanged exactly.
#[test]
fn native_uniform_input_is_exact_identity() {
    let input = NativeImage::from_flat_on(
        vec![7.25_f32; 4],
        [1, 2, 2],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = UnsharpMaskFilter::new(
        vec![GaussianSigma::new_unchecked(1.0)],
        2.0,
        0.0,
        ClampPolicy::NoClamp,
    )
    .apply_native(&input, &SequentialBackend)
    .expect("native unsharp mask succeeds");

    assert_eq!(
        output
            .data_slice()
            .expect("invariant: sequential storage is contiguous"),
        &[7.25_f32; 4]
    );
    assert_eq!(output.origin(), input.origin());
    assert_eq!(output.spacing(), input.spacing());
    assert_eq!(output.direction(), input.direction());
}
