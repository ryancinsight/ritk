use super::*;
use crate::native_support::LegacyBurnBackend;
use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, shape)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

// ── generate_kernel ───────────────────────────────────────────────────────

/// The Gaussian kernel normalizes correctly: interior voxels of a constant image
/// are preserved under convolution.
///
/// # Derivation
/// The kernel weights sum to 1.0 by construction. For interior voxels where the
/// full kernel fits within the image, the convolution output equals the constant:
///   out(x) = Σ w_k * C = C * Σ w_k = C * 1.0 = C.
///
/// Boundary voxels receive partial kernel support under zero-padding and may
/// deviate. We test only the center voxel of a large image (size=15) where the
/// radius-3 kernel (sigma=1, radius=ceil(3*1)=3) fits fully.
#[test]
fn gaussian_kernel_sums_to_one() {
    let size = 15usize;
    let filter = GaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(1.0)]);
    let img = make_image(vec![3.0_f32; size * size * size], [size, size, size]);
    let out = filter.apply(&img);
    let vals = voxels(&out);
    // Center voxel index: (size/2) * size * size + (size/2) * size + (size/2)
    let cx = size / 2;
    let center_idx = cx * size * size + cx * size + cx;
    let v = vals[center_idx];
    assert!(
        (v - 3.0).abs() < 5e-3,
        "center voxel of constant image under Gaussian must stay ≈ 3.0; got {v}"
    );
}

/// A z=1 (2-D promoted) image must not be darkened: the degenerate z-axis is
/// skipped, so an in-plane constant stays constant rather than being scaled by
/// the Gaussian kernel's centre weight (≈0.2) from convolving length-1 z with
/// zero padding.
#[test]
fn z1_image_constant_preserved() {
    let size = 15usize;
    let filter = GaussianFilter::<B>::new(vec![
        GaussianSigma::new_unchecked(2.0),
        GaussianSigma::new_unchecked(2.0),
        GaussianSigma::new_unchecked(2.0),
    ]);
    let img = make_image(vec![50.0_f32; size * size], [1, size, size]);
    let out = filter.apply(&img);
    let vals = voxels(&out);
    let cx = size / 2;
    let center = cx * size + cx;
    assert!(
        (vals[center] - 50.0).abs() < 0.5,
        "z=1 constant must stay ≈ 50.0 (degenerate z-axis skipped); got {}",
        vals[center]
    );
}

/// Zero sigma must skip smoothing (output identical to input).
///
/// # Derivation
/// The implementation has `if sigma <= 1e-6 { continue; }` which bypasses the
/// convolution entirely. The output tensor must be identical to the input.
#[test]
fn zero_sigma_skips_smoothing() {
    let filter = GaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(1e-9)]);
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let img = make_image(data.clone(), [2, 2, 2]);
    let out = filter.apply(&img);
    let got = voxels(&out);
    for (i, (&a, &b)) in got.iter().zip(data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "zero sigma must not change voxel {i}: expected {b}, got {a}"
        );
    }
}

/// Spatial metadata (origin, spacing, direction) is preserved.
#[test]
fn gaussian_preserves_metadata() {
    let filter = GaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(0.5)]);
    let sp = Spacing::new([2.0, 3.0, 4.0]);
    let device = Default::default();
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 2 * 2 * 2], Shape::new([2usize, 2, 2])),
        &device,
    );
    let img = Image::new(t, Point::new([10.0, 20.0, 30.0]), sp, Direction::identity());
    let out = filter.apply(&img);
    assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    assert_eq!(out.origin(), img.origin(), "origin must be preserved");
}

/// Output shape must equal input shape after smoothing (padding=kernel_size/2).
#[test]
fn gaussian_preserves_shape() {
    let filter = GaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(1.5)]);
    let img = make_image(vec![1.0_f32; 5 * 6 * 7], [5, 6, 7]);
    let out = filter.apply(&img);
    assert_eq!(
        out.shape(),
        img.shape(),
        "shape must be preserved after Gaussian"
    );
}

#[test]
fn native_gaussian_matches_tensor_backed_path() {
    let shape = [5, 4, 3];
    let values: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|index| index as f32 * 0.25 - 3.0)
        .collect();
    let filter = GaussianFilter::<B>::new(vec![GaussianSigma::new_unchecked(1.0); 3]);
    let legacy = filter.apply(&make_image(values.clone(), shape));
    let native = NativeImage::from_flat_on(
        values,
        shape,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let native = filter
        .apply_native(&native, &SequentialBackend)
        .expect("native Gaussian succeeds");

    for (index, (&expected, &actual)) in voxels(&legacy)
        .iter()
        .zip(
            native
                .data_slice()
                .expect("invariant: sequential storage is contiguous"),
        )
        .enumerate()
    {
        assert!(
            (expected - actual).abs() <= 2.0e-5,
            "native zero-padded convolution diverged at {index}: expected {expected}, got {actual}"
        );
    }
}
