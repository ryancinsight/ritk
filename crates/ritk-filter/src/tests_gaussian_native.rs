//! Native (coeus) Gaussian filter tests.
//! Extracted from gaussian.rs to keep that file under 500 lines.
//! (Sprint 350: DRY/SRP file-size discipline)

use super::GaussianFilter;
use ritk_image::native::Image as NativeImage;
use coeus_core::SequentialBackend;
use ritk_spatial::{Direction, Point, Spacing};
use crate::edge::GaussianSigma;

type B = SequentialBackend;

fn make_native_image(data: Vec<f32>, shape: [usize; 3]) -> NativeImage<f32, B, 3> {
    let backend = B::default();
    NativeImage::from_flat_on(
        data,
        shape,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &backend,
    )
    .expect("valid image")
}

#[test]
fn gaussian_apply_native_preserves_shape() {
    let data: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = make_native_image(data, [3, 3, 3]);
    let filter = GaussianFilter::new(vec![
        GaussianSigma::new_unchecked(1.0),
        GaussianSigma::new_unchecked(1.0),
        GaussianSigma::new_unchecked(1.0),
    ]);
    let result = filter.apply_native(&img, &B::default()::default()).unwrap();
    assert_eq!(result.shape(), [3, 3, 3]);
}

#[test]
fn gaussian_apply_native_zero_sigma_is_identity() {
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let img = make_native_image(data.clone(), [2, 2, 2]);
    let filter = GaussianFilter::new_isotropic(0.0);
    let result = filter.apply_native(&img, &B::default()::default()).unwrap();
    let out_vals = result.data_slice().expect("contiguous");
    assert_eq!(out_vals.as_ref(), data.as_slice(), "zero-sigma must be identity");
}
