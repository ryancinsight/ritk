use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support::make_image;

use super::{BinaryFillHoles, MorphologicalGradient, MorphologicalOperation, Skeletonization};

type LegacyBackend = NdArray<f32>;

fn native_image(values: Vec<f32>) -> NativeImage<f32, SequentialBackend, 3> {
    NativeImage::from_flat_on(
        values,
        [3, 3, 3],
        Point::new([2.0, 3.0, 5.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image")
}

fn assert_exact(
    source: &NativeImage<f32, SequentialBackend, 3>,
    native: &NativeImage<f32, SequentialBackend, 3>,
    legacy: &[f32],
) {
    assert_eq!(native.data_slice().expect("contiguous output"), legacy);
    assert_eq!(native.origin(), source.origin());
    assert_eq!(native.spacing(), source.spacing());
    assert_eq!(native.direction(), source.direction());
}

#[test]
fn filter_owned_native_postprocessing_matches_legacy_exactly() {
    let mut values = vec![0.0; 27];
    for value in &mut values[9..18] {
        *value = 1.0;
    }
    values[13] = 0.0;
    let native = native_image(values.clone());
    let legacy = make_image::<LegacyBackend, 3>(values, [3, 3, 3]);

    let native_fill = BinaryFillHoles
        .apply_native(&native, &SequentialBackend)
        .expect("native fill holes succeeds");
    let legacy_fill = BinaryFillHoles.apply(&legacy);
    assert_exact(&native, &native_fill, legacy_fill.data_slice().as_ref());

    let gradient = MorphologicalGradient::new(1);
    let native_gradient = gradient
        .apply_native(&native, &SequentialBackend)
        .expect("native gradient succeeds");
    let legacy_gradient = gradient.apply(&legacy);
    assert_exact(
        &native,
        &native_gradient,
        legacy_gradient.data_slice().as_ref(),
    );
    assert_eq!(gradient.radius(), 1);

    let native_skeleton = Skeletonization::new()
        .apply_native(&native, &SequentialBackend)
        .expect("native skeletonization succeeds");
    let legacy_skeleton = Skeletonization::new().apply(&legacy);
    assert_exact(
        &native,
        &native_skeleton,
        legacy_skeleton.data_slice().as_ref(),
    );
}

#[test]
fn native_postprocessing_rejects_non_finite_masks() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let native = native_image(vec![value; 27]);
        assert!(BinaryFillHoles
            .apply_native(&native, &SequentialBackend)
            .is_err());
        assert!(MorphologicalGradient::new(1)
            .apply_native(&native, &SequentialBackend)
            .is_err());
        assert!(Skeletonization::new()
            .apply_native(&native, &SequentialBackend)
            .is_err());
    }
}

fn native_image_dim<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
) -> NativeImage<f32, SequentialBackend, D> {
    NativeImage::from_flat_on(
        values,
        shape,
        Point::new([0.0; D]),
        Spacing::new([1.0; D]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image")
}

#[test]
fn native_skeletonization_rejects_unsupported_dimensions() {
    let scalar = native_image_dim(vec![1.0], []);
    assert!(Skeletonization::new()
        .apply_native(&scalar, &SequentialBackend)
        .is_err());
    let rank_four = native_image_dim(vec![1.0], [1, 1, 1, 1]);
    assert!(Skeletonization::new()
        .apply_native(&rank_four, &SequentialBackend)
        .is_err());
}
