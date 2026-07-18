use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing, VoxelIndex};
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support::make_image;

use super::{ConfidenceConnectedFilter, ConnectedThresholdFilter, NeighborhoodConnectedFilter};

type LegacyBackend = SequentialBackend;

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

fn assert_native_matches_legacy(
    native: &NativeImage<f32, SequentialBackend, 3>,
    native_output: &NativeImage<f32, SequentialBackend, 3>,
    legacy_values: &[f32],
) {
    assert_eq!(native_output.shape(), native.shape());
    assert_eq!(native_output.origin(), native.origin());
    assert_eq!(native_output.spacing(), native.spacing());
    assert_eq!(native_output.direction(), native.direction());
    assert_eq!(
        native_output.data_slice().expect("contiguous native mask"),
        legacy_values
    );
}

#[test]
fn filter_owned_native_region_growing_matches_legacy_boundaries_exactly() {
    let values: Vec<f32> = (0..27).map(|index| index as f32).collect();
    let native = native_image(values.clone());
    let legacy = make_image::<f32, LegacyBackend, 3>(values, [3, 3, 3]);
    let seed = VoxelIndex::from([1, 1, 1]);

    let connected = ConnectedThresholdFilter::new(seed, 8.0, 18.0);
    let connected_native = connected
        .apply_native(&native, &SequentialBackend)
        .expect("native connected threshold succeeds");
    assert_native_matches_legacy(
        &native,
        &connected_native,
        connected.apply(&legacy).data_slice().as_ref(),
    );
    assert_eq!(connected.seed(), seed);
    assert_eq!(connected.lower(), 8.0);
    assert_eq!(connected.upper(), 18.0);

    let confidence = ConfidenceConnectedFilter::new(seed, 8.0, 18.0)
        .with_multiplier(2.5)
        .expect("test multiplier is valid")
        .with_max_iterations(3);
    let confidence_native = confidence
        .apply_native(&native, &SequentialBackend)
        .expect("native confidence-connected succeeds");
    assert_native_matches_legacy(
        &native,
        &confidence_native,
        confidence.apply(&legacy).data_slice().as_ref(),
    );
    assert_eq!(confidence.seed(), seed);
    assert_eq!(confidence.initial_lower(), 8.0);
    assert_eq!(confidence.initial_upper(), 18.0);
    assert_eq!(confidence.multiplier(), 2.5);
    assert_eq!(confidence.max_iterations(), 3);

    let neighborhood = NeighborhoodConnectedFilter::new(seed, 8.0, 18.0).with_radius([0, 0, 0]);
    let neighborhood_native = neighborhood
        .apply_native(&native, &SequentialBackend)
        .expect("native neighborhood-connected succeeds");
    assert_native_matches_legacy(
        &native,
        &neighborhood_native,
        neighborhood.apply(&legacy).data_slice().as_ref(),
    );
    assert_eq!(neighborhood.seed(), seed);
    assert_eq!(neighborhood.lower(), 8.0);
    assert_eq!(neighborhood.upper(), 18.0);
    assert_eq!(neighborhood.radius(), [0, 0, 0]);
}

#[test]
fn native_region_growing_rejects_out_of_bounds_seeds() {
    let native = native_image(vec![0.0; 27]);
    let seed = [3, 0, 0];
    assert!(ConnectedThresholdFilter::new(seed, 0.0, 1.0)
        .apply_native(&native, &SequentialBackend)
        .is_err());
    assert!(ConfidenceConnectedFilter::new(seed, 0.0, 1.0)
        .apply_native(&native, &SequentialBackend)
        .is_err());
    assert!(NeighborhoodConnectedFilter::new(seed, 0.0, 1.0)
        .apply_native(&native, &SequentialBackend)
        .is_err());
}

#[test]
fn confidence_multiplier_rejects_nan() {
    assert!(ConfidenceConnectedFilter::new([0, 0, 0], 0.0, 1.0)
        .with_multiplier(f32::NAN)
        .is_err());
}

#[test]
fn non_finite_voxels_are_never_region_foreground() {
    let seed = [1, 1, 1];
    let mut values = vec![f32::NAN; 27];
    values[13] = 1.0;
    values[14] = f32::INFINITY;
    values[12] = f32::NEG_INFINITY;
    let native = native_image(values);
    let mut expected = vec![0.0; 27];
    expected[13] = 1.0;

    let connected = ConnectedThresholdFilter::new(seed, f32::NEG_INFINITY, f32::INFINITY)
        .apply_native(&native, &SequentialBackend)
        .expect("native connected threshold succeeds");
    assert_eq!(connected.data_slice().expect("contiguous mask"), expected);

    let confidence = ConfidenceConnectedFilter::new(seed, f32::NEG_INFINITY, f32::INFINITY)
        .apply_native(&native, &SequentialBackend)
        .expect("native confidence-connected succeeds");
    assert_eq!(confidence.data_slice().expect("contiguous mask"), expected);

    let neighborhood = NeighborhoodConnectedFilter::new(seed, f32::NEG_INFINITY, f32::INFINITY)
        .with_radius([0, 0, 0])
        .apply_native(&native, &SequentialBackend)
        .expect("native neighborhood-connected succeeds");
    assert_eq!(
        neighborhood.data_slice().expect("contiguous mask"),
        expected
    );
}

#[test]
fn non_finite_seed_produces_an_empty_region() {
    let native = native_image(vec![f32::NAN; 27]);
    let output = ConnectedThresholdFilter::new([1, 1, 1], f32::NEG_INFINITY, f32::INFINITY)
        .apply_native(&native, &SequentialBackend)
        .expect("native connected threshold succeeds");
    assert_eq!(output.data_slice().expect("contiguous mask"), vec![0.0; 27]);
}

#[test]
fn neighborhood_radius_saturates_at_image_boundaries() {
    let native = native_image(vec![1.0; 27]);
    let output = NeighborhoodConnectedFilter::new([1, 1, 1], 0.0, 2.0)
        .with_radius([usize::MAX; 3])
        .apply_native(&native, &SequentialBackend)
        .expect("native neighborhood-connected succeeds");
    assert_eq!(output.data_slice().expect("contiguous mask"), vec![1.0; 27]);
}
