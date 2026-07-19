use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::test_support::make_image;
use ritk_image::Image as NativeImage;

use super::{ConnectedComponentsFilter, Connectivity};
use crate::RelabelComponentFilter;

type LegacyBackend = SequentialBackend;

fn native_image(values: Vec<f32>) -> NativeImage<f32, SequentialBackend, 3> {
    NativeImage::from_flat_on(
        values,
        [2, 2, 2],
        Point::new([2.0, 3.0, 5.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image")
}

#[test]
fn filter_owned_native_labeling_matches_legacy_exactly() {
    let values = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0];
    let native = native_image(values.clone());
    let legacy = make_image::<f32, LegacyBackend, 3>(values, [2, 2, 2]);
    let filter = ConnectedComponentsFilter::with_connectivity(Connectivity::Six);
    let (native_labels, native_statistics) = filter
        .apply_native(&native, &SequentialBackend)
        .expect("native connected components succeeds");
    let (legacy_labels, legacy_statistics) = filter.apply(&legacy);

    assert_eq!(
        native_labels.data_slice().expect("contiguous labels"),
        legacy_labels
            .data_slice()
            .expect("invariant: contiguous host storage")
    );
    assert_eq!(native_statistics, legacy_statistics);
    assert_eq!(native_labels.origin(), native.origin());
    assert_eq!(native_labels.spacing(), native.spacing());
    assert_eq!(native_labels.direction(), native.direction());
    assert_eq!(filter.connectivity(), Connectivity::Six);
    assert_eq!(filter.background_value(), 0.0);
}

#[test]
fn filter_owned_native_relabeling_matches_legacy_exactly() {
    let values = vec![1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 3.0, 0.0];
    let native = native_image(values.clone());
    let legacy = make_image::<f32, LegacyBackend, 3>(values, [2, 2, 2]);
    let filter = RelabelComponentFilter::with_minimum_object_size(2);
    let (native_labels, native_statistics) = filter
        .apply_native(&native, &SequentialBackend)
        .expect("native relabel components succeeds");
    let (legacy_labels, legacy_statistics) = filter.apply(&legacy).expect("valid legacy labels");

    assert_eq!(
        native_labels.data_slice().expect("contiguous labels"),
        legacy_labels
            .data_slice()
            .expect("invariant: contiguous host storage")
    );
    assert_eq!(native_statistics, legacy_statistics);
    assert_eq!(native_labels.origin(), native.origin());
    assert_eq!(native_labels.spacing(), native.spacing());
    assert_eq!(native_labels.direction(), native.direction());
    assert_eq!(filter.minimum_object_size(), 2);
}

#[test]
fn native_relabel_rejects_invalid_label_values_without_allocating() {
    for value in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        1.5,
        4_294_967_296.0,
    ] {
        let image = native_image(vec![value; 8]);
        assert!(
            RelabelComponentFilter::new()
                .apply_native(&image, &SequentialBackend)
                .is_err(),
            "invalid label {value} must be rejected"
        );
    }
}

#[test]
fn native_relabel_handles_sparse_maximum_exact_label() {
    let image = native_image(vec![0.0, 16_777_216.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let (output, statistics) = RelabelComponentFilter::new()
        .apply_native(&image, &SequentialBackend)
        .expect("maximum exact sparse label is valid");
    assert_eq!(
        output.data_slice().expect("contiguous labels"),
        &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(statistics[0].original_label, 16_777_216);
    assert_eq!(statistics[0].new_label, 1);
    assert_eq!(statistics[0].voxel_count, 1);
}

#[test]
fn connected_components_pins_non_finite_mask_semantics() {
    let image = native_image(vec![
        0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    let (output, statistics) = ConnectedComponentsFilter::new()
        .apply_native(&image, &SequentialBackend)
        .expect("finite background with non-finite foreground succeeds");
    assert_eq!(
        output.data_slice().expect("contiguous labels"),
        &[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(statistics.len(), 1);
    assert_eq!(statistics[0].voxel_count, 3);
    assert!(ConnectedComponentsFilter::new()
        .with_background(f32::NAN)
        .is_err());
}
