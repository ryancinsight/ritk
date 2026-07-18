use super::*;
use coeus_core::SequentialBackend;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support::make_image;

type LegacyBackend = SequentialBackend;

fn assert_native_case<A: AutoThreshold>(
    algorithm: &A,
    values: Vec<f32>,
    dimensions: [usize; 3],
) -> (f32, Vec<f32>) {
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let native = NativeImage::from_flat_on(
        values.clone(),
        dimensions,
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let legacy = make_image::<f32, LegacyBackend, 3>(values, dimensions);

    let native_threshold = algorithm
        .compute_native(&native)
        .expect("native threshold computation succeeds");
    let (native_mask, fused_threshold) = algorithm
        .apply_native_with_threshold(&native, &SequentialBackend)
        .expect("native threshold application succeeds");
    let native_mask_only = algorithm
        .apply_native(&native, &SequentialBackend)
        .expect("native mask-only application succeeds");
    let legacy_threshold = algorithm.compute(&legacy);
    let legacy_mask = algorithm.apply(&legacy);

    assert_eq!(native_threshold, legacy_threshold);
    assert_eq!(fused_threshold, legacy_threshold);
    assert_eq!(native_mask.shape(), dimensions);
    assert_eq!(*native_mask.origin(), origin);
    assert_eq!(*native_mask.spacing(), spacing);
    assert_eq!(*native_mask.direction(), direction);
    assert_eq!(
        native_mask.data_slice().expect("contiguous native mask"),
        legacy_mask.data_slice().as_ref()
    );
    assert_eq!(
        native_mask_only
            .data_slice()
            .expect("contiguous native mask"),
        native_mask.data_slice().expect("contiguous native mask")
    );
    (
        native_threshold,
        native_mask
            .data_slice()
            .expect("contiguous native mask")
            .to_vec(),
    )
}

fn assert_algorithm_suite<A: AutoThreshold>(algorithm: &A) {
    assert_eq!(threshold_from_slice(algorithm, &[]), 0.0);

    let bimodal: Vec<f32> = (0..64)
        .map(|index| if index < 32 { 20.0 } else { 200.0 })
        .collect();
    let _ = assert_native_case(algorithm, bimodal, [4, 4, 4]);

    let (constant_threshold, constant_mask) =
        assert_native_case(algorithm, vec![5.0; 8], [2, 2, 2]);
    assert_eq!(constant_threshold, 5.0);
    assert_eq!(constant_mask, vec![1.0; 8]);

    let (singleton_threshold, singleton_mask) = assert_native_case(algorithm, vec![7.0], [1, 1, 1]);
    assert_eq!(singleton_threshold, 7.0);
    assert_eq!(singleton_mask, [1.0]);

    let (_, mixed_mask) = assert_native_case(
        algorithm,
        vec![f32::NAN, f32::NEG_INFINITY, 1.0, 2.0, f32::INFINITY, 3.0],
        [1, 2, 3],
    );
    assert_eq!(mixed_mask[0], 0.0);
    assert_eq!(mixed_mask[1], 0.0);
    assert_eq!(mixed_mask[4], 0.0);

    let (nonfinite_threshold, nonfinite_mask) = assert_native_case(
        algorithm,
        vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
        [1, 1, 3],
    );
    assert_eq!(nonfinite_threshold, 0.0);
    assert_eq!(nonfinite_mask, [0.0; 3]);
}

#[test]
fn every_sealed_algorithm_inherits_exact_native_conformance() {
    assert_algorithm_suite(&OtsuThreshold::new());
    assert_algorithm_suite(&LiThreshold::new());
    assert_algorithm_suite(&YenThreshold::new());
    assert_algorithm_suite(&KapurThreshold::new());
    assert_algorithm_suite(&TriangleThreshold::new());
    assert_algorithm_suite(&super::super::isodata::IsoDataThreshold::new());
    assert_algorithm_suite(&super::super::moments::MomentsThreshold::new());
    assert_algorithm_suite(&super::super::huang::HuangThreshold::new());
    assert_algorithm_suite(&super::super::intermodes::IntermodesThreshold::new());
    assert_algorithm_suite(&super::super::shanbhag::ShanbhagThreshold::new());
    assert_algorithm_suite(&super::super::kittler::KittlerIllingworthThreshold::new());
    assert_algorithm_suite(&super::super::renyi::RenyiEntropyThreshold::new());
}
