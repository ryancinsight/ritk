use ritk_codecs::{SampleBuffer, SampleType};
use ritk_image::ImageMetadata;
use ritk_spatial::{CoordinateMap, Direction, Point, SliceSeries, SliceTransform, Spacing};

use super::{StoredVolume, VolumeError};
use crate::IntensityCalibration;

#[test]
fn stored_volume_retains_samples_and_geometry() {
    let samples = SampleBuffer::from_samples(vec![u64::MAX, 16_777_217]);
    let expected_bytes = samples
        .encode(ritk_codecs::ByteOrder::LeastSignificantByteFirst)
        .expect("sample encoding");
    let metadata = ImageMetadata::default_for_shape([1, 1, 2]);
    let volume = StoredVolume::new(
        [1, 1, 2],
        samples,
        metadata,
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )
    .expect("valid stored volume");

    assert_eq!(volume.shape(), [1, 1, 2]);
    assert_eq!(volume.samples().sample_type(), SampleType::U64);
    assert_eq!(volume.samples().len(), 2);
    assert_eq!(
        volume
            .samples()
            .encode(ritk_codecs::ByteOrder::LeastSignificantByteFirst)
            .expect("retained sample encoding"),
        expected_bytes
    );
    assert_eq!(volume.calibration(), &IntensityCalibration::Identity);
}

#[test]
fn stored_volume_rejects_shape_and_sample_mismatch() {
    let make_samples = || SampleBuffer::from_samples(vec![7_u16]);
    let metadata = || ImageMetadata::default_for_shape([1, 1, 1]);
    assert!(matches!(
        StoredVolume::new(
            [1, 0, 1],
            make_samples(),
            metadata(),
            CoordinateMap::Cartesian,
            IntensityCalibration::Identity
        ),
        Err(VolumeError::EmptyAxis { axis: 1 })
    ));
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 2],
            make_samples(),
            metadata(),
            CoordinateMap::Cartesian,
            IntensityCalibration::Identity
        ),
        Err(VolumeError::SampleCountMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn stored_volume_rejects_overflowing_shape_and_bad_calibration() {
    let samples = SampleBuffer::from_samples(Vec::<u8>::new());
    assert!(matches!(
        StoredVolume::new(
            [usize::MAX, 2, 1],
            samples,
            ImageMetadata::default_for_shape([usize::MAX, 2, 1]),
            CoordinateMap::Cartesian,
            IntensityCalibration::Identity
        ),
        Err(VolumeError::ShapeProductOverflow { .. })
    ));

    let calibration = IntensityCalibration::PerFrameLinear(Box::from([]));
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 1],
            SampleBuffer::from_samples(vec![7_u16]),
            ImageMetadata::default_for_shape([1, 1, 1]),
            CoordinateMap::Cartesian,
            calibration
        ),
        Err(VolumeError::CalibrationShape(_))
    ));
}

#[test]
fn stored_volume_rejects_invalid_affine_metadata() {
    let valid_samples = || SampleBuffer::from_samples(vec![1_u16]);
    let valid_map = || CoordinateMap::Cartesian;
    let invalid_origin = ImageMetadata::new(
        Point::new([0.0, f64::NAN, 0.0]),
        Spacing::uniform(1.0),
        Direction::identity(),
    );
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 1],
            valid_samples(),
            invalid_origin,
            valid_map(),
            IntensityCalibration::Identity,
        ),
        Err(VolumeError::NonFiniteOrigin { axis: 1 })
    ));

    let mut invalid_spacing = Spacing::uniform(1.0);
    invalid_spacing[2] = f64::INFINITY;
    let invalid_spacing =
        ImageMetadata::new(Point::origin(), invalid_spacing, Direction::identity());
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 1],
            valid_samples(),
            invalid_spacing,
            valid_map(),
            IntensityCalibration::Identity,
        ),
        Err(VolumeError::InvalidSpacing { axis: 2 })
    ));

    let invalid_direction =
        ImageMetadata::new(Point::origin(), Spacing::uniform(1.0), Direction::zeros());
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 1],
            valid_samples(),
            invalid_direction,
            valid_map(),
            IntensityCalibration::Identity,
        ),
        Err(VolumeError::SingularDirection)
    ));

    let invalid_direction = ImageMetadata::new(
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::from_rows([[1.0, f64::NAN, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 1],
            valid_samples(),
            invalid_direction,
            valid_map(),
            IntensityCalibration::Identity,
        ),
        Err(VolumeError::NonFiniteDirection { row: 0, column: 1 })
    ));

    let mut overflowing_spacing = Spacing::uniform(1.0);
    overflowing_spacing[0] = f64::MAX;
    let overflowing_axis = ImageMetadata::new(
        Point::origin(),
        overflowing_spacing,
        Direction::from_rows([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 1],
            valid_samples(),
            overflowing_axis,
            valid_map(),
            IntensityCalibration::Identity,
        ),
        Err(VolumeError::UnrepresentablePhysicalAxis { row: 0, column: 0 })
    ));

    let mut underflowing_spacing = Spacing::uniform(1.0);
    underflowing_spacing[0] = 1e-300;
    let underflowing_axis = ImageMetadata::new(
        Point::origin(),
        underflowing_spacing,
        Direction::from_rows([[1e-100, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 1],
            valid_samples(),
            underflowing_axis,
            valid_map(),
            IntensityCalibration::Identity,
        ),
        Err(VolumeError::UnrepresentablePhysicalAxis { row: 0, column: 0 })
    ));
}

#[test]
fn stored_volume_rejects_unrepresentable_physical_axis_geometry() {
    let samples = || SampleBuffer::from_samples(vec![1_u8]);
    let spacing = Spacing::new([f64::MAX, 1.0, 1.0]);
    let overflowing_length = ImageMetadata::new(
        Point::origin(),
        spacing,
        Direction::from_rows([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 1],
            samples(),
            overflowing_length,
            CoordinateMap::Cartesian,
            IntensityCalibration::Identity,
        ),
        Err(VolumeError::UnrepresentablePhysicalAxisNorm { column: 0 })
    ));

    let lost_direction_component = ImageMetadata::new(
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::from_rows([[1e308, 0.0, 0.0], [1e-308, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    assert!(matches!(
        StoredVolume::new(
            [1, 1, 1],
            samples(),
            lost_direction_component,
            CoordinateMap::Cartesian,
            IntensityCalibration::Identity,
        ),
        Err(VolumeError::UnrepresentablePhysicalAxisDirection { row: 1, column: 0 })
    ));
}

#[test]
fn stored_volume_requires_one_transform_per_slice() {
    let direction = Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let map = || {
        CoordinateMap::SliceSeries(
            SliceSeries::try_new(vec![SliceTransform::new(direction, [0.0, 0.0, 0.0])])
                .expect("one transform"),
        )
    };
    let samples = || SampleBuffer::from_samples(vec![1_u16, 2]);
    assert!(matches!(
        StoredVolume::new(
            [2, 1, 1],
            samples(),
            ImageMetadata::default_for_shape([2, 1, 1]),
            map(),
            IntensityCalibration::Identity,
        ),
        Err(VolumeError::CoordinateMapSliceCountMismatch {
            expected: 2,
            actual: 1
        })
    ));
}
