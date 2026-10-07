use super::*;
use crate::header::NiftiHeader;
use ritk_codecs::{SampleBuffer, SampleType};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    ConversionLocation, ConversionLoss, IntensityCalibration, LinearCalibration, SeriesAxis,
    StoredSeries, StoredVolume,
};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
use tempfile::tempdir;

fn scalar_payloads() -> Vec<(SampleType, Vec<u8>)> {
    vec![
        (SampleType::U8, vec![0, 128, 255]),
        (
            SampleType::I8,
            [i8::MIN, 0, i8::MAX]
                .into_iter()
                .flat_map(i8::to_le_bytes)
                .collect(),
        ),
        (
            SampleType::U16,
            [0, 0x8000, u16::MAX]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
        (
            SampleType::I16,
            [i16::MIN, 0, i16::MAX]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect(),
        ),
        (
            SampleType::U32,
            [0, 0x8000_0000, u32::MAX]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        ),
        (
            SampleType::I32,
            [i32::MIN, 0, i32::MAX]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
        ),
        (
            SampleType::U64,
            [0, 0x8000_0000_0000_0000, u64::MAX]
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect(),
        ),
        (
            SampleType::I64,
            [i64::MIN, 0, i64::MAX]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
        ),
        (
            SampleType::F32,
            [0_u32, 0x8000_0000, 0x7fc0_1234]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        ),
        (
            SampleType::F64,
            [0_u64, 0x8000_0000_0000_0000, 0x7ff8_1234_5678_9abc]
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect(),
        ),
    ]
}

fn endian_bytes(little_endian: &[u8], sample_width: usize, order: ByteOrder) -> Vec<u8> {
    if order == ByteOrder::LeastSignificantByteFirst || sample_width == 1 {
        return little_endian.to_vec();
    }
    little_endian
        .chunks_exact(sample_width)
        .flat_map(|sample| sample.iter().rev().copied())
        .collect()
}

fn volume(
    shape: [usize; 3],
    samples: SampleBuffer,
    metadata: ImageMetadata<3>,
    calibration: IntensityCalibration,
) -> StoredVolume {
    StoredVolume::new(
        shape,
        samples,
        metadata,
        CoordinateMap::Cartesian,
        calibration,
    )
    .expect("test volume satisfies the stored-value contract")
}

fn nearly_collinear_metadata() -> ImageMetadata<3> {
    let separation = 2.0_f64.powi(-30);
    ImageMetadata::new(
        Point::new([0.0; 3]),
        Spacing::try_new([1.0; 3]).expect("unit spacing is positive"),
        Direction::from_rows([
            [1.0, 1.0, 0.0],
            [1.0, 1.0 + separation, 0.0],
            [0.0, 0.0, 1.0],
        ]),
    )
}

#[path = "stored_series_rejection_tests.rs"]
mod rejection_tests;

#[test]
fn both_nifti_versions_preserve_every_scalar_sample_bit_in_both_input_endians() {
    for version in [NiftiVersion::One, NiftiVersion::Two] {
        for (sample_type, expected) in scalar_payloads() {
            for order in [
                ByteOrder::LeastSignificantByteFirst,
                ByteOrder::MostSignificantByteFirst,
            ] {
                let encoded_source = endian_bytes(&expected, sample_type.byte_width(), order);
                let samples = SampleBuffer::decode(sample_type, &encoded_source, order)
                    .expect("source samples decode in the declared byte order");
                let stored = volume(
                    [1, 1, 3],
                    samples,
                    ImageMetadata::default_for_shape([1, 1, 3]),
                    IntensityCalibration::Identity,
                );
                let series = StoredSeries::new(vec![stored], SeriesAxis::SingleVolume)
                    .expect("one-volume series");

                let document = NiftiDocument::from_stored_series("test", &series, version, [])
                    .expect("NIfTI preserves each supported stored scalar type");

                assert_eq!(document.sample_bytes(), expected);
                assert_eq!(document.header().dimensions, [3, 3, 1, 1, 1, 1, 1, 1]);
                assert_eq!(
                    document.header().datatype_code,
                    NiftiDatatype::try_from(sample_type)
                        .expect("all codec sample types map to NIfTI")
                        .code()
                );
            }
        }
    }
}

#[test]
fn singleton_ordered_axis_remains_rank_four() {
    let stored = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![0x1234_u16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Identity,
    );
    let series = StoredSeries::new(vec![stored], SeriesAxis::List).expect("one-entry list");

    let document = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::One, [])
        .expect("NIfTI can represent a rank-four axis of length one");

    assert_eq!(document.header().dimensions, [4, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(document.sample_bytes(), 0x1234_u16.to_le_bytes());
}

#[test]
fn ordered_volumes_keep_rank_four_shape_and_sample_order() {
    let first_samples = SampleBuffer::from_samples(vec![0_u16, 0x1234]);
    let second_samples = SampleBuffer::from_samples(vec![0x8000_u16, u16::MAX]);
    let first_bytes = first_samples
        .encode(ByteOrder::LeastSignificantByteFirst)
        .expect("first stored volume encodes");
    let second_bytes = second_samples
        .encode(ByteOrder::LeastSignificantByteFirst)
        .expect("second stored volume encodes");
    let metadata = ImageMetadata::default_for_shape([1, 1, 2]);
    let first = volume(
        [1, 1, 2],
        first_samples,
        metadata.clone(),
        IntensityCalibration::Identity,
    );
    let second = volume(
        [1, 1, 2],
        second_samples,
        metadata,
        IntensityCalibration::Identity,
    );
    let series = StoredSeries::new(vec![first, second], SeriesAxis::List)
        .expect("two-volume ordered series");

    let document = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::Two, [])
        .expect("NIfTI stores a shared-grid acquisition axis");

    assert_eq!(document.header().dimensions, [4, 2, 1, 1, 2, 1, 1, 1]);
    assert_eq!(
        document.sample_bytes(),
        [first_bytes, second_bytes].concat()
    );
}

#[test]
fn source_metadata_loss_is_returned_with_its_exact_scope() {
    let stored = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![7_u8]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Identity,
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");
    let loss = FormatMetadataLoss::UnknownSemantics {
        location: ConversionLocation::Volume { volume_index: 0 },
        field: "source.private_field".into(),
    };

    let error =
        NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::Two, [loss.clone()])
            .expect_err("unrepresented source fields block conversion");

    let NiftiStoredSeriesError::Preparation(ConversionPrepareError::Capabilities(report)) = error
    else {
        panic!("metadata loss must return its capability report");
    };
    assert_eq!(report.source_format, "dicom");
    assert_eq!(report.target_format, "nifti");
    assert_eq!(
        report.losses.as_ref(),
        [ConversionLoss::FormatMetadata(loss)]
    );
}

#[test]
fn uniform_linear_calibration_uses_nifti_scaling_without_changing_samples() {
    let calibration = LinearCalibration::new(-2.5, 18.0).expect("finite calibration");
    let samples = SampleBuffer::from_samples(vec![u16::MIN, 7, u16::MAX]);
    let expected = samples
        .encode(ByteOrder::LeastSignificantByteFirst)
        .expect("stored sample bytes");
    let stored = volume(
        [1, 1, 3],
        samples,
        ImageMetadata::default_for_shape([1, 1, 3]),
        IntensityCalibration::Linear(calibration),
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");

    for version in [NiftiVersion::One, NiftiVersion::Two] {
        let document = NiftiDocument::from_stored_series("dicom", &series, version, [])
            .expect("NIfTI scaling preserves a uniform linear transform");
        assert_eq!(document.sample_bytes(), expected);
        assert_eq!(document.header().scl_slope, calibration.slope());
        assert_eq!(document.header().scl_inter, calibration.intercept());
    }
}

#[test]
fn singleton_per_frame_calibration_is_written_as_global_scaling() {
    let calibration = LinearCalibration::new(0.5, -4.0).expect("finite calibration");
    let stored = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![0x8000_u16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::PerFrameLinear(Box::from([calibration])),
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");

    let document = NiftiDocument::from_stored_series("nrrd", &series, NiftiVersion::Two, [])
        .expect("one frame has one global calibration");

    assert_eq!(document.header().scl_slope, 0.5);
    assert_eq!(document.header().scl_inter, -4.0);
}

#[test]
fn per_frame_calibration_difference_is_rejected_at_the_frame() {
    let first = LinearCalibration::new(1.0, 0.0).expect("finite calibration");
    let second = LinearCalibration::new(2.0, 0.0).expect("finite calibration");
    let stored = volume(
        [2, 1, 1],
        SampleBuffer::from_samples(vec![11_u16, 29]),
        ImageMetadata::default_for_shape([2, 1, 1]),
        IntensityCalibration::PerFrameLinear(Box::from([first, second])),
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("two-frame series");

    let error = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::Two, [])
        .expect_err("one NIfTI slope cannot express varying frame calibration");

    assert!(matches!(
        error,
        NiftiStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Frame {
                volume_index: 0,
                frame_index: 1
            },
            source: NiftiStoredSeriesRejection::FrameCalibrationMismatch {
                volume_index: 0,
                frame_index: 1
            },
            ..
        })
    ));
}

#[test]
fn zero_slope_calibration_is_not_silently_treated_as_unscaled() {
    let calibration = LinearCalibration::new(0.0, 9.0).expect("finite calibration");
    let stored = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![12_i16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Linear(calibration),
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");

    let error = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::Two, [])
        .expect_err("NIfTI zero slope disables calibration");

    assert!(matches!(
        error,
        NiftiStoredSeriesError::Preparation(ConversionPrepareError::Target {
            source: NiftiStoredSeriesRejection::ZeroSlopeCalibration { volume_index: 0 },
            ..
        })
    ));
}

#[test]
fn nifti2_sform_retains_high_precision_geometry() {
    let metadata = ImageMetadata::new(
        Point::new([0.123_456_789_012_345, -2.25, 7.75]),
        Spacing::try_new([0.321_987_654_321, 2.0, 1.25]).expect("positive spacing"),
        Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let stored = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![1_u8]),
        metadata,
        IntensityCalibration::Identity,
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");

    let document = NiftiDocument::from_stored_series("nrrd", &series, NiftiVersion::Two, [])
        .expect("NIfTI-2 stores f64 spatial fields");
    let header = NiftiHeader::parse(document.uncompressed_bytes())
        .expect("constructed document header parses");

    assert_eq!(header.srow_x[3], -0.123_456_789_012_345);
    assert_eq!(header.srow_y[3], 2.25);
    assert_eq!(header.srow_z[3], 7.75);
    assert_eq!(header.srow_y[2], -0.321_987_654_321);
}

#[test]
fn nifti1_rejects_geometry_that_becomes_singular_when_encoded() {
    let metadata = nearly_collinear_metadata();
    assert_ne!(metadata.direction().determinant(), 0.0);
    let stored = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![1_u8]),
        metadata,
        IntensityCalibration::Identity,
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");

    let error = NiftiDocument::from_stored_series("nrrd", &series, NiftiVersion::One, [])
        .expect_err("NIfTI-1 must reject a singular sform after f32 narrowing");
    let detail = format!("{error:#}");

    assert!(matches!(
        error,
        NiftiStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Volume { volume_index: 0 },
            source: NiftiStoredSeriesRejection::HeaderEncoding(_),
            ..
        })
    ));
    assert!(
        detail.contains("singular after header encoding"),
        "{detail}"
    );
}

#[test]
fn nifti2_preserves_geometry_that_nifti1_cannot_represent() {
    let separation = 2.0_f64.powi(-30);
    let stored = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![1_u8]),
        nearly_collinear_metadata(),
        IntensityCalibration::Identity,
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");

    let document = NiftiDocument::from_stored_series("nrrd", &series, NiftiVersion::Two, [])
        .expect("NIfTI-2 retains the invertible f64 transform");
    let header = NiftiHeader::parse(document.uncompressed_bytes())
        .expect("constructed NIfTI-2 header parses");

    assert_eq!(header.srow_y[1], -(1.0 + separation));
    assert_eq!(
        crate::spatial::sform_handedness([header.srow_x, header.srow_y, header.srow_z]),
        Some(crate::spatial::SpatialHandedness::Left)
    );
}

#[test]
fn nifti2_preserves_finite_geometry_at_extreme_spatial_scales() {
    for scale in [1.0e-110, 1.0e110] {
        let metadata = ImageMetadata::new(
            Point::new([0.0; 3]),
            Spacing::try_new([scale; 3]).expect("finite positive spacing"),
            Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        );
        let stored = volume(
            [1, 1, 1],
            SampleBuffer::from_samples(vec![1_u8]),
            metadata,
            IntensityCalibration::Identity,
        );
        let series = StoredSeries::new(vec![stored], SeriesAxis::SingleVolume)
            .expect("one-volume series with finite geometry");

        let document = NiftiDocument::from_stored_series("nrrd", &series, NiftiVersion::Two, [])
            .expect("NIfTI-2 represents finite extreme spatial scales");
        let header = NiftiHeader::parse(document.uncompressed_bytes())
            .expect("constructed NIfTI-2 header parses");

        assert_eq!(header.srow_x[2], -scale);
        assert_eq!(header.srow_y[1], -scale);
        assert_eq!(header.srow_z[0], scale);
        assert_eq!(
            crate::spatial::sform_handedness([header.srow_x, header.srow_y, header.srow_z]),
            Some(crate::spatial::SpatialHandedness::Left)
        );
    }
}

#[test]
fn constructed_document_round_trips_through_gzip_transport() {
    let stored = volume(
        [1, 1, 2],
        SampleBuffer::from_samples(vec![0_u64, u64::MAX]),
        ImageMetadata::default_for_shape([1, 1, 2]),
        IntensityCalibration::Identity,
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");
    let original = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::Two, [])
        .expect("NIfTI document from stored values");
    let directory = tempdir().expect("temporary directory");
    let path = directory.path().join("stored-series.nii.gz");

    original.write(&path).expect("write gzip transport");
    let restored = NiftiDocument::read(&path).expect("read gzip transport");

    assert_eq!(restored.uncompressed_bytes(), original.uncompressed_bytes());
    assert_eq!(
        restored.sample_bytes(),
        [0_u64.to_le_bytes(), u64::MAX.to_le_bytes()].concat()
    );
}
