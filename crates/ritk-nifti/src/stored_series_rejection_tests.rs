use super::*;

#[test]
fn later_shape_mismatch_is_rejected_at_that_volume() {
    let first = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![1_u16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Identity,
    );
    let second = volume(
        [1, 1, 2],
        SampleBuffer::from_samples(vec![2_u16, 3]),
        ImageMetadata::default_for_shape([1, 1, 2]),
        IntensityCalibration::Identity,
    );
    let series = StoredSeries::new(vec![first, second], SeriesAxis::List).expect("ordered series");

    let error = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::One, [])
        .expect_err("NIfTI has one shared grid for all volumes");

    assert!(matches!(
        error,
        NiftiStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Volume { volume_index: 1 },
            source: NiftiStoredSeriesRejection::ShapeMismatch {
                volume_index: 1,
                ..
            },
            ..
        })
    ));
}

#[test]
fn later_sample_type_mismatch_is_rejected_at_that_volume() {
    let first = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![1_u16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Identity,
    );
    let second = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![2_i16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Identity,
    );
    let series = StoredSeries::new(vec![first, second], SeriesAxis::List).expect("ordered series");

    let error = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::One, [])
        .expect_err("one NIfTI series must use one stored sample type");

    assert!(matches!(
        error,
        NiftiStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Volume { volume_index: 1 },
            source: NiftiStoredSeriesRejection::SampleTypeMismatch {
                volume_index: 1,
                ..
            },
            ..
        })
    ));
}

#[test]
fn later_geometry_mismatch_is_rejected_at_that_volume() {
    let first = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![1_u16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Identity,
    );
    let second = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![2_u16]),
        ImageMetadata::new(
            Point::new([0.0, 0.0, 1.0]),
            Spacing::try_new([1.0; 3]).expect("unit spacing is positive"),
            Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ),
        IntensityCalibration::Identity,
    );
    let series = StoredSeries::new(vec![first, second], SeriesAxis::List).expect("ordered series");

    let error = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::One, [])
        .expect_err("one NIfTI series must use one physical grid");

    assert!(matches!(
        error,
        NiftiStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Volume { volume_index: 1 },
            source: NiftiStoredSeriesRejection::GeometryMismatch { volume_index: 1 },
            ..
        })
    ));
}

#[test]
fn later_calibration_mismatch_is_rejected_at_that_volume() {
    let calibration = LinearCalibration::new(2.0, -4.0).expect("finite calibration");
    let first = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![1_u16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Identity,
    );
    let second = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![2_u16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Linear(calibration),
    );
    let series = StoredSeries::new(vec![first, second], SeriesAxis::List).expect("ordered series");

    let error = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::One, [])
        .expect_err("one NIfTI series must use one intensity calibration");

    assert!(matches!(
        error,
        NiftiStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Volume { volume_index: 1 },
            source: NiftiStoredSeriesRejection::CalibrationMismatch { volume_index: 1 },
            ..
        })
    ));
}

#[test]
fn nifti1_header_dimension_rejection_points_to_the_first_volume() {
    let stored = volume(
        [1, 1, 32_768],
        SampleBuffer::from_samples(vec![1_u8; 32_768]),
        ImageMetadata::default_for_shape([1, 1, 32_768]),
        IntensityCalibration::Identity,
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");

    let error = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::One, [])
        .expect_err("NIfTI-1 cannot encode a spatial dimension larger than i16");

    assert!(matches!(
        error,
        NiftiStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Volume { volume_index: 0 },
            source: NiftiStoredSeriesRejection::HeaderEncoding(_),
            ..
        })
    ));
}

#[test]
fn nifti1_acquisition_count_rejection_is_series_scoped() {
    let volumes = (0..32_768)
        .map(|_| {
            volume(
                [1, 1, 1],
                SampleBuffer::from_samples(vec![1_u8]),
                ImageMetadata::default_for_shape([1, 1, 1]),
                IntensityCalibration::Identity,
            )
        })
        .collect();
    let series = StoredSeries::new(volumes, SeriesAxis::List).expect("ordered list series");

    let error = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::One, [])
        .expect_err("NIfTI-1 cannot encode 32,768 volumes in dim[4]");

    assert!(matches!(
        error,
        NiftiStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Series,
            source: NiftiStoredSeriesRejection::VolumeCountOutOfRange {
                version: NiftiVersion::One,
                volume_count: 32_768,
            },
            ..
        })
    ));
}

#[test]
fn nifti1_rejects_sform_made_singular_by_header_rounding() {
    let metadata = ImageMetadata::new(
        Point::new([0.0; 3]),
        Spacing::try_new([1.0; 3]).expect("unit spacing is positive"),
        Direction::from_rows([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [5.0, 7.0, 9.0 + 2.0_f64.powi(-30)],
        ]),
    );
    let stored = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![1_u8]),
        metadata,
        IntensityCalibration::Identity,
    );
    let series = StoredSeries::new(vec![stored], SeriesAxis::SingleVolume)
        .expect("one-volume series with finite geometry");

    let nifti2 = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::Two, [])
        .expect("NIfTI-2 preserves the invertible source geometry");
    let header =
        NiftiHeader::parse(nifti2.uncompressed_bytes()).expect("constructed NIfTI-2 header parses");
    assert_eq!(header.srow_x, [-3.0, -2.0, -1.0, 0.0]);
    assert_eq!(header.srow_y, [-6.0, -5.0, -4.0, 0.0]);
    assert_eq!(header.srow_z, [9.0 + 2.0_f64.powi(-30), 7.0, 5.0, 0.0]);
    assert_eq!(
        crate::spatial::sform_handedness([header.srow_x, header.srow_y, header.srow_z]),
        Some(crate::spatial::SpatialHandedness::Right)
    );

    let error = NiftiDocument::from_stored_series("dicom", &series, NiftiVersion::One, [])
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
