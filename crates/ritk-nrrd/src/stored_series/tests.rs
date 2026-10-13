use super::*;
use crate::document::write_nrrd_document;
use crate::NrrdDocumentError;
use ritk_codecs::SampleBuffer;
use ritk_image::ImageMetadata;
use ritk_image_io::{IntensityCalibration, LinearCalibration, StoredVolume};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
use std::fs;
use tempfile::tempdir;

fn metadata() -> ImageMetadata<3> {
    ImageMetadata::new(
        Point::new([10.0, 20.0, 30.0]),
        Spacing::new([0.5, 1.5, 2.0]),
        Direction::identity(),
    )
}

fn u16_volume_at(shape: [usize; 3], origin: [f64; 3]) -> StoredVolume {
    let count = shape.iter().product();
    StoredVolume::new(
        shape,
        SampleBuffer::from_samples(vec![7_u16; count]),
        ImageMetadata::new(
            Point::new(origin),
            Spacing::new([0.5, 1.5, 2.0]),
            Direction::identity(),
        ),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )
    .expect("valid stored volume")
}

fn u16_volume(shape: [usize; 3]) -> StoredVolume {
    u16_volume_at(shape, [10.0, 20.0, 30.0])
}

fn i16_volume(shape: [usize; 3]) -> StoredVolume {
    let count = shape.iter().product();
    StoredVolume::new(
        shape,
        SampleBuffer::from_samples(vec![1_i16; count]),
        metadata(),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )
    .expect("valid stored volume")
}

#[test]
fn a_single_volume_series_is_prepared_and_round_trips() {
    let series = StoredSeries::new(vec![u16_volume([2, 3, 4])], SeriesAxis::SingleVolume)
        .expect("single-volume series");

    let document = NrrdDocument::from_stored_series("dicom", series, Vec::new(), Vec::new(), [])
        .expect("a Cartesian u16 volume is fully representable");

    assert_eq!(document.series().volumes().len(), 1);
    assert_eq!(document.series().volumes()[0].shape(), [2, 3, 4]);
}

#[test]
fn later_shape_mismatch_is_rejected_at_that_volume() {
    let series = StoredSeries::new(
        vec![u16_volume([1, 1, 1]), u16_volume([1, 1, 2])],
        SeriesAxis::List,
    )
    .expect("ordered series");

    let error = NrrdDocument::from_stored_series("dicom", series, Vec::new(), Vec::new(), [])
        .expect_err("NRRD series share one `sizes` record");

    assert!(matches!(
        error,
        NrrdStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Volume { volume_index: 1 },
            source: NrrdStoredSeriesRejection::ShapeMismatch {
                volume_index: 1,
                ..
            },
            ..
        })
    ));
}

#[test]
fn later_sample_type_mismatch_is_rejected_at_that_volume() {
    let first = u16_volume([1, 1, 1]);
    let second = i16_volume([1, 1, 1]);
    let series = StoredSeries::new(vec![first, second], SeriesAxis::List).expect("ordered series");

    let error = NrrdDocument::from_stored_series("dicom", series, Vec::new(), Vec::new(), [])
        .expect_err("one NRRD series must use one element type");

    assert!(matches!(
        error,
        NrrdStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Volume { volume_index: 1 },
            source: NrrdStoredSeriesRejection::SampleTypeMismatch { volume_index: 1 },
            ..
        })
    ));
}

#[test]
fn later_geometry_mismatch_is_rejected_at_that_volume() {
    let first = u16_volume([1, 1, 1]);
    let shifted = u16_volume_at([1, 1, 1], [99.0, 20.0, 30.0]);
    let series = StoredSeries::new(vec![first, shifted], SeriesAxis::List).expect("ordered series");

    let error = NrrdDocument::from_stored_series("dicom", series, Vec::new(), Vec::new(), [])
        .expect_err("one NRRD series has one space origin");

    assert!(matches!(
        error,
        NrrdStoredSeriesError::Preparation(ConversionPrepareError::Target {
            location: ConversionLocation::Volume { volume_index: 1 },
            source: NrrdStoredSeriesRejection::GeometryMismatch { volume_index: 1 },
            ..
        })
    ));
}

#[test]
fn non_identity_calibration_is_reported_at_its_volume() {
    let scaled = StoredVolume::new(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![1_u16]),
        metadata(),
        CoordinateMap::Cartesian,
        IntensityCalibration::Linear(
            LinearCalibration::new(2.0, 1.0).expect("finite linear calibration"),
        ),
    )
    .expect("valid stored volume");
    let series = StoredSeries::new(vec![scaled], SeriesAxis::SingleVolume).expect("series");

    let error = NrrdDocument::from_stored_series("dicom", series, Vec::new(), Vec::new(), [])
        .expect_err("NRRD has no field for a stored-to-real transform");

    assert!(matches!(
        error,
        NrrdStoredSeriesError::Preparation(ConversionPrepareError::Capabilities(report))
            if report.losses.iter().any(|loss| matches!(
                loss,
                ritk_image_io::ConversionLoss::UnsupportedFeature {
                    location: ConversionLocation::Volume { volume_index: 0 },
                    feature: ConversionFeature::LinearCalibration,
                }
            ))
    ));
}

#[test]
fn a_rejected_write_never_creates_the_destination() {
    let directory = tempdir().expect("scratch directory");
    let path = directory.path().join("absent.nrrd");

    let series = StoredSeries::new(vec![u16_volume([1, 1, 1])], SeriesAxis::SingleVolume)
        .expect("single-volume series");
    let mut document =
        NrrdDocument::from_stored_series("dicom", series, Vec::new(), Vec::new(), [])
            .expect("valid document");
    document.records = vec![("type".to_owned(), "float".to_owned())];

    assert!(
        write_nrrd_document(&path, &document).is_err(),
        "a conflicting generated field must be rejected"
    );
    assert!(
        !path.exists(),
        "the write guard must reject before the destination is opened"
    );
}

#[test]
fn a_rejected_write_preserves_an_existing_destination() {
    let directory = tempdir().expect("scratch directory");
    let path = directory.path().join("existing.nrrd");
    fs::write(&path, b"sentinel").expect("seed destination");

    let series = StoredSeries::new(vec![u16_volume([1, 1, 1])], SeriesAxis::SingleVolume)
        .expect("single-volume series");
    let mut document =
        NrrdDocument::from_stored_series("dicom", series, Vec::new(), Vec::new(), [])
            .expect("valid document");
    // `records` is crate-visible, so a document can be mutated after
    // construction; the write guard has to catch that, not trust the value.
    document.records = vec![("type".to_owned(), "float".to_owned())];

    assert!(matches!(
        write_nrrd_document(&path, &document),
        Err(NrrdDocumentError::ConflictingMetadata { .. })
    ));
    assert_eq!(fs::read(&path).expect("read destination"), b"sentinel");
}
