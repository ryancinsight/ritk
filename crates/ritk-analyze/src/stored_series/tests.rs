//! Behavior tests for the Analyze stored-sample conversion.

use super::*;
use ritk_spatial::{CurvilinearArray, Point, Spacing};
use tempfile::tempdir;

/// Depth, row, column shape shared by the fixtures.
const SHAPE: [usize; 3] = [2, 3, 4];

fn i16_values() -> Vec<i16> {
    (0..24).map(|index| index as i16 - 5).collect()
}

fn expected_i16_bytes(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn metadata() -> ImageMetadata<3> {
    ImageMetadata::new(
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
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
    .expect("valid stored volume")
}

fn i16_volume(values: Vec<i16>) -> StoredVolume {
    volume(
        SHAPE,
        SampleBuffer::from_samples(values),
        metadata(),
        IntensityCalibration::Identity,
    )
}

fn single_volume(volume: StoredVolume) -> StoredSeries {
    StoredSeries::new(vec![volume], SeriesAxis::SingleVolume).expect("single-volume series")
}

fn i16_document(values: Vec<i16>, description: Vec<u8>) -> AnalyzeStoredSeries {
    AnalyzeStoredSeries::from_stored_series(
        ANALYZE_STORED_SOURCE,
        single_volume(i16_volume(values)),
        description,
        [],
    )
    .expect("representable series")
}

/// Runs the target adapter directly, bypassing the capability report.
fn prepare(series: &StoredSeries) -> Result<AnalyzeStoredSeriesPlan, AnalyzeStoredSeriesRejection> {
    AnalyzeStoredSeriesTarget {
        description: b"RITK",
    }
    .prepare(series)
}

#[test]
fn prepare_rejects_an_unsigned_short_series() {
    let series = single_volume(volume(
        SHAPE,
        SampleBuffer::from_samples(vec![7_u16; 24]),
        metadata(),
        IntensityCalibration::Identity,
    ));

    let error = prepare(&series).expect_err("Analyze 7.5 has no unsigned short code");
    assert!(matches!(
        error,
        AnalyzeStoredSeriesRejection::UnsupportedSampleType {
            volume_index: 0,
            sample_type: SampleType::U16,
        }
    ));
    assert_eq!(
        error.location(),
        ConversionLocation::Volume { volume_index: 0 }
    );
}

#[test]
fn prepare_rejects_a_non_cartesian_coordinate_map() {
    let map = CoordinateMap::CurvilinearArray(
        CurvilinearArray::try_new(1.0, 0.0, 0.01, 0.0).expect("valid curvilinear map"),
    );
    let stored = StoredVolume::new(
        SHAPE,
        SampleBuffer::from_samples(i16_values()),
        metadata(),
        map,
        IntensityCalibration::Identity,
    )
    .expect("valid stored volume");
    let series = single_volume(stored);

    let error = prepare(&series).expect_err("Analyze stores a Cartesian grid only");
    assert!(matches!(
        error,
        AnalyzeStoredSeriesRejection::UnsupportedCoordinateMap {
            volume_index: 0,
            ..
        }
    ));
}

#[test]
fn prepare_rejects_a_per_frame_calibration() {
    let frame = LinearCalibration::new(1.0, 0.0).expect("finite coefficients");
    let series = single_volume(volume(
        SHAPE,
        SampleBuffer::from_samples(i16_values()),
        metadata(),
        // One entry per depth frame, matching `SHAPE[0]`.
        IntensityCalibration::PerFrameLinear(vec![frame; SHAPE[0]].into()),
    ));

    let error = prepare(&series).expect_err("Analyze has one scale factor, not one per frame");
    assert!(matches!(
        error,
        AnalyzeStoredSeriesRejection::UnsupportedCalibration {
            volume_index: 0,
            ..
        }
    ));
}

#[test]
fn prepare_rejects_a_non_identity_direction() {
    let oblique = ImageMetadata::new(
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let series = single_volume(volume(
        SHAPE,
        SampleBuffer::from_samples(i16_values()),
        oblique,
        IntensityCalibration::Identity,
    ));

    let error = prepare(&series).expect_err("Analyze has no direction field");
    assert!(matches!(
        error,
        AnalyzeStoredSeriesRejection::NonIdentityDirection { volume_index: 0 }
    ));
}

#[test]
fn prepare_rejects_a_calibration_with_an_additive_term() {
    let series = single_volume(volume(
        SHAPE,
        SampleBuffer::from_samples(i16_values()),
        metadata(),
        IntensityCalibration::Linear(
            LinearCalibration::new(2.0, 1.5).expect("finite coefficients"),
        ),
    ));

    let error = prepare(&series).expect_err("Analyze's scale factor has no additive term");
    assert!(matches!(
        error,
        AnalyzeStoredSeriesRejection::NonZeroInterceptCalibration {
            volume_index: 0,
            intercept,
        } if intercept == 1.5
    ));
}

#[test]
fn prepare_rejects_a_zero_slope_calibration() {
    let series = single_volume(volume(
        SHAPE,
        SampleBuffer::from_samples(i16_values()),
        metadata(),
        IntensityCalibration::Linear(
            LinearCalibration::new(0.0, 0.0).expect("finite coefficients"),
        ),
    ));

    let error = prepare(&series).expect_err("a stored zero scale means no scaling");
    assert!(matches!(
        error,
        AnalyzeStoredSeriesRejection::ZeroSlopeCalibration { volume_index: 0 }
    ));
}

#[test]
fn prepare_rejects_a_dimension_beyond_the_header_field() {
    let series = single_volume(volume(
        [1, 1, 40_000],
        SampleBuffer::from_samples(vec![1_i16; 40_000]),
        metadata(),
        IntensityCalibration::Identity,
    ));

    let error = prepare(&series).expect_err("dim[1] is an i16 field");
    let AnalyzeStoredSeriesRejection::HeaderEncoding(source) = &error else {
        panic!("expected a header encoding rejection, got {error}");
    };
    assert!(
        source.to_string().contains("exceeds i16::MAX"),
        "unexpected error: {source}"
    );
    assert_eq!(error.location(), ConversionLocation::Series);
}

#[test]
fn prepare_rejects_a_multi_volume_axis() {
    let series = StoredSeries::new(
        vec![i16_volume(i16_values()), i16_volume(i16_values())],
        SeriesAxis::List,
    )
    .expect("two-volume list series");

    let error = prepare(&series).expect_err("Analyze stores exactly one 3-D volume");
    assert!(matches!(
        error,
        AnalyzeStoredSeriesRejection::UnsupportedSeriesAxis
    ));
}

#[test]
fn the_capability_report_blocks_an_unsigned_short_series() {
    let series = single_volume(volume(
        SHAPE,
        SampleBuffer::from_samples(vec![7_u16; 24]),
        metadata(),
        IntensityCalibration::Identity,
    ));

    let error =
        AnalyzeStoredSeries::from_stored_series(ANALYZE_STORED_SOURCE, series, Vec::new(), [])
            .expect_err("unsigned short is not a declared Analyze feature");
    assert!(
        matches!(
            error,
            AnalyzeStoredSeriesError::Preparation(ConversionPrepareError::Capabilities(_))
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn an_i16_pair_round_trips_exact_stored_samples() {
    let directory = tempdir().expect("scratch directory");
    let path = directory.path().join("volume.hdr");
    let values = i16_values();
    let document = i16_document(values.clone(), b"round trip".to_vec());

    write_analyze_stored(&path, &document).expect("write");
    let read = read_analyze_stored(&path).expect("read");

    let stored = &read.series().volumes()[0];
    assert_eq!(stored.shape(), SHAPE);
    assert!(matches!(
        stored.calibration(),
        IntensityCalibration::Identity
    ));
    assert!(matches!(stored.coordinate_map(), CoordinateMap::Cartesian));
    assert_eq!(stored.metadata().direction(), &Direction::identity());
    assert_eq!(stored.metadata().spacing().to_array(), [1.0; 3]);
    assert_eq!(stored.metadata().origin().as_slice(), [0.0; 3]);
    assert_eq!(
        stored
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)
            .expect("encode"),
        expected_i16_bytes(&values)
    );
    assert_eq!(read.description(), b"round trip".as_slice());
}

#[test]
fn a_linear_scale_round_trips_as_calibration_rather_than_being_applied() {
    let directory = tempdir().expect("scratch directory");
    let path = directory.path().join("scaled.hdr");
    let values = i16_values();
    let series = single_volume(volume(
        SHAPE,
        SampleBuffer::from_samples(values.clone()),
        metadata(),
        IntensityCalibration::Linear(
            LinearCalibration::new(2.5, 0.0).expect("finite coefficients"),
        ),
    ));
    let document =
        AnalyzeStoredSeries::from_stored_series(ANALYZE_STORED_SOURCE, series, Vec::new(), [])
            .expect("representable series");

    write_analyze_stored(&path, &document).expect("write");
    let read = read_analyze_stored(&path).expect("read");

    let stored = &read.series().volumes()[0];
    // The scale travels beside the samples instead of being folded into them.
    assert_eq!(
        stored
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)
            .expect("encode"),
        expected_i16_bytes(&values)
    );
    let IntensityCalibration::Linear(linear) = stored.calibration() else {
        panic!("the header scale must be carried as a linear calibration");
    };
    assert_eq!(linear.slope(), 2.5);
    assert_eq!(linear.intercept(), 0.0);
}

#[test]
fn a_rejected_write_preserves_both_destinations() {
    let directory = tempdir().expect("scratch directory");
    let path = directory.path().join("existing.hdr");
    let img_path = path.with_extension("img");
    let mut document = i16_document(i16_values(), b"RITK".to_vec());
    write_analyze_stored(&path, &document).expect("initial write");
    let header_before = std::fs::read(&path).expect("header bytes");
    let payload_before = std::fs::read(&img_path).expect("payload bytes");

    // `description` is reachable inside the crate, so a document can be mutated
    // after construction. The write guard must refuse it before either file is
    // touched.
    document.description = vec![b'x'; 81];
    let error = write_analyze_stored(&path, &document)
        .expect_err("81 description bytes exceed the 80-byte descrip field");
    assert!(
        error.to_string().contains("descrip holds at most 80"),
        "unexpected error: {error}"
    );

    assert_eq!(std::fs::read(&path).expect("header bytes"), header_before);
    assert_eq!(
        std::fs::read(&img_path).expect("payload bytes"),
        payload_before
    );
}
