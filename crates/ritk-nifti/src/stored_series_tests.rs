use crate::document::{NiftiDocument, NiftiVersion};
use crate::header::{NiftiDatatype, NiftiHeader};
use crate::stored_series::{NiftiStoredSeriesError, NiftiStoredSeriesIssue};
use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    ConversionFeature, ConversionLocation, ConversionLoss, IntensityCalibration, LinearCalibration,
    SeriesAxis, StoredSeries, StoredVolume,
};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};

fn metadata(origin: [f64; 3], spacing: [f64; 3]) -> ImageMetadata<3> {
    ImageMetadata::new(
        Point::new(origin),
        Spacing::new(spacing),
        Direction::identity(),
    )
}

fn volume(
    shape: [usize; 3],
    sample_type: SampleType,
    encoded: &[u8],
    geometry: ImageMetadata<3>,
    calibration: IntensityCalibration,
) -> StoredVolume {
    let samples = SampleBuffer::decode(sample_type, encoded, ByteOrder::LeastSignificantByteFirst)
        .expect("fixture bytes match the declared stored sample type");
    StoredVolume::new(
        shape,
        samples,
        geometry,
        CoordinateMap::Cartesian,
        calibration,
    )
    .expect("fixture volume satisfies stored-volume invariants")
}

fn one_sample_series(sample_type: SampleType, encoded: &[u8]) -> StoredSeries {
    StoredSeries::new(
        vec![volume(
            [1, 1, 1],
            sample_type,
            encoded,
            metadata([0.0; 3], [1.0; 3]),
            IntensityCalibration::Identity,
        )],
        SeriesAxis::SingleVolume,
    )
    .expect("single-volume fixture is valid")
}

#[test]
fn both_header_versions_preserve_every_scalar_samples_bit_pattern() {
    let cases = [
        (SampleType::U8, 2, 8, vec![0xa7]),
        (SampleType::I8, 256, 8, vec![0x87]),
        (SampleType::U16, 512, 16, 0xa13f_u16.to_le_bytes().to_vec()),
        (SampleType::I16, 4, 16, (-12_225_i16).to_le_bytes().to_vec()),
        (
            SampleType::U32,
            768,
            32,
            0xa13f_7701_u32.to_le_bytes().to_vec(),
        ),
        (
            SampleType::I32,
            8,
            32,
            (-784_369_919_i32).to_le_bytes().to_vec(),
        ),
        (
            SampleType::U64,
            1280,
            64,
            0xa13f_7701_c345_6789_u64.to_le_bytes().to_vec(),
        ),
        (
            SampleType::I64,
            1024,
            64,
            (-3_368_843_146_795_063_415_i64).to_le_bytes().to_vec(),
        ),
        (
            SampleType::F32,
            16,
            32,
            0xffc0_1234_u32.to_le_bytes().to_vec(),
        ),
        (
            SampleType::F64,
            64,
            64,
            0xfff8_0000_0000_1234_u64.to_le_bytes().to_vec(),
        ),
    ];

    for version in [NiftiVersion::One, NiftiVersion::Two] {
        for (sample_type, datatype_code, bits_per_sample, sample_bytes) in &cases {
            let document = NiftiDocument::from_stored_series(
                &one_sample_series(*sample_type, sample_bytes),
                version,
            )
            .expect("supported stored scalar converts without changing bits");

            assert_eq!(document.header().datatype_code, *datatype_code);
            assert_eq!(document.header().bits_per_sample, *bits_per_sample);
            assert_eq!(document.sample_bytes(), sample_bytes);
        }
    }
}

#[test]
fn list_axis_becomes_rank_four_in_acquisition_order() {
    let geometry = ImageMetadata::new(
        Point::new([10.0, 20.0, 30.0]),
        Spacing::new([2.0, 3.0, 4.0]),
        Direction::from_row_major([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
    );
    let series = StoredSeries::new(
        vec![
            volume(
                [1, 1, 2],
                SampleType::U16,
                &[11, 0, 13, 0],
                geometry.clone(),
                IntensityCalibration::Identity,
            ),
            volume(
                [1, 1, 2],
                SampleType::U16,
                &[29, 0, 31, 0],
                geometry,
                IntensityCalibration::Identity,
            ),
        ],
        SeriesAxis::List,
    )
    .expect("two-volume list fixture is valid");

    let document = NiftiDocument::from_stored_series(&series, NiftiVersion::Two)
        .expect("uniform ordered list converts to a rank-four NIfTI document");
    let header = NiftiHeader::parse(document.uncompressed_bytes()).expect("header parses");

    assert_eq!(header.dim, [4, 2, 1, 1, 2, 1, 1, 1]);
    assert_eq!(header.datatype, NiftiDatatype::Uint16);
    assert_eq!(document.sample_bytes(), [11, 0, 13, 0, 29, 0, 31, 0]);
    assert_eq!(header.pixdim[1..4], [4.0, 3.0, 2.0]);
    assert_eq!(header.srow_x, [-4.0, -0.0, -0.0, -10.0]);
    assert_eq!(header.srow_y, [-0.0, -3.0, -0.0, -20.0]);
    assert_eq!(header.srow_z, [0.0, 0.0, 2.0, 30.0]);
}

#[test]
fn singleton_list_axis_is_rejected_before_axis_is_erased() {
    let series = StoredSeries::new(
        vec![volume(
            [1, 1, 1],
            SampleType::U8,
            &[7],
            metadata([0.0; 3], [1.0; 3]),
            IntensityCalibration::Identity,
        )],
        SeriesAxis::List,
    )
    .expect("singleton list fixture is valid");

    assert!(matches!(
        NiftiDocument::from_stored_series(&series, NiftiVersion::Two),
        Err(NiftiStoredSeriesError::Rejected(rejection))
            if rejection.issue == NiftiStoredSeriesIssue::ListAxisRequiresMultipleVolumes
    ));
}

fn unspecified_axis_is_reported_before_document_creation(volume_count: usize) {
    let volumes = (0..volume_count)
        .map(|_| {
            volume(
                [1, 1, 1],
                SampleType::U8,
                &[7],
                metadata([0.0; 3], [1.0; 3]),
                IntensityCalibration::Identity,
            )
        })
        .collect();
    let series = StoredSeries::new(volumes, SeriesAxis::Unspecified)
        .expect("unspecified-axis fixture is valid");

    let Err(NiftiStoredSeriesError::Capabilities(report)) =
        NiftiDocument::from_stored_series(&series, NiftiVersion::Two)
    else {
        panic!("unsupported axis must be reported before document creation");
    };
    assert_eq!(
        report.losses.as_ref(),
        &[ConversionLoss::UnsupportedFeature {
            location: ConversionLocation::Series,
            feature: ConversionFeature::UnspecifiedAxis,
        }]
    );
}

#[test]
fn unspecified_axis_is_reported_for_single_and_multiple_volumes() {
    for volume_count in [1, 2] {
        unspecified_axis_is_reported_before_document_creation(volume_count);
    }
}

#[test]
fn one_global_linear_calibration_is_encoded_in_the_header() {
    let calibration = IntensityCalibration::Linear(
        LinearCalibration::new(2.5, -1024.0).expect("finite coefficients"),
    );
    let series = StoredSeries::new(
        vec![volume(
            [1, 1, 1],
            SampleType::I16,
            &[7, 0],
            metadata([0.0; 3], [1.0; 3]),
            calibration,
        )],
        SeriesAxis::SingleVolume,
    )
    .expect("calibrated fixture is valid");

    for version in [NiftiVersion::One, NiftiVersion::Two] {
        let document = NiftiDocument::from_stored_series(&series, version)
            .expect("nonzero linear scaling maps to NIfTI fields");
        assert_eq!(document.header().scl_slope, 2.5);
        assert_eq!(document.header().scl_inter, -1024.0);
        assert_eq!(document.sample_bytes(), [7, 0]);
    }
}

#[test]
fn inconsistent_shape_sample_type_geometry_and_calibration_are_rejected() {
    let geometry = metadata([0.0; 3], [1.0; 3]);
    let identity = IntensityCalibration::Identity;
    let cases = [
        (
            vec![
                volume(
                    [1, 1, 1],
                    SampleType::U16,
                    &[1, 0],
                    geometry.clone(),
                    identity.clone(),
                ),
                volume(
                    [1, 1, 2],
                    SampleType::U16,
                    &[2, 0, 3, 0],
                    geometry.clone(),
                    identity.clone(),
                ),
            ],
            NiftiStoredSeriesIssue::ShapeMismatch {
                expected: [1, 1, 1],
                actual: [1, 1, 2],
            },
        ),
        (
            vec![
                volume(
                    [1, 1, 1],
                    SampleType::U16,
                    &[1, 0],
                    geometry.clone(),
                    identity.clone(),
                ),
                volume(
                    [1, 1, 1],
                    SampleType::I16,
                    &[2, 0],
                    geometry.clone(),
                    identity.clone(),
                ),
            ],
            NiftiStoredSeriesIssue::SampleTypeMismatch {
                expected: SampleType::U16,
                actual: SampleType::I16,
            },
        ),
        (
            vec![
                volume(
                    [1, 1, 1],
                    SampleType::U16,
                    &[1, 0],
                    geometry.clone(),
                    identity.clone(),
                ),
                volume(
                    [1, 1, 1],
                    SampleType::U16,
                    &[2, 0],
                    metadata([0.0, 0.0, 1.0], [1.0; 3]),
                    identity.clone(),
                ),
            ],
            NiftiStoredSeriesIssue::GeometryMismatch,
        ),
        (
            vec![
                volume(
                    [1, 1, 1],
                    SampleType::U16,
                    &[1, 0],
                    geometry.clone(),
                    identity.clone(),
                ),
                volume(
                    [1, 1, 1],
                    SampleType::U16,
                    &[2, 0],
                    geometry,
                    IntensityCalibration::Linear(
                        LinearCalibration::new(2.0, 0.0).expect("finite coefficients"),
                    ),
                ),
            ],
            NiftiStoredSeriesIssue::CalibrationMismatch,
        ),
    ];

    for (volumes, expected) in cases {
        let series = StoredSeries::new(volumes, SeriesAxis::List)
            .expect("inconsistent target values remain valid stored-series inputs");
        let Err(NiftiStoredSeriesError::Rejected(rejection)) =
            NiftiDocument::from_stored_series(&series, NiftiVersion::Two)
        else {
            panic!("target must reject {expected:?}");
        };
        assert_eq!(rejection.issue, expected);
    }
}

#[test]
fn zero_slope_and_varying_frame_calibrations_are_rejected_at_their_scope() {
    let zero_slope = StoredSeries::new(
        vec![volume(
            [1, 1, 1],
            SampleType::I16,
            &[5, 0],
            metadata([0.0; 3], [1.0; 3]),
            IntensityCalibration::Linear(
                LinearCalibration::new(0.0, 4.0).expect("finite zero slope is valid in RITK"),
            ),
        )],
        SeriesAxis::SingleVolume,
    )
    .expect("constant-map calibration is valid in RITK");
    assert!(matches!(
        NiftiDocument::from_stored_series(&zero_slope, NiftiVersion::Two),
        Err(NiftiStoredSeriesError::Rejected(rejection))
            if rejection.issue == NiftiStoredSeriesIssue::ZeroSlopeCalibration
    ));

    let frame_calibration = IntensityCalibration::PerFrameLinear(Box::from([
        LinearCalibration::new(1.0, 0.0).expect("identity coefficients"),
        LinearCalibration::new(2.0, 0.0).expect("finite coefficients"),
    ]));
    let frames = StoredSeries::new(
        vec![volume(
            [2, 1, 1],
            SampleType::I16,
            &[5, 0, 7, 0],
            metadata([0.0; 3], [1.0; 3]),
            frame_calibration,
        )],
        SeriesAxis::SingleVolume,
    )
    .expect("per-frame calibration matches depth");
    assert!(matches!(
        NiftiDocument::from_stored_series(&frames, NiftiVersion::Two),
        Err(NiftiStoredSeriesError::Rejected(rejection))
            if matches!(rejection.location, ritk_image_io::ConversionLocation::Frame {
                volume_index: 0,
                frame_index: 1
            })
    ));
}

#[test]
fn unsupported_calibration_is_reported_before_document_creation() {
    let lookup = ritk_image_io::ModalityLookupTable::new(
        0,
        Box::from([0_u16, 5]),
        ritk_image_io::LutOutputBits::Sixteen,
    )
    .expect("lookup fixture is valid");
    let series = StoredSeries::new(
        vec![volume(
            [1, 1, 1],
            SampleType::U16,
            &[5, 0],
            metadata([0.0; 3], [1.0; 3]),
            IntensityCalibration::ModalityLookup(lookup),
        )],
        SeriesAxis::SingleVolume,
    )
    .expect("lookup-calibrated fixture is valid");

    let Err(NiftiStoredSeriesError::Capabilities(report)) =
        NiftiDocument::from_stored_series(&series, NiftiVersion::Two)
    else {
        panic!("unsupported lookup calibration must be reported");
    };
    assert!(
        report.losses.iter().any(|loss| matches!(
            loss,
            ritk_image_io::ConversionLoss::UnsupportedFeature {
                feature: ritk_image_io::ConversionFeature::ModalityLookupCalibration,
                ..
            }
        )),
        "capability report names the unsupported calibration category"
    );
}

#[test]
fn nifti_one_rejects_geometry_outside_its_header_precision_range() {
    let oversized_geometry = StoredSeries::new(
        vec![volume(
            [1, 1, 1],
            SampleType::U8,
            &[7],
            metadata([0.0; 3], [1.0, 1.0, 1.0e100]),
            IntensityCalibration::Identity,
        )],
        SeriesAxis::SingleVolume,
    )
    .expect("large finite spacing remains valid physical metadata");

    assert!(matches!(
        NiftiDocument::from_stored_series(&oversized_geometry, NiftiVersion::One),
        Err(NiftiStoredSeriesError::Rejected(rejection))
            if matches!(rejection.issue, NiftiStoredSeriesIssue::HeaderField { .. })
    ));
    let document = NiftiDocument::from_stored_series(&oversized_geometry, NiftiVersion::Two)
        .expect("NIfTI-2 retains the supported f64 spacing");
    let header = NiftiHeader::parse(document.uncompressed_bytes()).expect("header parses");
    assert_eq!(header.dim, [3, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(header.datatype, NiftiDatatype::Uint8);
    assert_eq!(header.pixdim[1..4], [1.0e100, 1.0, 1.0]);
    assert_eq!(header.srow_x, [0.0, 0.0, -1.0, 0.0]);
    assert_eq!(header.srow_y, [0.0, -1.0, 0.0, 0.0]);
    assert_eq!(header.srow_z, [1.0e100, 0.0, 0.0, 0.0]);
    assert_eq!(document.sample_bytes(), [7]);
}

#[test]
fn nifti_one_rejects_inexact_calibration_and_nifti_two_preserves_it() {
    let series = StoredSeries::new(
        vec![volume(
            [1, 1, 1],
            SampleType::U16,
            &[17, 0],
            metadata([0.0; 3], [1.0; 3]),
            IntensityCalibration::Linear(
                LinearCalibration::new(0.1, 0.0).expect("finite coefficients"),
            ),
        )],
        SeriesAxis::SingleVolume,
    )
    .expect("one-volume calibration fixture is valid");

    let error = NiftiDocument::from_stored_series(&series, NiftiVersion::One)
        .expect_err("NIfTI-1 must not round an unrepresentable calibration");
    match error {
        NiftiStoredSeriesError::Rejected(rejection) => {
            assert_eq!(
                rejection.location,
                ritk_image_io::ConversionLocation::Series
            );
            match rejection.issue {
                NiftiStoredSeriesIssue::HeaderField { detail } => {
                    assert!(detail.contains("scl_slope"));
                }
                issue => panic!("expected a scaling-field rejection, got {issue:?}"),
            }
        }
        error => panic!("expected a scoped header-field rejection, got {error:?}"),
    }

    let document = NiftiDocument::from_stored_series(&series, NiftiVersion::Two)
        .expect("NIfTI-2 stores this f64 calibration value");
    assert_eq!(document.header().scl_slope, 0.1);
    assert_eq!(document.sample_bytes(), [17, 0]);
}
