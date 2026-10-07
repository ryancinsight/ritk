//! Round-trip and edge-case tests for the NIfTI stored-read path.

use super::*;
use crate::NiftiVersion;
use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
use ritk_image::ImageMetadata;
use ritk_image_io::{IntensityCalibration, LinearCalibration, SeriesAxis, StoredVolume};
use ritk_spatial::{Direction, Point, Spacing};

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

fn put_i16(bytes: &mut [u8], offset: usize, value: i16, big_endian: bool) {
    let raw = if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    };
    bytes[offset..offset + 2].copy_from_slice(&raw);
}

fn put_i32(bytes: &mut [u8], offset: usize, value: i32, big_endian: bool) {
    let raw = if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    };
    bytes[offset..offset + 4].copy_from_slice(&raw);
}

fn put_f32(bytes: &mut [u8], offset: usize, value: f32, big_endian: bool) {
    let raw = if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    };
    bytes[offset..offset + 4].copy_from_slice(&raw);
}

/// Minimal two-voxel NIfTI-1 single-file document in a chosen byte order.
///
/// `scl_slope` is left at zero so the read path must treat scaling as disabled.
fn raw_nifti1_document(big_endian: bool, spatial_units: u8) -> Vec<u8> {
    let mut bytes = vec![0_u8; 354];
    put_i32(&mut bytes, 0, 348, big_endian); // sizeof_hdr
    for (index, value) in [3_i16, 2, 1, 1, 1, 1, 1, 1].into_iter().enumerate() {
        put_i16(&mut bytes, 40 + index * 2, value, big_endian); // dim
    }
    put_i16(&mut bytes, 70, 2, big_endian); // datatype = u8
    put_i16(&mut bytes, 72, 8, big_endian); // bitpix
    put_f32(&mut bytes, 76, 0.0, big_endian); // pixdim[0] = qfac
    for (index, value) in [1.0_f32, 1.0, 1.0].into_iter().enumerate() {
        put_f32(&mut bytes, 80 + index * 4, value, big_endian); // pixdim[1..3]
    }
    put_f32(&mut bytes, 108, 352.0, big_endian); // vox_offset
    put_f32(&mut bytes, 112, 0.0, big_endian); // scl_slope = 0 (disabled)
    put_f32(&mut bytes, 116, 0.0, big_endian); // scl_inter
    bytes[123] = spatial_units; // xyzt_units (u8, order independent)
    put_i16(&mut bytes, 252, 0, big_endian); // qform_code
    put_i16(&mut bytes, 254, 0, big_endian); // sform_code
    bytes[344..348].copy_from_slice(b"n+1\0"); // single-file magic
    bytes[352..354].copy_from_slice(&[9, 8]); // two u8 voxels
    bytes
}

#[test]
fn every_scalar_type_round_trips_through_a_nifti_document() {
    for (sample_type, payload) in scalar_payloads() {
        for version in [NiftiVersion::One, NiftiVersion::Two] {
            let samples =
                SampleBuffer::decode(sample_type, &payload, ByteOrder::LeastSignificantByteFirst)
                    .expect("source samples decode");
            let stored = volume(
                [1, 1, 3],
                samples,
                ImageMetadata::default_for_shape([1, 1, 3]),
                IntensityCalibration::Identity,
            );
            let series = StoredSeries::new(vec![stored], SeriesAxis::SingleVolume)
                .expect("one-volume series");

            let document = NiftiDocument::from_stored_series("test", &series, version, [])
                .expect("NIfTI stores every scalar type");
            let restored = document.to_stored_series().expect("document decodes");

            assert_eq!(restored.axis(), &SeriesAxis::SingleVolume);
            assert_eq!(restored.volumes().len(), 1);
            let restored_volume = &restored.volumes()[0];
            assert_eq!(restored_volume.shape(), [1, 1, 3]);
            assert_eq!(restored_volume.samples().sample_type(), sample_type);
            assert_eq!(
                restored_volume
                    .samples()
                    .encode(ByteOrder::LeastSignificantByteFirst)
                    .expect("restored samples encode"),
                payload,
                "stored sample bits must survive a {version:?} {sample_type:?} round trip"
            );
        }
    }
}

#[test]
fn geometry_and_linear_calibration_round_trip() {
    let metadata = ImageMetadata::new(
        Point::new([10.0, -20.0, 30.0]),
        Spacing::try_new([2.0, 3.0, 4.0]).expect("positive spacing"),
        Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let calibration = LinearCalibration::new(-2.5, 18.0).expect("finite calibration");
    let stored = volume(
        [1, 1, 3],
        SampleBuffer::from_samples(vec![0_u16, 7, u16::MAX]),
        metadata,
        IntensityCalibration::Linear(calibration),
    );
    let series =
        StoredSeries::new(vec![stored], SeriesAxis::SingleVolume).expect("one-volume series");

    for version in [NiftiVersion::One, NiftiVersion::Two] {
        let document = NiftiDocument::from_stored_series("test", &series, version, [])
            .expect("NIfTI stores geometry and scaling");
        let restored = document.to_stored_series().expect("document decodes");
        let restored_volume = &restored.volumes()[0];

        assert_eq!(
            restored_volume.metadata().origin().as_slice(),
            &[10.0, -20.0, 30.0]
        );
        assert_eq!(
            restored_volume.metadata().spacing().to_array(),
            [2.0, 3.0, 4.0]
        );
        assert_eq!(
            restored_volume.calibration(),
            &IntensityCalibration::Linear(calibration)
        );
    }
}

#[test]
fn rank_four_single_volume_reads_as_an_ordered_list() {
    let stored = volume(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![0x1234_u16]),
        ImageMetadata::default_for_shape([1, 1, 1]),
        IntensityCalibration::Identity,
    );
    let series = StoredSeries::new(vec![stored], SeriesAxis::List).expect("one-entry list");

    let document = NiftiDocument::from_stored_series("test", &series, NiftiVersion::One, [])
        .expect("rank-four axis of length one");
    let restored = document.to_stored_series().expect("document decodes");

    assert_eq!(restored.axis(), &SeriesAxis::List);
    assert_eq!(restored.volumes().len(), 1);
}

#[test]
fn ordered_volumes_round_trip_in_acquisition_order() {
    let metadata = ImageMetadata::default_for_shape([1, 1, 2]);
    let first = volume(
        [1, 1, 2],
        SampleBuffer::from_samples(vec![0_u16, 0x1234]),
        metadata.clone(),
        IntensityCalibration::Identity,
    );
    let second = volume(
        [1, 1, 2],
        SampleBuffer::from_samples(vec![0x8000_u16, u16::MAX]),
        metadata,
        IntensityCalibration::Identity,
    );
    let series = StoredSeries::new(vec![first, second], SeriesAxis::List).expect("two volumes");

    let document = NiftiDocument::from_stored_series("test", &series, NiftiVersion::Two, [])
        .expect("NIfTI stores a shared-grid acquisition axis");
    let restored = document.to_stored_series().expect("document decodes");

    assert_eq!(restored.axis(), &SeriesAxis::List);
    assert_eq!(restored.volumes().len(), 2);
    assert_eq!(
        restored.volumes()[0]
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)
            .expect("first volume encodes"),
        vec![0x00, 0x00, 0x34, 0x12]
    );
    assert_eq!(
        restored.volumes()[1]
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)
            .expect("second volume encodes"),
        vec![0x00, 0x80, 0xff, 0xff]
    );
}

#[test]
fn big_endian_payload_decodes_in_the_declared_byte_order() {
    let document =
        NiftiDocument::from_bytes(&raw_nifti1_document(true, 2)).expect("big-endian document");
    let restored = document
        .to_stored_series()
        .expect("big-endian document decodes");

    let restored_volume = &restored.volumes()[0];
    assert_eq!(restored_volume.shape(), [1, 1, 2]);
    assert_eq!(
        restored_volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)
            .expect("samples encode"),
        vec![9, 8]
    );
    assert_eq!(
        restored_volume.calibration(),
        &IntensityCalibration::Identity
    );
}

#[test]
fn non_millimeter_spatial_units_are_reported_as_typed_loss() {
    // Spatial-unit code 1 is metres; the stored model is LPS-millimetre.
    let document =
        NiftiDocument::from_bytes(&raw_nifti1_document(false, 1)).expect("metre-unit document");

    let error = document
        .to_stored_series()
        .expect_err("metre units must not be read as millimetres");

    assert!(matches!(
        error,
        NiftiStoredReadError::UnsupportedSpatialUnits { code: 1 }
    ));
}

#[test]
fn unknown_spatial_units_keep_the_millimeter_interpretation() {
    let document =
        NiftiDocument::from_bytes(&raw_nifti1_document(false, 0)).expect("unit-less document");

    let restored = document
        .to_stored_series()
        .expect("an absent unit field keeps LPS millimetres");
    assert_eq!(restored.volumes().len(), 1);
}
