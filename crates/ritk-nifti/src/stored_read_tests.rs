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

fn put_signed_short(bytes: &mut [u8], offset: usize, value: i16, big_endian: bool) {
    let raw = if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    };
    bytes[offset..offset + 2].copy_from_slice(&raw);
}

fn put_signed_int(bytes: &mut [u8], offset: usize, value: i32, big_endian: bool) {
    let raw = if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    };
    bytes[offset..offset + 4].copy_from_slice(&raw);
}

fn put_float(bytes: &mut [u8], offset: usize, value: f32, big_endian: bool) {
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
    put_signed_int(&mut bytes, 0, 348, big_endian); // sizeof_hdr
    for (index, value) in [3_i16, 2, 1, 1, 1, 1, 1, 1].into_iter().enumerate() {
        put_signed_short(&mut bytes, 40 + index * 2, value, big_endian); // dim
    }
    put_signed_short(&mut bytes, 70, 2, big_endian); // datatype = u8
    put_signed_short(&mut bytes, 72, 8, big_endian); // bitpix
    put_float(&mut bytes, 76, 0.0, big_endian); // pixdim[0] = qfac
    for (index, value) in [1.0_f32, 1.0, 1.0].into_iter().enumerate() {
        put_float(&mut bytes, 80 + index * 4, value, big_endian); // pixdim[1..3]
    }
    put_float(&mut bytes, 108, 352.0, big_endian); // vox_offset
    put_float(&mut bytes, 112, 0.0, big_endian); // scl_slope = 0 (disabled)
    put_float(&mut bytes, 116, 0.0, big_endian); // scl_inter
    bytes[123] = spatial_units; // xyzt_units (u8, order independent)
    put_signed_short(&mut bytes, 252, 0, big_endian); // qform_code
    put_signed_short(&mut bytes, 254, 0, big_endian); // sform_code
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

/// A NIfTI-1 document whose spatial forms the caller activates.
///
/// The sform carries a diagonal `(2, 3, 4)` mm grid at RAS origin
/// `(10, 20, 30)`; the qform carries an identity rotation at RAS origin
/// `(100, 200, 300)`. The two disagree, so a reader that consults the wrong
/// form is *caught* rather than merely unobserved. `pixdim_xyz` is the
/// fallback grid used only when neither form is active, and `declared_voxels`
/// may exceed `voxel_bytes.len()` to build a short payload.
fn raw_nifti1_spatial_document(
    qform_code: i16,
    sform_code: i16,
    pixdim_xyz: [f32; 3],
    voxel_bytes: &[u8],
    declared_voxels: i16,
) -> Vec<u8> {
    let mut bytes = vec![0_u8; 352 + voxel_bytes.len()];
    put_signed_int(&mut bytes, 0, 348, false); // sizeof_hdr
    for (index, value) in [3_i16, declared_voxels, 1, 1, 1, 1, 1, 1]
        .into_iter()
        .enumerate()
    {
        put_signed_short(&mut bytes, 40 + index * 2, value, false); // dim
    }
    put_signed_short(&mut bytes, 70, 2, false); // datatype = u8
    put_signed_short(&mut bytes, 72, 8, false); // bitpix
    put_float(&mut bytes, 76, 1.0, false); // pixdim[0] = qfac
    for (index, value) in pixdim_xyz.into_iter().enumerate() {
        put_float(&mut bytes, 80 + index * 4, value, false); // pixdim[1..3]
    }
    put_float(&mut bytes, 108, 352.0, false); // vox_offset
    bytes[123] = 2; // xyzt_units = millimetres
    put_signed_short(&mut bytes, 252, qform_code, false);
    put_signed_short(&mut bytes, 254, sform_code, false);
    // qform: identity rotation (quatern_b/c/d stay zero), RAS origin (100, 200, 300).
    for (index, value) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        put_float(&mut bytes, 268 + index * 4, value, false); // qoffset_x/y/z
    }
    // sform: diagonal (2, 3, 4) mm grid at RAS origin (10, 20, 30).
    for (index, value) in [2.0_f32, 0.0, 0.0, 10.0].into_iter().enumerate() {
        put_float(&mut bytes, 280 + index * 4, value, false); // srow_x
    }
    for (index, value) in [0.0_f32, 3.0, 0.0, 20.0].into_iter().enumerate() {
        put_float(&mut bytes, 296 + index * 4, value, false); // srow_y
    }
    for (index, value) in [0.0_f32, 0.0, 4.0, 30.0].into_iter().enumerate() {
        put_float(&mut bytes, 312 + index * 4, value, false); // srow_z
    }
    bytes[344..348].copy_from_slice(b"n+1\0"); // single-file magic
    bytes[352..].copy_from_slice(voxel_bytes);
    bytes
}

/// Parse and decode a hand-built document into its stored series.
fn decode_stored_series(bytes: &[u8]) -> StoredSeries {
    NiftiDocument::from_bytes(bytes)
        .expect("fixture parses")
        .to_stored_series()
        .expect("fixture decodes")
}

/// An active `sform` wins over an active `qform`.
///
/// The oracle is self-referential: the same fixture is read three ways —
/// sform-only, qform-only, and both active — and the both-active read must
/// equal the sform-only read. The `assert_ne!` proves the fixture actually
/// distinguishes the forms, so a passing equality cannot be vacuous.
#[test]
fn an_active_sform_takes_precedence_over_an_active_qform() {
    let voxels = [9_u8, 8];
    let pixdim = [1.0_f32, 1.0, 1.0];
    let sform_only = decode_stored_series(&raw_nifti1_spatial_document(0, 1, pixdim, &voxels, 2));
    let qform_only = decode_stored_series(&raw_nifti1_spatial_document(1, 0, pixdim, &voxels, 2));
    let both = decode_stored_series(&raw_nifti1_spatial_document(1, 1, pixdim, &voxels, 2));

    let (sform_volume, qform_volume, both_volume) = (
        &sform_only.volumes()[0],
        &qform_only.volumes()[0],
        &both.volumes()[0],
    );

    assert_ne!(
        sform_volume.metadata().origin().as_slice(),
        qform_volume.metadata().origin().as_slice(),
        "the fixture must distinguish the two forms, or precedence is untested"
    );

    assert_eq!(
        both_volume.metadata().origin().as_slice(),
        sform_volume.metadata().origin().as_slice(),
        "an active sform is authoritative for the origin"
    );
    assert_eq!(
        both_volume.metadata().spacing().to_array(),
        sform_volume.metadata().spacing().to_array(),
        "an active sform is authoritative for the spacing"
    );
    // Pinned, so a *shared* regression in the spatial mapping is still caught.
    // The sform's RAS rows are diag(2, 3, 4) at (10, 20, 30); RAS→LPS negates
    // the x and y rows, and the internal order is [depth, row, col] = [z, y, x].
    assert_eq!(both_volume.metadata().spacing().to_array(), [4.0, 3.0, 2.0]);
    assert_eq!(
        both_volume.metadata().origin().as_slice(),
        &[-10.0, -20.0, 30.0]
    );
}

/// With neither form active, geometry is the `pixdim` diagonal.
#[test]
fn absent_spatial_forms_fall_back_to_the_pixdim_diagonal() {
    let series = decode_stored_series(&raw_nifti1_spatial_document(
        0,
        0,
        [2.0, 3.0, 4.0],
        &[9, 8],
        2,
    ));
    let volume = &series.volumes()[0];

    assert_eq!(
        volume.metadata().spacing().to_array(),
        [4.0, 3.0, 2.0],
        "pixdim (2, 3, 4) reverses into [Δdepth, Δrow, Δcol]"
    );
    assert_eq!(volume.metadata().origin().as_slice(), &[0.0, 0.0, 0.0]);
}

/// A payload shorter than the declared volume is rejected before a series escapes.
///
/// The header declares two voxels and the file supplies one. `NiftiDocument`
/// validates the payload span when it parses, so the rejection is a document
/// error and `to_stored_series` is never reached — no partial series escapes.
/// `NiftiStoredReadError::TruncatedPayload` is the decode loop's defensive
/// counterpart to that check, not the primary rejection.
#[test]
fn a_payload_shorter_than_the_declared_volume_is_rejected_before_a_series_escapes() {
    let bytes = raw_nifti1_spatial_document(0, 0, [1.0, 1.0, 1.0], &[9], 2);
    let error = NiftiDocument::from_bytes(&bytes)
        .expect_err("a one-byte payload cannot satisfy two declared voxels");

    assert!(
        format!("{error}").to_lowercase().contains("payload"),
        "the rejection must name the payload, got: {error}"
    );
}

/// A non-positive `pixdim` is not a grid, so the fallback form rejects it.
#[test]
fn a_non_positive_pixdim_is_reported_as_typed_loss() {
    let document = NiftiDocument::from_bytes(&raw_nifti1_spatial_document(
        0,
        0,
        [0.0, 1.0, 1.0],
        &[9, 8],
        2,
    ))
    .expect("the header itself is well formed");
    let error = document
        .to_stored_series()
        .expect_err("a zero pixdim is not a physical grid");

    assert!(
        matches!(error, NiftiStoredReadError::Spatial(_)),
        "an unrepresentable fallback grid is spatial typed loss, got: {error}"
    );
}
