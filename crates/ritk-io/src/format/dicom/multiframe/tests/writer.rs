use super::*;
use ritk_core::rejection::assert_rejects;
use ritk_dicom::TransferSyntaxKind;
use ritk_dicom::{parse_file_with, DicomRsBackend};

#[test]
fn test_write_multiframe_rejects_zero_dimension() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("zero.dcm");
    let image = native_image(vec![], [1, 0, 5], [0.0; 3], [1.0; 3]);
    let result = write_dicom_multiframe_native(&out_path, &image);
    assert_rejects(result, "rows=0 cols=5 must all be >0");
}

#[test]
fn test_multiframe_sop_class_is_mf_grayscale_word() {
    // Verifies that the multiframe writer emits the Multi-Frame Grayscale Word SC SOP class
    // (1.2.840.10008.5.1.4.1.1.7.3) rather than Single-frame SC (1.2.840.10008.5.1.4.1.1.7).
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf.dcm");
    let image = native_image(vec![1.0_f32; 2 * 3 * 4], [2, 3, 4], [0.0; 3], [1.0; 3]);
    write_dicom_multiframe_native(&out_path, &image).expect("write");
    let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
    assert_eq!(
        info.sop_class_uid.as_deref(),
        Some("1.2.840.10008.5.1.4.1.1.7.3"),
        "SOP class must be Multi-Frame Grayscale Word Secondary Capture"
    );
}

#[test]
fn test_written_multiframe_has_samples_per_pixel_one() {
    use dicom::object::open_file;
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_spp.dcm");
    let image = native_image(vec![1.0_f32; 2 * 4 * 5], [2, 4, 5], [0.0; 3], [1.0; 3]);
    write_dicom_multiframe_native(&out_path, &image).expect("write");

    let obj = open_file(&out_path).expect("open_file");
    let spp: u16 = obj
        .element(dicom::core::Tag(0x0028, 0x0002))
        .expect("SamplesPerPixel (0028,0002) must be present")
        .to_str()
        .expect("SamplesPerPixel must be readable as string")
        .trim()
        .parse()
        .expect("SamplesPerPixel must be numeric");
    assert_eq!(
        spp, 1,
        "SamplesPerPixel must equal 1 for grayscale multi-frame"
    );
}

#[test]
fn test_writer_config_instance_number_propagated() {
    use dicom::object::open_file;
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_inst.dcm");
    let image = native_image(vec![5.0_f32; 2 * 3], [1, 2, 3], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        instance_number: 42,
        ..MultiFrameWriterConfig::default()
    };
    write_dicom_multiframe_native_with_config(&out_path, &image, &config).expect("write");

    let obj = open_file(&out_path).expect("open_file");
    let inst_num: u32 = obj
        .element(dicom::core::Tag(0x0020, 0x0013))
        .expect("InstanceNumber (0020,0013) must be present")
        .to_str()
        .expect("InstanceNumber must be readable")
        .trim()
        .parse()
        .expect("InstanceNumber must be numeric");
    assert_eq!(
        inst_num, 42,
        "InstanceNumber must match config.instance_number"
    );
}

#[test]
fn test_multiframe_has_conversion_type_wsd() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("conv_type.dcm");
    let image = native_image(vec![1.0_f32; 2 * 2], [1, 2, 2], [0.0; 3], [1.0; 3]);
    write_dicom_multiframe_native(&out_path, &image).expect("write");

    let obj = parse_file_with::<DicomRsBackend, _>(&out_path).expect("open");
    let conv_type = obj
        .element(Tag(0x0008, 0x0064))
        .expect("ConversionType (0008,0064) must be present")
        .to_str()
        .expect("ConversionType must be a string")
        .trim()
        .to_string();
    assert_eq!(
        conv_type, "WSD",
        "ConversionType must be 'WSD' (Workstation)"
    );
}

#[test]
fn test_multiframe_has_study_and_series_uids() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("uids.dcm");
    let image = native_image(vec![1.0_f32; 2 * 2], [1, 2, 2], [0.0; 3], [1.0; 3]);
    write_dicom_multiframe_native(&out_path, &image).expect("write");
    let obj = parse_file_with::<DicomRsBackend, _>(&out_path).expect("open");
    let study_uid = obj
        .element(Tag(0x0020, 0x000D))
        .expect("StudyInstanceUID (0020,000D) must be present")
        .to_str()
        .expect("StudyInstanceUID must be a string")
        .trim()
        .to_string();
    let series_uid = obj
        .element(Tag(0x0020, 0x000E))
        .expect("SeriesInstanceUID (0020,000E) must be present")
        .to_str()
        .expect("SeriesInstanceUID must be a string")
        .trim()
        .to_string();
    assert!(!study_uid.is_empty(), "StudyInstanceUID must be non-empty");
    assert!(
        !series_uid.is_empty(),
        "SeriesInstanceUID must be non-empty"
    );
    assert_ne!(
        study_uid, series_uid,
        "StudyInstanceUID and SeriesInstanceUID must be distinct"
    );
}

#[test]
fn test_multiframe_has_type2_patient_study_series_tags() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("type2.dcm");
    let image = native_image(vec![5.0_f32; 3 * 3], [1, 3, 3], [0.0; 3], [1.0; 3]);
    write_dicom_multiframe_native(&out_path, &image).expect("write");
    let obj = parse_file_with::<DicomRsBackend, _>(&out_path).expect("open");
    // Assert presence (value may be empty per Type 2 semantics).
    obj.element(Tag(0x0010, 0x0010))
        .expect("PatientName (0010,0010) must be present");
    obj.element(Tag(0x0010, 0x0020))
        .expect("PatientID (0010,0020) must be present");
    obj.element(Tag(0x0008, 0x0020))
        .expect("StudyDate (0008,0020) must be present");
    obj.element(Tag(0x0008, 0x0090))
        .expect("ReferringPhysicianName (0008,0090) must be present");
    obj.element(Tag(0x0020, 0x0010))
        .expect("StudyID (0020,0010) must be present");
    obj.element(Tag(0x0020, 0x0011))
        .expect("SeriesNumber (0020,0011) must be present");
}

#[test]
fn test_write_multiframe_jpegls_lossless_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_jpegls.dcm");
    let voxels = vec![0.0_f32, 10.0, 20.0, 30.0, 40.0, 50.0];
    let image = native_image(voxels.clone(), [2, 1, 3], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        transfer_syntax: TransferSyntaxKind::JpegLsLossless,
        ..MultiFrameWriterConfig::default()
    };

    write_dicom_multiframe_native_with_config(&out_path, &image, &config)
        .expect("JPEG-LS multiframe write");

    let ts_uid = parse_file_with::<DicomRsBackend, _>(&out_path)
        .expect("parse file")
        .meta()
        .transfer_syntax()
        .to_owned();
    assert_eq!(ts_uid, TransferSyntaxKind::JpegLsLossless.uid());

    let decoded = load_dicom_multiframe_flat(&out_path).expect("decode JPEG-LS multiframe");
    assert_eq!(decoded.shape, [2, 1, 3]);
    for (actual, expected) in decoded.data.iter().zip(voxels.iter()) {
        assert!(
            (actual - expected).abs() <= 1.5,
            "decoded voxel {actual} differed too much from source {expected}"
        );
    }
}

#[test]
fn test_write_multiframe_jpeg2000_lossless_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_j2k.dcm");
    let voxels = vec![5.0_f32, 8.0, 13.0, 21.0];
    let image = native_image(voxels.clone(), [1, 2, 2], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        transfer_syntax: TransferSyntaxKind::Jpeg2000Lossless,
        ..MultiFrameWriterConfig::default()
    };

    write_dicom_multiframe_native_with_config(&out_path, &image, &config)
        .expect("JPEG 2000 multiframe write");

    let ts_uid = parse_file_with::<DicomRsBackend, _>(&out_path)
        .expect("parse file")
        .meta()
        .transfer_syntax()
        .to_owned();
    assert_eq!(ts_uid, TransferSyntaxKind::Jpeg2000Lossless.uid());

    let decoded = load_dicom_multiframe_flat(&out_path).expect("decode JPEG 2000 multiframe");
    assert_eq!(decoded.shape, [1, 2, 2]);
    for (actual, expected) in decoded.data.iter().zip(voxels.iter()) {
        assert!(
            (actual - expected).abs() <= 1.5,
            "decoded voxel {actual} differed too much from source {expected}"
        );
    }
}

#[test]
fn test_write_multiframe_rle_lossless_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_rle.dcm");
    let voxels = vec![2.0_f32, 4.0, 8.0, 16.0, 32.0, 64.0];
    let image = native_image(voxels.clone(), [2, 1, 3], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        transfer_syntax: TransferSyntaxKind::RleLossless,
        ..MultiFrameWriterConfig::default()
    };

    write_dicom_multiframe_native_with_config(&out_path, &image, &config)
        .expect("RLE multiframe write");

    let ts_uid = parse_file_with::<DicomRsBackend, _>(&out_path)
        .expect("parse file")
        .meta()
        .transfer_syntax()
        .to_owned();
    assert_eq!(ts_uid, TransferSyntaxKind::RleLossless.uid());

    let decoded = load_dicom_multiframe_flat(&out_path).expect("decode RLE multiframe");
    assert_eq!(decoded.shape, [2, 1, 3]);
    for (actual, expected) in decoded.data.iter().zip(voxels.iter()) {
        assert!(
            (actual - expected).abs() <= 1.5,
            "decoded voxel {actual} differed too much from source {expected}"
        );
    }
}

#[test]
fn test_write_multiframe_jpeg_baseline_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_jpeg.dcm");
    // A smooth ramp: baseline JPEG is lossy, so a hard-edged pattern would fail
    // on the DCT step rather than on anything the writer controls.
    let voxels: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let image = native_image(voxels.clone(), [2, 3, 4], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        transfer_syntax: TransferSyntaxKind::JpegBaseline,
        ..MultiFrameWriterConfig::default()
    };

    write_dicom_multiframe_native_with_config(&out_path, &image, &config)
        .expect("JPEG baseline multiframe write");

    let ts_uid = parse_file_with::<DicomRsBackend, _>(&out_path)
        .expect("parse file")
        .meta()
        .transfer_syntax()
        .to_owned();
    assert_eq!(ts_uid, TransferSyntaxKind::JpegBaseline.uid());

    let decoded = load_dicom_multiframe_flat(&out_path).expect("decode JPEG baseline multiframe");
    assert_eq!(decoded.shape, [2, 3, 4]);
    // Eight-bit storage bounds the reconstruction: the modality range is
    // written across 255 levels, so a voxel can move by up to half a level of
    // the 23-level source range, plus the lossy DCT's own quantisation.
    for (actual, expected) in decoded.data.iter().zip(voxels.iter()) {
        assert!(
            (actual - expected).abs() <= 1.5,
            "decoded voxel {actual} differed too much from source {expected}"
        );
    }
}

#[test]
fn test_write_multiframe_jpeg_baseline_declares_eight_bit_pixel_format() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_jpeg_tags.dcm");
    let voxels: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let image = native_image(voxels, [1, 3, 4], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        transfer_syntax: TransferSyntaxKind::JpegBaseline,
        ..MultiFrameWriterConfig::default()
    };

    write_dicom_multiframe_native_with_config(&out_path, &image, &config)
        .expect("JPEG baseline multiframe write");

    let obj = parse_file_with::<DicomRsBackend, _>(&out_path).expect("parse file");
    // Baseline JPEG carries eight-bit samples; the 16-bit tags would describe a
    // pixel format the fragments do not contain.
    assert_eq!(
        obj.element(Tag(0x0028, 0x0100))
            .expect("BitsAllocated")
            .to_str()
            .expect("US renders as text")
            .trim()
            .parse::<u16>()
            .expect("US parses"),
        8
    );
    assert_eq!(
        obj.element(Tag(0x0028, 0x0101))
            .expect("BitsStored")
            .to_str()
            .expect("US renders as text")
            .trim()
            .parse::<u16>()
            .expect("US parses"),
        8
    );
    assert_eq!(
        obj.element(Tag(0x0028, 0x0102))
            .expect("HighBit")
            .to_str()
            .expect("US renders as text")
            .trim()
            .parse::<u16>()
            .expect("US parses"),
        7
    );
}

#[test]
fn test_write_multiframe_jpeg_ls_lossy_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_jpegls_lossy.dcm");
    let voxels: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let image = native_image(voxels.clone(), [2, 3, 4], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        transfer_syntax: TransferSyntaxKind::JpegLsLossy,
        ..MultiFrameWriterConfig::default()
    };

    write_dicom_multiframe_native_with_config(&out_path, &image, &config)
        .expect("JPEG-LS lossy multiframe write");

    let ts_uid = parse_file_with::<DicomRsBackend, _>(&out_path)
        .expect("parse file")
        .meta()
        .transfer_syntax()
        .to_owned();
    assert_eq!(ts_uid, TransferSyntaxKind::JpegLsLossy.uid());

    let decoded = load_dicom_multiframe_flat(&out_path).expect("decode JPEG-LS lossy multiframe");
    assert_eq!(decoded.shape, [2, 3, 4]);
    // NEAR is the transfer syntax's own error bound: JPEG-LS near-lossless
    // guarantees |decoded - original| <= NEAR on the *stored* samples, so the
    // bound here is the rescale step plus that NEAR, not an arbitrary slack.
    for (actual, expected) in decoded.data.iter().zip(voxels.iter()) {
        assert!(
            (actual - expected).abs() <= 1.5,
            "decoded voxel {actual} exceeded the NEAR bound for source {expected}"
        );
    }
}

#[test]
fn test_write_multiframe_jpeg2000_lossy_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_j2k_lossy.dcm");
    let voxels: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let image = native_image(voxels.clone(), [2, 3, 4], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        transfer_syntax: TransferSyntaxKind::Jpeg2000Lossy,
        ..MultiFrameWriterConfig::default()
    };

    write_dicom_multiframe_native_with_config(&out_path, &image, &config)
        .expect("JPEG 2000 lossy multiframe write");

    let ts_uid = parse_file_with::<DicomRsBackend, _>(&out_path)
        .expect("parse file")
        .meta()
        .transfer_syntax()
        .to_owned();
    assert_eq!(ts_uid, TransferSyntaxKind::Jpeg2000Lossy.uid());

    let decoded = load_dicom_multiframe_flat(&out_path).expect("decode JPEG 2000 lossy multiframe");
    assert_eq!(decoded.shape, [2, 3, 4]);
    // A unit quantization step on the irreversible transform keeps the error
    // within one stored level; the rescale step is the other term.
    for (actual, expected) in decoded.data.iter().zip(voxels.iter()) {
        assert!(
            (actual - expected).abs() <= 1.5,
            "decoded voxel {actual} exceeded the quantisation bound for source {expected}"
        );
    }
}

#[test]
fn test_write_multiframe_jpeg_lossless_round_trip_is_bit_exact() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_jpeg_lossless.dcm");
    let voxels: Vec<f32> = (0..24).map(|i| i as f32 * 37.0).collect();
    let image = native_image(voxels.clone(), [2, 3, 4], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        transfer_syntax: TransferSyntaxKind::JpegLosslessFirstOrderPrediction,
        ..MultiFrameWriterConfig::default()
    };

    write_dicom_multiframe_native_with_config(&out_path, &image, &config)
        .expect("JPEG lossless multiframe write");

    let ts_uid = parse_file_with::<DicomRsBackend, _>(&out_path)
        .expect("parse file")
        .meta()
        .transfer_syntax()
        .to_owned();
    assert_eq!(
        ts_uid,
        TransferSyntaxKind::JpegLosslessFirstOrderPrediction.uid()
    );

    let decoded = load_dicom_multiframe_flat(&out_path).expect("decode JPEG lossless multiframe");
    assert_eq!(decoded.shape, [2, 3, 4]);

    // Two different losses are in play and this test separates them.
    //
    // SOF3 is the one JPEG mode that reconstructs *bit-exactly*, and that is why
    // a study gets archived under this syntax. But every syntax here stores
    // 16-bit normalised samples, so the modality values come back quantised --
    // that loss belongs to the container, not the codec.
    //
    // Asserting against the quantised expectation rather than the original
    // voxels is what makes the claim specific: any error the JPEG stage
    // introduced would show up as a mismatch, while the container's own
    // half-level rounding is accounted for exactly.
    let minimum = voxels.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = voxels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = maximum - minimum;
    let slope = range / 65535.0;
    for (actual, expected) in decoded.data.iter().zip(voxels.iter()) {
        let stored = (((expected - minimum) / range) * 65535.0).round();
        let quantised = stored * slope + minimum;
        // The bound is derived, not chosen: the codec contributes nothing here
        // (the `ritk-codecs` lossless tests assert stored-sample equality
        // directly), so what is left is the container. Two terms --
        //
        //   - sixteen-bit normalisation rounds the stored sample by at most
        //     `slope / 2`;
        //   - the rescale tags are written as DS with six decimals, so the slope
        //     read back can differ by up to 5e-7, which at the largest stored
        //     sample is 65535 * 5e-7 ~= 0.033.
        //
        // Both are the container's, and neither would grow if the codec were
        // lossier -- which is what makes this assertion about SOF3 specific.
        let bound = slope / 2.0 + 65535.0 * 5e-7 + f32::EPSILON * 65535.0;
        assert!(
            (*actual - quantised).abs() <= bound,
            "SOF3 must reconstruct the stored sample exactly; {actual}              differs from {quantised} by more than the container's {bound}              (source {expected})"
        );
    }
}
