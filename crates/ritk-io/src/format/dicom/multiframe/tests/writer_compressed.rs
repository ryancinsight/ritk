//! Compression transfer-syntax round-trips.
//!
//! Split from `writer.rs` because the compression matrix -- JPEG baseline,
//! JPEG-LS lossless and near-lossless, JPEG 2000 lossless and lossy, and both
//! JPEG lossless predictor modes -- is one concern with its own fixtures, and
//! keeping it beside the uncompressed writer tests pushed the module past the
//! size the scanner measures.

use super::*;
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;
use ritk_dicom::{parse_file_with, DicomRsBackend, TransferSyntaxKind};

fn assert_pixel_format(object: &DefaultDicomObject, bits_allocated: u16) {
    for (tag, expected) in [
        (Tag(0x0028, 0x0100), bits_allocated),
        (Tag(0x0028, 0x0101), bits_allocated),
        (Tag(0x0028, 0x0102), bits_allocated - 1),
        (Tag(0x0028, 0x0103), 0),
    ] {
        let actual = object
            .element(tag)
            .expect("pixel attribute must exist")
            .to_str()
            .expect("US attribute must render")
            .trim()
            .parse::<u16>()
            .expect("US attribute must parse");
        assert_eq!(actual, expected, "pixel attribute {tag}");
    }
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

    let object = parse_file_with::<DicomRsBackend, _>(&out_path).expect("parse file");
    assert_pixel_format(&object, 16);
    let ts_uid = object.meta().transfer_syntax().to_owned();
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
    assert_pixel_format(&obj, 8);
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
    assert_eq!(
        obj.element(Tag(0x0028, 0x0103))
            .expect("PixelRepresentation")
            .to_str()
            .expect("US renders as text")
            .trim()
            .parse::<u16>()
            .expect("US parses"),
        0
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

#[test]
fn test_write_multiframe_jpeg_lossless_non_hierarchical_round_trip() {
    // `Ss = 0`, the DICOM selector T.81's table does not define and that
    // consus-raster now accepts. This proves the whole path: writer -> fragment ->
    // provider decode, not just the codec in isolation.
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_jpeg_nh.dcm");
    let voxels: Vec<f32> = (0..24).map(|i| i as f32 * 53.0).collect();
    let image = native_image(voxels.clone(), [2, 3, 4], [0.0; 3], [1.0; 3]);
    let config = MultiFrameWriterConfig {
        transfer_syntax: TransferSyntaxKind::JpegLosslessNonHierarchical,
        ..MultiFrameWriterConfig::default()
    };

    write_dicom_multiframe_native_with_config(&out_path, &image, &config)
        .expect("JPEG lossless non-hierarchical multiframe write");

    let ts_uid = parse_file_with::<DicomRsBackend, _>(&out_path)
        .expect("parse file")
        .meta()
        .transfer_syntax()
        .to_owned();
    assert_eq!(
        ts_uid,
        TransferSyntaxKind::JpegLosslessNonHierarchical.uid()
    );

    let decoded =
        load_dicom_multiframe_flat(&out_path).expect("decode non-hierarchical multiframe");
    assert_eq!(decoded.shape, [2, 3, 4]);
    // SOF3 reconstructs bit-exactly; what is left is the container's own
    // sixteen-bit normalisation and its six-decimal rescale, exactly as in the
    // first-order sibling test.
    let minimum = voxels.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = voxels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = maximum - minimum;
    let bound = (range / 65535.0) / 2.0 + 65535.0 * 5e-7 + f32::EPSILON * 65535.0;
    for (actual, expected) in decoded.data.iter().zip(voxels.iter()) {
        let stored = (((expected - minimum) / range) * 65535.0).round();
        let quantised = stored * (range / 65535.0) + minimum;
        assert!(
            (*actual - quantised).abs() <= bound,
            "non-hierarchical lossless must reconstruct the stored sample; \
             {actual} differs from {quantised} by more than {bound}"
        );
    }
}
