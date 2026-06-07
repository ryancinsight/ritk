use super::super::{read_dicom_seg, write_dicom_seg, DicomSegmentInfo, DicomSegmentation};
use arrayvec::ArrayString;

/// Invariant: write_dicom_seg packs BINARY frames MSB-first (inverse of unpack_pixel_data).
/// Frame 0: pixels 0-7 = 1 → byte 0 = 0xFF; pixels 8-15 = 0 → byte 1 = 0x00.
/// Frame 1: pixels 0-7 = 0 → byte 0 = 0x00; pixels 8-15 = 1 → byte 1 = 0xFF.
/// read_dicom_seg must recover the original pixel vectors exactly.
#[test]
fn test_write_dicom_seg_binary_roundtrip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("seg_write_binary.dcm");

    let seg = DicomSegmentation {
        rows: 4,
        cols: 4,
        n_frames: 2,
        bits_allocated: 1,
        segmentation_type: ArrayString::from("BINARY").unwrap(),
        segments: vec![DicomSegmentInfo {
            segment_number: 1,
            segment_label: "body".to_owned(),
            segment_description: None,
            algorithm_type: None,
        }],
        frame_segment_numbers: vec![1, 1],
        pixel_data: vec![
            {
                let mut f = vec![1u8; 8];
                f.extend(vec![0u8; 8]);
                f
            },
            {
                let mut f = vec![0u8; 8];
                f.extend(vec![1u8; 8]);
                f
            },
        ],
        image_position_per_frame: vec![None, None],
        image_orientation: None,
        pixel_spacing: None,
        slice_thickness: None,
    };

    write_dicom_seg(&path, &seg).expect("write_dicom_seg binary");
    let result = read_dicom_seg(&path).expect("read_dicom_seg binary roundtrip");

    assert_eq!(result.rows, 4, "rows");
    assert_eq!(result.cols, 4, "cols");
    assert_eq!(result.n_frames, 2, "n_frames");
    assert_eq!(result.bits_allocated, 1, "bits_allocated");
    assert_eq!(
        result.segmentation_type.as_str(),
        "BINARY",
        "segmentation_type"
    );
    assert_eq!(result.pixel_data.len(), 2, "frame count");
    let expected_f0: Vec<u8> = std::iter::repeat_n(1u8, 8)
        .chain(std::iter::repeat_n(0u8, 8))
        .collect();
    let expected_f1: Vec<u8> = std::iter::repeat_n(0u8, 8)
        .chain(std::iter::repeat_n(1u8, 8))
        .collect();
    assert_eq!(result.pixel_data[0], expected_f0, "frame 0 must match");
    assert_eq!(result.pixel_data[1], expected_f1, "frame 1 must match");
}

#[test]
fn test_write_dicom_seg_shared_functional_groups_roundtrip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("seg_write_shared_fg.dcm");

    let seg = DicomSegmentation {
        rows: 2,
        cols: 2,
        n_frames: 1,
        bits_allocated: 1,
        segmentation_type: ArrayString::from("BINARY").unwrap(),
        segments: vec![DicomSegmentInfo {
            segment_number: 1,
            segment_label: "organ".to_owned(),
            segment_description: None,
            algorithm_type: None,
        }],
        frame_segment_numbers: vec![1],
        pixel_data: vec![vec![1, 0, 0, 1]],
        image_position_per_frame: vec![Some([10.0, 20.0, 30.0])],
        image_orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        pixel_spacing: Some([0.8, 0.9]),
        slice_thickness: Some(2.2),
    };

    write_dicom_seg(&path, &seg).expect("write_dicom_seg shared fg");
    let result = read_dicom_seg(&path).expect("read_dicom_seg shared fg");

    assert_eq!(
        result.image_orientation,
        Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        "image orientation must round-trip via shared FG"
    );
    assert_eq!(
        result.pixel_spacing,
        Some([0.8, 0.9]),
        "pixel spacing must round-trip via shared FG"
    );
    assert_eq!(
        result.slice_thickness,
        Some(2.2),
        "slice thickness must round-trip via shared FG"
    );
    assert_eq!(
        result.image_position_per_frame,
        vec![Some([10.0, 20.0, 30.0])],
        "per-frame image position must round-trip"
    );
}

/// Invariant: FRACTIONAL frames are stored as raw bytes (byte-per-pixel).
/// All four pixel values [0, 128, 200, 255] must survive the write-read cycle.
#[test]
fn test_write_dicom_seg_fractional_roundtrip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("seg_write_fractional.dcm");

    let seg = DicomSegmentation {
        rows: 2,
        cols: 2,
        n_frames: 1,
        bits_allocated: 8,
        segmentation_type: ArrayString::from("FRACTIONAL").unwrap(),
        segments: vec![DicomSegmentInfo {
            segment_number: 1,
            segment_label: "prob".to_owned(),
            segment_description: None,
            algorithm_type: None,
        }],
        frame_segment_numbers: vec![1],
        pixel_data: vec![vec![0, 128, 200, 255]],
        image_position_per_frame: vec![None],
        image_orientation: None,
        pixel_spacing: None,
        slice_thickness: None,
    };

    write_dicom_seg(&path, &seg).expect("write_dicom_seg fractional");
    let result = read_dicom_seg(&path).expect("read_dicom_seg fractional roundtrip");

    assert_eq!(result.rows, 2, "rows");
    assert_eq!(result.cols, 2, "cols");
    assert_eq!(result.n_frames, 1, "n_frames");
    assert_eq!(result.pixel_data.len(), 1, "frame count");
    assert_eq!(result.pixel_data[0][0], 0u8, "pixel[0]");
    assert_eq!(result.pixel_data[0][1], 128u8, "pixel[1]");
    assert_eq!(result.pixel_data[0][2], 200u8, "pixel[2]");
    assert_eq!(result.pixel_data[0][3], 255u8, "pixel[3]");
}

/// Invariant: pixel_data.len() must equal n_frames; otherwise Err without creating a file.
#[test]
fn test_write_dicom_seg_rejects_mismatched_frame_count() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("seg_bad.dcm");

    let seg = DicomSegmentation {
        rows: 4,
        cols: 4,
        n_frames: 2,
        bits_allocated: 1,
        segmentation_type: ArrayString::from("BINARY").unwrap(),
        segments: vec![DicomSegmentInfo {
            segment_number: 1,
            segment_label: "x".to_owned(),
            segment_description: None,
            algorithm_type: None,
        }],
        frame_segment_numbers: vec![1],
        pixel_data: vec![vec![0u8; 16]],
        image_position_per_frame: vec![None],
        image_orientation: None,
        pixel_spacing: None,
        slice_thickness: None,
    };

    let result = write_dicom_seg(&path, &seg);
    assert!(result.is_err(), "mismatched frame count must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("pixel_data") || msg.contains("n_frames"),
        "error must identify the frame-count mismatch; got: {msg}"
    );
}

#[test]
fn test_write_dicom_seg_rejects_mismatched_frame_segment_numbers() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("seg_bad_frame_segment_count.dcm");

    let seg = DicomSegmentation {
        rows: 2,
        cols: 2,
        n_frames: 2,
        bits_allocated: 1,
        segmentation_type: ArrayString::from("BINARY").unwrap(),
        segments: vec![DicomSegmentInfo {
            segment_number: 1,
            segment_label: "x".to_owned(),
            segment_description: None,
            algorithm_type: None,
        }],
        frame_segment_numbers: vec![1],
        pixel_data: vec![vec![1, 0, 0, 1], vec![0, 1, 1, 0]],
        image_position_per_frame: vec![None, None],
        image_orientation: None,
        pixel_spacing: None,
        slice_thickness: None,
    };

    let result = write_dicom_seg(&path, &seg);
    assert!(
        result.is_err(),
        "expected frame segment count mismatch error"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("frame_segment_numbers") || msg.contains("n_frames"),
        "error must identify frame_segment_numbers/n_frames mismatch; got: {msg}"
    );
}
