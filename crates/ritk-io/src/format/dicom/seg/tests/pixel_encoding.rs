use super::super::{read_dicom_seg, write_dicom_seg, DicomSegmentation, SegmentationType};
use crate::format::dicom::DicomWriteError;
use dicom::core::Tag;

fn segmentation(kind: SegmentationType, bits: u16, frames: Vec<Vec<u8>>) -> DicomSegmentation {
    let count = frames.len();
    DicomSegmentation {
        rows: 1,
        cols: 3,
        n_frames: count,
        bits_allocated: bits,
        segmentation_type: kind,
        segments: Vec::new(),
        frame_segment_numbers: vec![1; count],
        pixel_data: frames,
        image_position_per_frame: vec![None; count],
        image_orientation: None,
        pixel_spacing: None,
        slice_thickness: None,
    }
}

#[test]
fn tags_describe_packed_and_fractional_payloads() {
    let directory = tempfile::tempdir().expect("invariant: prepared fixture is valid");
    let path = directory.path().join("seg.dcm");
    // PS3.5 D.1 and 8.1.1: sample i occupies bit i%8, consecutive frames
    // share the byte containing their boundary, and only the value is padded.
    for (kind, bits, frames, expected) in [
        (
            SegmentationType::Binary,
            1,
            vec![vec![1, 0, 1], vec![0, 1, 1], vec![1, 0, 1]],
            vec![0x75, 0x01],
        ),
        (
            SegmentationType::Fractional,
            8,
            vec![vec![0, 127, 255]],
            vec![0, 127, 255, 0],
        ),
    ] {
        let seg = segmentation(kind, bits, frames);
        write_dicom_seg(&path, &seg).expect("invariant: prepared fixture is valid");
        let object = dicom::object::open_file(&path).expect("invariant: prepared fixture is valid");
        for (tag, expected) in [
            (0x0100, bits),
            (0x0101, bits),
            (0x0102, bits - 1),
            (0x0103, 0),
        ] {
            assert_eq!(
                object
                    .element(Tag(0x0028, tag))
                    .expect("invariant: prepared fixture is valid")
                    .to_int::<u16>()
                    .expect("invariant: prepared fixture is valid"),
                expected
            );
        }
        let payload = object
            .element(Tag(0x7FE0, 0x0010))
            .expect("invariant: prepared fixture is valid")
            .to_bytes()
            .expect("invariant: prepared fixture is valid");
        let width = object
            .element(Tag(0x0028, 0x0100))
            .expect("invariant: prepared fixture is valid")
            .to_int::<usize>()
            .expect("invariant: prepared fixture is valid");
        let bytes = (seg.n_frames * seg.rows * seg.cols * width).div_ceil(8);
        assert_eq!(payload.len(), bytes + bytes % 2);
        assert_eq!(payload.as_ref(), expected);
        assert_eq!(
            read_dicom_seg(&path)
                .expect("invariant: prepared fixture is valid")
                .pixel_data,
            seg.pixel_data
        );
    }
}

#[test]
fn invalid_segmentation_preserves_output() {
    let directory = tempfile::tempdir().expect("invariant: prepared fixture is valid");
    let path = directory.path().join("seg.dcm");
    std::fs::write(&path, b"existing segmentation").expect("invariant: prepared fixture is valid");
    let mut seg = segmentation(SegmentationType::Binary, 1, vec![vec![0, 1, 2]]);
    let error = write_dicom_seg(&path, &seg).expect_err("invariant: malformed fixture is rejected");
    assert_eq!(
        error.downcast_ref::<DicomWriteError>(),
        Some(&DicomWriteError::InvalidBinaryPixel { index: 2 })
    );
    assert_eq!(
        std::fs::read(&path).expect("invariant: prepared fixture is valid"),
        b"existing segmentation"
    );
    seg.pixel_data[0][2] = 0;
    seg.bits_allocated = 8;
    let error = write_dicom_seg(&path, &seg).expect_err("invariant: malformed fixture is rejected");
    assert_eq!(
        error.downcast_ref::<DicomWriteError>(),
        Some(&DicomWriteError::PixelDescriptionMismatch {
            declared: 8,
            encoded: 1
        })
    );
    assert_eq!(
        std::fs::read(&path).expect("invariant: prepared fixture is valid"),
        b"existing segmentation"
    );
    seg.bits_allocated = 1;
    seg.image_position_per_frame[0] = Some([0.0, f64::INFINITY, 0.0]);
    let error = write_dicom_seg(&path, &seg).expect_err("invariant: malformed fixture is rejected");
    assert_eq!(
        error.downcast_ref::<DicomWriteError>(),
        Some(&DicomWriteError::InvalidSpatialMetadata)
    );
    assert_eq!(
        std::fs::read(&path).expect("invariant: prepared fixture is valid"),
        b"existing segmentation"
    );
}
