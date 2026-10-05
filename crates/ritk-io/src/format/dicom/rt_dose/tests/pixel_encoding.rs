use super::super::{write_rt_dose, RtDoseGrid, RtDoseSummationType, RtDoseType};
use crate::format::dicom::DicomWriteError;
use dicom::core::Tag;

fn grid() -> RtDoseGrid {
    RtDoseGrid {
        rows: 1,
        cols: 3,
        n_frames: 1,
        dose_type: RtDoseType::Physical,
        dose_summation_type: RtDoseSummationType::Plan,
        dose_grid_scaling: 0.5,
        frame_offsets: vec![0.0],
        dose_gy: vec![0.0, 1.0, 2.0],
        image_position: None,
        image_orientation: None,
        pixel_spacing: None,
        referenced_rt_plan_sop_instance_uid: None,
    }
}

#[test]
fn tags_describe_dose_payload() {
    let directory = tempfile::tempdir().expect("invariant: prepared fixture is valid");
    let path = directory.path().join("dose.dcm");
    write_rt_dose(&path, &grid()).expect("invariant: prepared fixture is valid");
    let object = dicom::object::open_file(&path).expect("invariant: prepared fixture is valid");
    for (tag, expected) in [(0x0100, 32), (0x0101, 32), (0x0102, 31), (0x0103, 0)] {
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
    assert_eq!(payload.len() * 8, 3 * width);
    assert_eq!(payload.as_ref(), &[0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0]);
}

#[test]
fn invalid_dose_preserves_output() {
    let directory = tempfile::tempdir().expect("invariant: prepared fixture is valid");
    let path = directory.path().join("dose.dcm");
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, f64::MAX] {
        std::fs::write(&path, b"existing dose").expect("invariant: prepared fixture is valid");
        let mut dose = grid();
        dose.dose_gy[2] = value;
        let error =
            write_rt_dose(&path, &dose).expect_err("invariant: malformed fixture is rejected");
        let expected = if value.is_finite() {
            DicomWriteError::EncodedPixelOutOfRange { index: 2 }
        } else {
            DicomWriteError::NonFinitePixel { index: 2 }
        };
        assert_eq!(error.downcast_ref::<DicomWriteError>(), Some(&expected));
        assert_eq!(
            std::fs::read(&path).expect("invariant: prepared fixture is valid"),
            b"existing dose"
        );
    }
    let mut dose = grid();
    dose.frame_offsets[0] = f64::NAN;
    let error = write_rt_dose(&path, &dose).expect_err("invariant: malformed fixture is rejected");
    assert_eq!(
        error.downcast_ref::<DicomWriteError>(),
        Some(&DicomWriteError::InvalidSpatialMetadata)
    );
    assert_eq!(
        std::fs::read(&path).expect("invariant: prepared fixture is valid"),
        b"existing dose"
    );
}
