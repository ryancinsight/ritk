use super::*;
use crate::attribute::tags;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::{DefaultDicomObject, InMemDicomObject};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientFrame};

fn object_with(
    elements: impl IntoIterator<Item = DataElement<InMemDicomObject>>,
) -> DefaultDicomObject {
    let mut object = InMemDicomObject::new_empty();
    for element in elements {
        object.put(element);
    }
    object
        .with_meta(
            dicom::object::meta::FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.4")
                .media_storage_sop_instance_uid("2.25.1")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("synthetic DICOM metadata")
}

fn diffusion_elements(weighting: f64, direction: &[f64]) -> Vec<DataElement<InMemDicomObject>> {
    vec![
        DataElement::new(
            Tag::from(tags::DIFFUSION_B_VALUE),
            VR::FD,
            PrimitiveValue::from(weighting),
        ),
        DataElement::new(
            Tag::from(tags::DIFFUSION_GRADIENT_DIRECTION),
            VR::FD,
            PrimitiveValue::F64(SmallVec::from_vec(direction.to_vec())),
        ),
    ]
}

fn write_instance(path: &Path, weighting: f64, direction: &[f64], uid: &str) -> Result<()> {
    let mut object = InMemDicomObject::new_empty();
    for element in diffusion_elements(weighting, direction) {
        object.put(element);
    }
    let object = object.with_meta(
        dicom::object::meta::FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.4")
            .media_storage_sop_instance_uid(uid)
            .transfer_syntax("1.2.840.10008.1.2.1"),
    )?;
    object.write_to_file(path)?;
    Ok(())
}

#[test]
fn extracts_standard_finite_pair() -> Result<()> {
    let object = object_with(diffusion_elements(1_000.0, &[1.0, 0.0, 0.0]));
    assert_eq!(
        extract_diffusion_pair(&object)?,
        Some((1_000.0, Vector::new([1.0, 0.0, 0.0])))
    );
    Ok(())
}

#[test]
fn absent_standard_elements_return_none() -> Result<()> {
    assert_eq!(extract_diffusion_pair(&object_with([]))?, None);
    Ok(())
}

#[test]
fn incomplete_wrong_length_and_non_finite_metadata_are_rejected() {
    let weighting_only = object_with([DataElement::new(
        Tag::from(tags::DIFFUSION_B_VALUE),
        VR::FD,
        PrimitiveValue::from(1_000.0),
    )]);
    assert!(extract_diffusion_pair(&weighting_only).is_err());

    let short = object_with(diffusion_elements(1_000.0, &[1.0, 0.0]));
    assert!(extract_diffusion_pair(&short).is_err());

    let non_finite = object_with(diffusion_elements(f64::NAN, &[1.0, 0.0, 0.0]));
    assert!(extract_diffusion_pair(&non_finite).is_err());
}

#[test]
fn zero_weighting_without_orientation_is_a_valid_b0() -> Result<()> {
    let b0 = object_with([DataElement::new(
        Tag::from(tags::DIFFUSION_B_VALUE),
        VR::FD,
        PrimitiveValue::from(0.0),
    )]);
    assert_eq!(
        extract_diffusion_pair(&b0)?,
        Some((0.0, Vector::new([0.0, 0.0, 0.0])))
    );

    let weighted = object_with([DataElement::new(
        Tag::from(tags::DIFFUSION_B_VALUE),
        VR::FD,
        PrimitiveValue::from(1_000.0),
    )]);
    assert!(extract_diffusion_pair(&weighted).is_err());
    Ok(())
}

#[test]
fn file_sequence_preserves_caller_order_units_and_lps_frame() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let b0 = directory.path().join("reference.dcm");
    let dwi = directory.path().join("weighted.dcm");
    write_instance(&b0, 0.0, &[0.0, 0.0, 0.0], "2.25.10")?;
    write_instance(&dwi, 1_000.0, &[0.0, 1.0, 0.0], "2.25.11")?;

    let scheme = read_dicom_gradient_scheme_from_files([b0, dwi])?;
    assert_eq!(scheme.frame(), GradientFrame::Lps);
    assert_eq!(scheme.len(), 2);
    assert_eq!(
        scheme.directions()[0].weighting(),
        DiffusionWeighting::from_seconds_per_square_millimeter(0.0)?
    );
    assert_eq!(
        scheme.directions()[1].direction(),
        Vector::new([0.0, 1.0, 0.0])
    );
    Ok(())
}

#[test]
fn scheme_validation_rejects_non_unit_weighted_direction() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("invalid.dcm");
    write_instance(&path, 1_000.0, &[2.0, 0.0, 0.0], "2.25.20")?;
    let error = read_dicom_gradient_scheme_from_file(path).expect_err("non-unit direction");
    assert!(error.to_string().contains("unit vector"));
    Ok(())
}
