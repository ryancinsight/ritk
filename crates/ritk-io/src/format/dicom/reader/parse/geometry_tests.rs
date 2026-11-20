use super::super::types::SeriesFirstSeen;
use super::{extract_dicom_metadata, parse_ds_array, GeometryDecimalError};
use ritk_dicom::dataset::{DataElement, DicomTag, PrimitiveValue, VR};
use ritk_dicom::dataset::{Dataset, DicomFile};

fn object(attributes: &[(DicomTag, &str)]) -> DicomFile {
    let mut object = Dataset::new();
    object.put(DataElement::new(
        DicomTag::new(0x0028, 0x0101),
        VR::US,
        PrimitiveValue::from(16_u16),
    ));
    for &(tag, value) in attributes {
        object.put(DataElement::new(
            tag,
            VR::DS,
            PrimitiveValue::Strs(value.split('\\').map(str::to_owned).collect()),
        ));
    }
    object
        .with_meta(
            ritk_dicom::dataset::FileMetaBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                .media_storage_sop_instance_uid("2.25.1")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("invariant: synthetic object metadata is valid")
}

#[test]
fn malformed_components_never_shift_into_geometry() {
    for values in [
        ["3\\bad\\5", "10\\bad\\20\\30", "1\\0\\bad\\0\\0\\1\\0"],
        ["3\\5\\7", "10\\20\\30\\40", "1\\0\\0\\0\\1\\0\\9"],
        ["3", "10\\20", "1\\0\\0\\0\\1"],
        ["3\\", "10\\\\30", "1\\0\\0\\\\1\\0"],
        ["NaN\\5", "10\\NaN\\30", "1\\0\\0\\0\\NaN\\0"],
        ["3\\inf", "-inf\\20\\30", "1\\0\\0\\0\\1\\inf"],
    ] {
        let input = object(&[
            (DicomTag::new(0x0028, 0x0030), values[0]),
            (DicomTag::new(0x0020, 0x0032), values[1]),
            (DicomTag::new(0x0020, 0x0037), values[2]),
        ]);
        let mut first = SeriesFirstSeen::default();
        let (slice, _) = extract_dicom_metadata(&input, "slice.dcm".into(), &mut first)
            .expect("invariant: optional malformed geometry does not prevent metadata reading");
        assert_eq!(slice.pixel_spacing, None, "{}", values[0]);
        assert_eq!(first.pixel_spacing, None, "{}", values[0]);
        assert_eq!(slice.image_position_patient, None, "{}", values[1]);
        assert_eq!(slice.image_orientation_patient, None, "{}", values[2]);
    }
}

#[test]
fn exact_finite_geometry_preserves_values_and_later_spacing() {
    let malformed = object(&[(DicomTag::new(0x0028, 0x0030), "3\\bad\\5")]);
    let valid = object(&[
        (DicomTag::new(0x0028, 0x0030), " 3 \\ +5e0 "),
        (DicomTag::new(0x0020, 0x0032), "10\\-20\\30"),
        (DicomTag::new(0x0020, 0x0037), "1\\0\\0\\0\\1\\0"),
    ]);
    let mut first = SeriesFirstSeen::default();
    let (rejected, _) = extract_dicom_metadata(&malformed, "first.dcm".into(), &mut first)
        .expect("invariant: malformed optional geometry remains readable");
    assert_eq!(rejected.pixel_spacing, None);
    assert_eq!(first.pixel_spacing, None);
    let (accepted, _) = extract_dicom_metadata(&valid, "second.dcm".into(), &mut first)
        .expect("invariant: finite exact geometry is readable");
    assert_eq!(accepted.pixel_spacing, Some([3.0, 5.0]));
    assert_eq!(first.pixel_spacing, Some([3.0, 5.0]));
    assert_eq!(accepted.image_position_patient, Some([10.0, -20.0, 30.0]));
    assert_eq!(
        accepted.image_orientation_patient,
        Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    );
}

#[test]
fn missing_geometry_stays_absent() {
    let mut first = SeriesFirstSeen::default();
    let (slice, _) = extract_dicom_metadata(&object(&[]), "slice.dcm".into(), &mut first)
        .expect("invariant: absent optional geometry is readable");
    assert_eq!(slice.pixel_spacing, None);
    assert_eq!(first.pixel_spacing, None);
    assert_eq!(slice.image_position_patient, None);
    assert_eq!(slice.image_orientation_patient, None);
}

#[test]
fn string_and_sequence_representations_do_not_hide_bad_geometry() {
    use ritk_dicom::dataset::{Sequence, Value};

    let spacing = DicomTag::new(0x0028, 0x0030);
    let mut input = object(&[]);
    input.put(DataElement::new(
        spacing,
        VR::DS,
        PrimitiveValue::from("3\\5"),
    ));
    let mut first = SeriesFirstSeen::default();
    let (slice, _) = extract_dicom_metadata(&input, "slice.dcm".into(), &mut first)
        .expect("invariant: delimited string geometry is readable");
    assert_eq!(slice.pixel_spacing, Some([3.0, 5.0]));
    assert_eq!(first.pixel_spacing, Some([3.0, 5.0]));

    input.put(DataElement::new(
        spacing,
        VR::SQ,
        Value::Sequence(Sequence::from(vec![Dataset::new()])),
    ));
    let mut first = SeriesFirstSeen::default();
    let (slice, _) = extract_dicom_metadata(&input, "slice.dcm".into(), &mut first)
        .expect("invariant: unusable optional geometry leaves the image readable");
    assert_eq!(slice.pixel_spacing, None);
    assert_eq!(first.pixel_spacing, None);
}

#[test]
fn decimal_failures_retain_component_and_multiplicity_reasons() {
    assert_eq!(parse_ds_array::<2>("0\\3"), Ok([0.0, 3.0]));
    assert_eq!(
        parse_ds_array::<2>("3\\bad\\5"),
        Err(GeometryDecimalError::WrongMultiplicity {
            expected: 2,
            actual: 3
        })
    );
    assert_eq!(
        parse_ds_array::<2>("3\\bad"),
        Err(GeometryDecimalError::InvalidComponent { index: 1 })
    );
    assert_eq!(
        parse_ds_array::<2>("3\\"),
        Err(GeometryDecimalError::InvalidComponent { index: 1 })
    );
    assert_eq!(
        parse_ds_array::<3>("10\\20"),
        Err(GeometryDecimalError::WrongMultiplicity {
            expected: 3,
            actual: 2
        })
    );
    for value in ["NaN", "inf", "-inf"] {
        assert_eq!(
            parse_ds_array::<2>(&format!("3\\{value}")),
            Err(GeometryDecimalError::NonFiniteComponent { index: 1 })
        );
    }
}

#[test]
fn part10_scan_uses_the_first_valid_spacing_without_repairing_bad_values() {
    use super::super::scan::scan_dicom_directory;
    use super::super::tests::support::write_stub_dicom;

    for malformed in ["3\\bad\\5", "3\\5\\7", "3\\", "NaN\\5"] {
        let directory = tempfile::tempdir().expect("invariant: fixture directory is writable");
        for (filename, uid, spacing) in [
            ("first.dcm", "2.25.1", malformed),
            ("second.dcm", "2.25.2", "3\\5"),
        ] {
            let path = directory.path().join(filename);
            write_stub_dicom(&path, "1.2.840.10008.5.1.4.1.1.2", uid);
            let mut input = ritk_dicom::dataset::DicomFile::open(&path)
                .expect("invariant: generated Part-10 fixture is readable");
            for (tag, size) in [
                (DicomTag::new(0x0028, 0x0010), 2_u16),
                (DicomTag::new(0x0028, 0x0011), 3),
            ] {
                input.put(DataElement::new(tag, VR::US, PrimitiveValue::from(size)));
            }
            input.put(DataElement::new(
                DicomTag::new(0x0028, 0x0030),
                VR::DS,
                PrimitiveValue::from(spacing),
            ));
            input
                .write_to_file(&path)
                .expect("invariant: generated Part-10 fixture is writable");
        }
        let scanned = scan_dicom_directory(directory.path())
            .expect("invariant: optional malformed spacing does not reject image instances");
        let [first, second] = scanned.metadata.slices.as_slice() else {
            panic!("invariant: the fixture contains exactly two image instances");
        };
        assert_eq!(first.sop_instance_uid.as_deref(), Some("2.25.1"));
        assert_eq!(first.pixel_spacing, None, "{malformed}");
        assert_eq!(second.pixel_spacing, Some([3.0, 5.0]));
        assert_eq!(scanned.metadata.spacing, [1.0, 3.0, 5.0]);
    }
}
