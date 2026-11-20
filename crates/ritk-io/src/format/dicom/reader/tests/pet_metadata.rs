use super::super::scan::scan_dicom_directory;
use super::support::*;

#[test]
fn dicom_isotope_activity_uses_the_sop_class_unit() {
    use ritk_dicom::dataset::{DicomFile, Sequence, Value};
    use ritk_dicom::DicomTag as Tag;

    for (modality, sop_class, dose, expected_bq) in [
        (
            "PT",
            "1.2.840.10008.5.1.4.1.1.128",
            "370123456",
            Some(370_123_456.0),
        ),
        (
            "NM",
            "1.2.840.10008.5.1.4.1.1.20",
            "370.125",
            Some(370_125_000.0),
        ),
        ("PT", "1.2.840.10008.5.1.4.1.1.128", "1E999", None),
        (
            "PT",
            "1.2.840.10008.5.1.4.1.1.130",
            "370.125",
            Some(370_125_000.0),
        ),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("isotope.dcm");
        write_stub_dicom(&path, sop_class, "2.25.72001.405");
        let mut dataset = DicomFile::open(&path).unwrap();
        tracing::info_span!("arrange").in_scope(|| {
            dataset.put(DataElement::new(
                Tag::new(0x0008, 0x0060),
                VR::CS,
                PrimitiveValue::from(modality),
            ));
            dataset.put(DataElement::new(
                Tag::new(0x0054, 0x1001),
                VR::CS,
                PrimitiveValue::from("BQML"),
            ));
            dataset.put(DataElement::new(
                Tag::new(0x0010, 0x1030),
                VR::DS,
                PrimitiveValue::from("71.25"),
            ));
            dataset.put(DataElement::new(
                Tag::new(0x0028, 0x0010),
                VR::US,
                PrimitiveValue::from(4_u16),
            ));
            dataset.put(DataElement::new(
                Tag::new(0x0028, 0x0011),
                VR::US,
                PrimitiveValue::from(5_u16),
            ));
            dataset.put(DataElement::new(
                Tag::new(0x0028, 0x0030),
                VR::DS,
                PrimitiveValue::from("1.25\\2.5"),
            ));
            let mut isotope = Dataset::new();
            isotope.put(DataElement::new(
                Tag::new(0x0018, 0x1074),
                VR::DS,
                PrimitiveValue::from(dose),
            ));
            isotope.put(DataElement::new(
                Tag::new(0x0018, 0x1075),
                VR::DS,
                PrimitiveValue::from("6586.2"),
            ));
            isotope.put(DataElement::new(
                Tag::new(0x0018, 0x1076),
                VR::DS,
                PrimitiveValue::from("0.967"),
            ));
            dataset.put(DataElement::new(
                Tag::new(0x0054, 0x0016),
                VR::SQ,
                Value::from(Sequence::from(vec![isotope])),
            ));
            dataset.write_to_file(&path).unwrap();
        });

        let metadata = tracing::info_span!("act")
            .in_scope(|| scan_dicom_directory(directory.path()).unwrap().metadata);
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(
                metadata.spacing.map(aequitas::Quantity::into_base)[1..],
                [0.00125, 0.0025]
            );
            assert_eq!(
                metadata.pet_pixels_in_bq_per_ml,
                sop_class == "1.2.840.10008.5.1.4.1.1.128"
            );
            assert_eq!(metadata.patient_weight_kg.unwrap().into_base(), 71.25);
            assert_eq!(
                metadata.radionuclide_half_life_s.unwrap().into_base(),
                6586.2
            );
            assert_eq!(
                metadata
                    .radionuclide_total_dose_bq
                    .map(super::super::types::Radioactivity::becquerels),
                expected_bq
            );
        });
    }
}
