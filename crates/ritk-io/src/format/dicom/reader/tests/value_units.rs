use super::super::scan::scan_dicom_directory;
use super::support::write_stub_dicom;
use ritk_dicom::dataset::{DicomTag, VR};

#[test]
fn dicom_voi_and_rescale_values_keep_declared_modality_units() {
    use crate::format::dicom::{DicomModalityValue, DicomValueUnit, SuvNormalization};
    use aequitas::systems::si::quantities::{ActivityConcentration, Dimensionless, MassDensity};

    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let directory = tempfile::tempdir().expect("fixture directory");
    const CT: &str = "1.2.840.10008.5.1.4.1.1.2";
    const PET: &str = "1.2.840.10008.5.1.4.1.1.128";
    for (index, (sop, image_type, rescale_type, pet_units, center, expected)) in [
        (
            CT,
            r"ORIGINAL\PRIMARY\AXIAL",
            None,
            None,
            "-400",
            DicomModalityValue::Hounsfield(Dimensionless::from_base(-400.0)),
        ),
        (
            CT,
            r"ORIGINAL\PRIMARY\LOCALIZER",
            None,
            None,
            "12",
            DicomValueUnit::Unspecified.sample(12.0),
        ),
        (
            CT,
            r"DERIVED\PRIMARY\AXIAL",
            Some("MGML"),
            None,
            "4",
            DicomModalityValue::MassConcentration(MassDensity::from_base(4.0)),
        ),
        (
            PET,
            "ORIGINAL",
            None,
            Some("BQML"),
            "8000",
            DicomModalityValue::ActivityConcentration(ActivityConcentration::from_base(8.0e9)),
        ),
        (
            PET,
            "DERIVED",
            None,
            Some("GML"),
            "2",
            DicomValueUnit::StandardizedUptake(SuvNormalization::BodyWeight).sample(2.0),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let path = tracing::info_span!("arrange", index).in_scope(|| {
            let subdir = directory.path().join(index.to_string());
            std::fs::create_dir(&subdir).expect("separate fixture series");
            let path = subdir.join("calibrated.dcm");
            write_stub_dicom(&path, sop, &format!("2.25.5800.{index}"));
            let mut file = ritk_dicom::dataset::DicomFile::open(&path).expect("open fixture");
            file.put_value(DicomTag::new(0x0008, 0x0008), VR::CS, image_type);
            if let Some(unit) = rescale_type {
                file.put_value(DicomTag::new(0x0028, 0x1054), VR::LO, unit);
            }
            if let Some(unit) = pet_units {
                file.put_value(DicomTag::new(0x0054, 0x1001), VR::CS, unit);
            }
            file.put_value(DicomTag::new(0x0028, 0x1052), VR::DS, center);
            file.put_value(
                DicomTag::new(0x0028, 0x1050),
                VR::DS,
                format!(r"{center}\1"),
            );
            file.put_value(DicomTag::new(0x0028, 0x1051), VR::DS, "-2");
            file.write_to_file(path).expect("write declared units");
            subdir
        });
        let metadata = tracing::info_span!("act", index).in_scope(|| {
            scan_dicom_directory(&path)
                .expect("scan calibrated object")
                .metadata
        });
        tracing::info_span!("assert", index).in_scope(|| {
            let slice = &metadata.slices[0];
            assert_eq!(slice.window_center, Some(expected));
            assert_eq!(slice.rescale_offset(), expected);
            assert!(slice.window_width.is_none());
            assert_eq!(metadata.pet_pixels_in_bq_per_ml, pet_units == Some("BQML"));
            assert_eq!(
                metadata.pixels_are_body_weight_suv(),
                pet_units == Some("GML")
            );
        });
    }
}
