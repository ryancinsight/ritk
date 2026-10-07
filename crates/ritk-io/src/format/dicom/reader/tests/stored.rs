#![expect(clippy::unwrap_used, reason = "DICOM Part 10 fixture setup")]

use std::path::Path;

use super::super::{load_dicom_stored_series, scan_dicom_path, StoredDicomError};
use super::support::*;
use dicom::core::value::DataSetSequence;
use dicom::core::Length;
use ritk_codecs::{ByteOrder, SampleType};
use ritk_image_io::IntensityCalibration;
use ritk_spatial::CoordinateMap;

#[derive(Clone, Copy)]
struct PixelEncoding {
    rows: u16,
    columns: u16,
    bits_allocated: u16,
    bits_stored: u16,
    high_bit: u16,
    orientation: Option<&'static str>,
    gantry_tilt: Option<&'static str>,
    slice_thickness: Option<&'static str>,
    spacing_between_slices: Option<&'static str>,
}

impl PixelEncoding {
    const DEFAULT: Self = Self {
        rows: 1,
        columns: 2,
        bits_allocated: 16,
        bits_stored: 12,
        high_bit: 11,
        orientation: Some("1\\0\\0\\0\\1\\0"),
        gantry_tilt: None,
        slice_thickness: Some("1"),
        spacing_between_slices: None,
    };
}

enum CalibrationFixture<'a> {
    Linear {
        slope: &'a str,
        intercept: &'a str,
        rescale_type: Option<&'a str>,
    },
    ModalityLookup {
        entry_count: i64,
        first_mapped_value: i64,
        output_bits: i64,
        entries: &'a [u16],
        signed_descriptor: bool,
        unit: Option<&'a str>,
    },
}

fn write_slice(
    path: &Path,
    instance: u16,
    z: &str,
    pixel_representation: u16,
    samples_per_pixel: u16,
    photometric: &str,
    calibration: CalibrationFixture<'_>,
    pixel_bytes: Vec<u8>,
) {
    write_slice_with_encoding(
        path,
        instance,
        z,
        pixel_representation,
        samples_per_pixel,
        photometric,
        calibration,
        pixel_bytes,
        PixelEncoding::DEFAULT,
    );
}

fn write_slice_with_encoding(
    path: &Path,
    instance: u16,
    z: &str,
    pixel_representation: u16,
    samples_per_pixel: u16,
    photometric: &str,
    calibration: CalibrationFixture<'_>,
    pixel_bytes: Vec<u8>,
    encoding: PixelEncoding,
) {
    use dicom::core::smallvec::SmallVec;

    const CT: &str = "1.2.840.10008.5.1.4.1.1.2";
    let mut object = InMemDicomObject::new_empty();
    for (tag, vr, value) in [
        (Tag(0x0008, 0x0016), VR::UI, CT.to_owned()),
        (
            Tag(0x0008, 0x0018),
            VR::UI,
            format!("2.25.88001.{instance}"),
        ),
        (Tag(0x0008, 0x0060), VR::CS, "CT".to_owned()),
        (Tag(0x0020, 0x000D), VR::UI, "2.25.88001".to_owned()),
        (Tag(0x0020, 0x000E), VR::UI, "2.25.88002".to_owned()),
        (Tag(0x0020, 0x0013), VR::IS, instance.to_string()),
        (Tag(0x0020, 0x0032), VR::DS, format!("10\\20\\{z}")),
        (Tag(0x0028, 0x0004), VR::CS, photometric.to_owned()),
        (Tag(0x0028, 0x0030), VR::DS, "0.5\\0.25".to_owned()),
    ] {
        object.put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
    }
    if let Some(orientation) = encoding.orientation {
        object.put(DataElement::new(
            Tag(0x0020, 0x0037),
            VR::DS,
            PrimitiveValue::from(orientation),
        ));
    }
    if let Some(tilt) = encoding.gantry_tilt {
        object.put(DataElement::new(
            Tag(0x0018, 0x1120),
            VR::DS,
            PrimitiveValue::from(tilt),
        ));
    }
    for (tag, value) in [
        (Tag(0x0018, 0x0050), encoding.slice_thickness),
        (Tag(0x0018, 0x0088), encoding.spacing_between_slices),
    ] {
        if let Some(value) = value {
            object.put(DataElement::new(tag, VR::DS, PrimitiveValue::from(value)));
        }
    }
    for (tag, value) in [
        (Tag(0x0028, 0x0002), samples_per_pixel),
        (Tag(0x0028, 0x0010), encoding.rows),
        (Tag(0x0028, 0x0011), encoding.columns),
        (Tag(0x0028, 0x0100), encoding.bits_allocated),
        (Tag(0x0028, 0x0101), encoding.bits_stored),
        (Tag(0x0028, 0x0102), encoding.high_bit),
        (Tag(0x0028, 0x0103), pixel_representation),
    ] {
        object.put(DataElement::new(tag, VR::US, PrimitiveValue::from(value)));
    }
    match calibration {
        CalibrationFixture::Linear {
            slope,
            intercept,
            rescale_type,
        } => {
            if let Some(rescale_type) = rescale_type {
                object.put(DataElement::new(
                    Tag(0x0028, 0x1054),
                    VR::LO,
                    PrimitiveValue::from(rescale_type),
                ));
            }
            object.put(DataElement::new(
                Tag(0x0028, 0x1052),
                VR::DS,
                PrimitiveValue::from(intercept),
            ));
            object.put(DataElement::new(
                Tag(0x0028, 0x1053),
                VR::DS,
                PrimitiveValue::from(slope),
            ));
        }
        CalibrationFixture::ModalityLookup {
            entry_count,
            first_mapped_value,
            output_bits,
            entries,
            signed_descriptor,
            unit,
        } => {
            let mut item = InMemDicomObject::new_empty();
            if signed_descriptor {
                let entry_count = u16::try_from(entry_count)
                    .expect("fixture LUT count fits unsigned descriptor bits");
                item.put(DataElement::new(
                    Tag(0x0028, 0x3002),
                    VR::SS,
                    PrimitiveValue::I16(SmallVec::from_vec(vec![
                        i16::from_le_bytes(entry_count.to_le_bytes()),
                        i16::try_from(first_mapped_value)
                            .expect("signed fixture LUT origin fits i16"),
                        i16::try_from(output_bits).expect("fixture LUT precision fits i16"),
                    ])),
                ));
            } else {
                item.put(DataElement::new(
                    Tag(0x0028, 0x3002),
                    VR::US,
                    PrimitiveValue::U16(SmallVec::from_vec(vec![
                        u16::try_from(entry_count).expect("unsigned fixture count fits u16"),
                        u16::try_from(first_mapped_value)
                            .expect("unsigned fixture starts at a non-negative pixel"),
                        u16::try_from(output_bits).expect("unsigned fixture precision fits u16"),
                    ])),
                ));
            }
            item.put(DataElement::new(
                Tag(0x0028, 0x3006),
                VR::OW,
                PrimitiveValue::U16(SmallVec::from_vec(entries.to_vec())),
            ));
            if let Some(unit) = unit {
                item.put(DataElement::new(
                    Tag(0x0028, 0x3004),
                    VR::LO,
                    PrimitiveValue::from(unit),
                ));
            }
            object.put(DataElement::new(
                Tag(0x0028, 0x3000),
                VR::SQ,
                DataSetSequence::new(vec![item], Length::UNDEFINED),
            ));
        }
    }
    object.put(DataElement::new(
        Tag(0x7fe0, 0x0010),
        if encoding.bits_allocated == 8 {
            VR::OB
        } else {
            VR::OW
        },
        PrimitiveValue::U8(SmallVec::from_vec(pixel_bytes)),
    ));
    let file = object
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid(CT)
                .media_storage_sop_instance_uid(format!("2.25.88001.{instance}"))
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("valid Part 10 metadata");
    file.write_to_file(path).expect("write DICOM fixture");
}

fn write_signed_series(directory: &Path) {
    write_slice(
        &directory.join("one.dcm"),
        1,
        "0",
        1,
        1,
        "MONOCHROME2",
        CalibrationFixture::Linear {
            slope: "2.5",
            intercept: "-100",
            rescale_type: Some("HU"),
        },
        vec![0xff, 0xff, 0x00, 0xf8],
    );
    write_slice(
        &directory.join("two.dcm"),
        2,
        "2",
        1,
        1,
        "MONOCHROME2",
        CalibrationFixture::Linear {
            slope: "0.5",
            intercept: "4",
            rescale_type: Some("HU"),
        },
        vec![0x01, 0xf8, 0x02, 0xf0],
    );
}

#[test]
fn stored_reader_retains_signed_samples_calibration_and_slice_geometry() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_signed_series(directory.path());
    let scanned = scan_dicom_path(directory.path()).expect("scan selected series");

    let (series, metadata) = load_dicom_stored_series(scanned).expect("stored series loads");
    assert_eq!(series.volumes().len(), 1);
    let volume = &series.volumes()[0];
    assert_eq!(volume.shape(), [2, 1, 2]);
    assert_eq!(volume.samples().sample_type(), SampleType::I16);
    assert_eq!(
        volume
            .samples()
            .encode(ByteOrder::LeastSignificantByteFirst)
            .expect("stored sample bytes"),
        [0xff, 0xff, 0x00, 0xf8, 0x01, 0xf8, 0x02, 0x00]
    );
    let IntensityCalibration::PerFrameLinear(calibrations) = volume.calibration() else {
        panic!("one linear transform must be retained per slice");
    };
    assert_eq!(calibrations.len(), 2);
    assert_eq!(calibrations[0].slope(), 2.5);
    assert_eq!(calibrations[0].intercept(), -100.0);
    assert_eq!(calibrations[1].slope(), 0.5);
    assert_eq!(calibrations[1].intercept(), 4.0);
    assert_eq!(
        volume.intensity_unit().map(|unit| unit.as_str()),
        Some("HU")
    );
    let CoordinateMap::SliceSeries(coordinates) = volume.coordinate_map() else {
        panic!("per-slice DICOM positions must remain represented");
    };
    assert_eq!(
        coordinates.world_from_index(1.0, 0.0, 1.0),
        [10.25, 20.0, 2.0]
    );
    assert!(metadata
        .slices
        .iter()
        .all(|slice| slice.part10_bytes.is_none()));
}

#[test]
fn stored_reader_rejects_invalid_pixel_representation() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice(
        &directory.path().join("invalid.dcm"),
        1,
        "0",
        2,
        1,
        "MONOCHROME2",
        CalibrationFixture::Linear {
            slope: "1",
            intercept: "0",
            rescale_type: Some("HU"),
        },
        vec![0, 0, 0, 0],
    );
    let scanned = scan_dicom_path(directory.path()).expect("scanner preserves the selected series");
    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::InvalidTag {
            tag: "PixelRepresentation (0028,0103)"
        })
    ));
}

#[test]
fn stored_reader_rejects_color_before_building_a_volume() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice(
        &directory.path().join("color.dcm"),
        1,
        "0",
        0,
        3,
        "RGB",
        CalibrationFixture::Linear {
            slope: "1",
            intercept: "0",
            rescale_type: Some("HU"),
        },
        vec![0, 0, 0, 0, 0, 0],
    );
    let scanned = scan_dicom_path(directory.path()).expect("scanner identifies the image series");
    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::UnsupportedSamples { samples: 3 })
    ));
}

#[test]
fn stored_reader_rejects_nonfinite_modality_calibration() {
    let directory = tempfile::tempdir().expect("temporary series directory");
    write_slice(
        &directory.path().join("invalid-calibration.dcm"),
        1,
        "0",
        0,
        1,
        "MONOCHROME2",
        CalibrationFixture::Linear {
            slope: "NaN",
            intercept: "0",
            rescale_type: Some("HU"),
        },
        vec![0, 0, 0, 0],
    );
    let scanned = scan_dicom_path(directory.path()).expect("scanner records the image instance");
    assert!(matches!(
        load_dicom_stored_series(scanned),
        Err(StoredDicomError::InvalidCalibration)
    ));
}

#[path = "stored_contract_cases.rs"]
mod contract_cases;
#[path = "stored/edge_cases.rs"]
mod edge_cases;
