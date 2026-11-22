//! Classic PET Part 10 fixtures for viewer integration tests.

use std::path::Path;

use ritk_dicom::dataset::{
    DataElement, Dataset, DicomTag as Tag, FileMetaBuilder, PrimitiveValue, Sequence, Value, VR,
};

const CLASSIC_PET_STORAGE: &str = "1.2.840.10008.5.1.4.1.1.128";

/// Timing and unit attributes for one image in a synthetic classic PET series.
#[derive(Debug, Clone, Copy)]
pub struct ClassicPetImage {
    pub decay_correction: Option<&'static str>,
    pub series_date: &'static str,
    pub series_time: Option<&'static str>,
    pub acquisition_time: Option<&'static str>,
    pub administration_time: Option<&'static str>,
    pub administration_datetime: Option<&'static str>,
    pub utc_offset: Option<&'static str>,
    pub frame_reference_ms: &'static str,
    pub frame_duration_ms: &'static str,
    pub start_decay_factor: &'static str,
    pub units: &'static str,
}

impl ClassicPetImage {
    /// One hour between administration and the reference time on the same day.
    pub const ONE_HOUR: Self = Self {
        decay_correction: Some("NONE"),
        series_date: "20261121",
        series_time: Some("110000"),
        acquisition_time: Some("110000"),
        administration_time: Some("100000"),
        administration_datetime: Some("20261121100000"),
        utc_offset: None,
        frame_reference_ms: "0",
        frame_duration_ms: "120000",
        start_decay_factor: "1",
        units: "BQML",
    };

    /// Write one spatial slice with the supplied instance number and z position.
    ///
    /// # Panics
    ///
    /// Panics if the fixture cannot be written as a Part 10 file.
    pub fn write(self, path: &Path, instance: u16, z_mm: i32) {
        let instance_uid = format!("2.25.110021.1.{instance}");
        let position = format!("0\\0\\{z_mm}");
        let mut image = Dataset::new();
        for (tag, vr, value) in [
            (Tag::new(8, 8), VR::CS, r"ORIGINAL\PRIMARY"),
            (Tag::new(8, 0x16), VR::UI, CLASSIC_PET_STORAGE),
            (Tag::new(8, 0x18), VR::UI, &instance_uid),
            (Tag::new(8, 0x20), VR::DA, "20261121"),
            (Tag::new(8, 0x21), VR::DA, self.series_date),
            (Tag::new(8, 0x22), VR::DA, self.series_date),
            (Tag::new(8, 0x60), VR::CS, "PT"),
            (Tag::new(0x10, 0x20), VR::LO, "SYNTHETIC"),
            (Tag::new(0x10, 0x1030), VR::DS, "70"),
            (Tag::new(0x18, 0x50), VR::DS, "2"),
            (Tag::new(0x18, 0x1242), VR::IS, self.frame_duration_ms),
            (Tag::new(0x20, 0x0d), VR::UI, "2.25.110021.2"),
            (Tag::new(0x20, 0x0e), VR::UI, "2.25.110021.3"),
            (Tag::new(0x20, 0x32), VR::DS, &position),
            (Tag::new(0x20, 0x37), VR::DS, r"1\0\0\0\1\0"),
            (Tag::new(0x20, 0x52), VR::UI, "2.25.110021.4"),
            (Tag::new(0x28, 4), VR::CS, "MONOCHROME2"),
            (Tag::new(0x28, 0x30), VR::DS, r"2\2"),
            (Tag::new(0x28, 0x1052), VR::DS, "0"),
            (Tag::new(0x28, 0x1053), VR::DS, "1"),
            (Tag::new(0x54, 0x1000), VR::CS, r"STATIC\IMAGE"),
            (Tag::new(0x54, 0x1001), VR::CS, self.units),
            (Tag::new(0x54, 0x1300), VR::DS, self.frame_reference_ms),
        ] {
            image.put_value(tag, vr, value);
        }
        let mode = self.decay_correction.map(|value| value.trim_matches(' '));
        let decay_factor = (mode == Some("START")).then_some(self.start_decay_factor);
        for (tag, vr, value) in [
            (Tag::new(8, 0x31), VR::TM, self.series_time),
            (Tag::new(8, 0x32), VR::TM, self.acquisition_time),
            (Tag::new(8, 0x0201), VR::SH, self.utc_offset),
            (Tag::new(0x54, 0x1102), VR::CS, self.decay_correction),
            (Tag::new(0x54, 0x1321), VR::DS, decay_factor),
        ] {
            if let Some(value) = value {
                image.put_value(tag, vr, value);
            }
        }
        image.put_value(Tag::new(0x20, 0x13), VR::IS, instance.to_string());
        let corrected_image = match mode {
            Some("START" | "ADMIN") => r"ATTN\DECY",
            _ => "ATTN",
        };
        image.put_value(Tag::new(0x28, 0x51), VR::CS, corrected_image);
        for (tag, value) in [
            (2, 1_u16),
            (0x10, 2),
            (0x11, 2),
            (0x100, 16),
            (0x101, 16),
            (0x102, 15),
            (0x103, 0),
        ] {
            image.put_value(Tag::new(0x28, tag), VR::US, value);
        }
        let mut isotope = Dataset::new();
        isotope.put_value(Tag::new(0x18, 0x1074), VR::DS, "350000000");
        isotope.put_value(Tag::new(0x18, 0x1075), VR::DS, "6584.04");
        if let Some(time) = self.administration_time {
            isotope.put_value(Tag::new(0x18, 0x1072), VR::TM, time);
        }
        if let Some(time) = self.administration_datetime {
            isotope.put_value(Tag::new(0x18, 0x1078), VR::DT, time);
        }
        image.put(DataElement::new(
            Tag::new(0x54, 0x0016),
            VR::SQ,
            Value::from(Sequence::from(vec![isotope])),
        ));
        let pixels = if self.units == "GML" {
            [2, 3, 0, 4]
        } else {
            [5_000, 7_500, 0, 4_000]
        };
        image.put(DataElement::new(
            Tag::new(0x7fe0, 0x10),
            VR::OW,
            PrimitiveValue::U16(pixels.as_slice().into()),
        ));
        image
            .with_meta(
                FileMetaBuilder::new()
                    .media_storage_sop_class_uid(CLASSIC_PET_STORAGE)
                    .media_storage_sop_instance_uid(&instance_uid)
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .expect("attach classic PET file metadata")
            .write_to_file(path)
            .expect("write classic PET Part 10 fixture");
    }
}
