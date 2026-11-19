//! Typed DICOM attribute access owned by RITK.
//!
//! This module keeps dataset access behind the `ritk-dicom`
//! boundary so consumers use one tag and value representation.

use crate::dataset::{DicomFile, Value};
use anyhow::{Context, Result};

/// DICOM tag in `(group, element)` form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DicomTag {
    /// DICOM group number.
    pub group: u16,
    /// DICOM element number.
    pub element: u16,
}

impl DicomTag {
    /// Create a DICOM tag.
    #[inline]
    #[must_use]
    pub const fn new(group: u16, element: u16) -> Self {
        Self { group, element }
    }

    /// The hexadecimal group and element without display punctuation.
    pub fn canonical(self) -> String {
        format!("{:04X},{:04X}", self.group, self.element)
    }
}

impl std::fmt::Display for DicomTag {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "({:04X},{:04X})", self.group, self.element)
    }
}

/// Private DICOM groups have odd group numbers.
#[inline]
pub const fn is_private_tag(tag: DicomTag) -> bool {
    tag.group % 2 == 1
}

/// Common DICOM image tags used by imaging consumers.
pub mod tags {
    use super::DicomTag;

    /// Rows (0028,0010).
    pub const ROWS: DicomTag = DicomTag::new(0x0028, 0x0010);
    /// Columns (0028,0011).
    pub const COLUMNS: DicomTag = DicomTag::new(0x0028, 0x0011);
    /// Samples per Pixel (0028,0002).
    pub const SAMPLES_PER_PIXEL: DicomTag = DicomTag::new(0x0028, 0x0002);
    /// Bits Allocated (0028,0100).
    pub const BITS_ALLOCATED: DicomTag = DicomTag::new(0x0028, 0x0100);
    /// Bits Stored (0028,0101).
    pub const BITS_STORED: DicomTag = DicomTag::new(0x0028, 0x0101);
    /// Pixel Representation (0028,0103).
    pub const PIXEL_REPRESENTATION: DicomTag = DicomTag::new(0x0028, 0x0103);
    /// Rescale Intercept (0028,1052).
    pub const RESCALE_INTERCEPT: DicomTag = DicomTag::new(0x0028, 0x1052);
    /// Rescale Slope (0028,1053).
    pub const RESCALE_SLOPE: DicomTag = DicomTag::new(0x0028, 0x1053);
    /// Pixel Spacing (0028,0030).
    pub const PIXEL_SPACING: DicomTag = DicomTag::new(0x0028, 0x0030);
    /// Slice Thickness (0018,0050).
    pub const SLICE_THICKNESS: DicomTag = DicomTag::new(0x0018, 0x0050);
    /// Image Position Patient (0020,0032).
    pub const IMAGE_POSITION_PATIENT: DicomTag = DicomTag::new(0x0020, 0x0032);
    /// Image Orientation Patient (0020,0037).
    ///
    /// The first three values are the direction cosine of the image rows and
    /// the last three values are the direction cosine of the image columns.
    pub const IMAGE_ORIENTATION_PATIENT: DicomTag = DicomTag::new(0x0020, 0x0037);

    // ── Diffusion MRI ─────────────────────────────────────────────────

    /// Diffusion b-value (0018,9087).
    ///
    /// The diffusion weighting factor in s/mm² for this frame/slice.
    pub const DIFFUSION_B_VALUE: DicomTag = DicomTag::new(0x0018, 0x9087);
    /// Diffusion Gradient Direction (0018,9089).
    ///
    /// The unit gradient direction vector `[x, y, z]` in the patient
    /// coordinate system for this frame/slice.
    pub const DIFFUSION_GRADIENT_DIRECTION: DicomTag = DicomTag::new(0x0018, 0x9089);

    // ── Identification ────────────────────────────────────────────────

    /// Specific Character Set (0008,0005).
    pub const SPECIFIC_CHARACTER_SET: DicomTag = DicomTag::new(0x0008, 0x0005);
    /// SOP Class UID (0008,0016).
    pub const SOP_CLASS_UID: DicomTag = DicomTag::new(0x0008, 0x0016);
    /// Modality (0008,0060).
    pub const MODALITY: DicomTag = DicomTag::new(0x0008, 0x0060);
    /// Series Description (0008,103E).
    pub const SERIES_DESCRIPTION: DicomTag = DicomTag::new(0x0008, 0x103E);
    /// Patient ID (0010,0020).
    pub const PATIENT_ID: DicomTag = DicomTag::new(0x0010, 0x0020);
    /// Series Instance UID (0020,000E).
    pub const SERIES_INSTANCE_UID: DicomTag = DicomTag::new(0x0020, 0x000E);

    /// Instance Number (0020,0013).
    pub const INSTANCE_NUMBER: DicomTag = DicomTag::new(0x0020, 0x0013);
    /// SOP Instance UID (0008,0018).
    pub const SOP_INSTANCE_UID: DicomTag = DicomTag::new(0x0008, 0x0018);
    /// Study Instance UID (0020,000D).
    pub const STUDY_INSTANCE_UID: DicomTag = DicomTag::new(0x0020, 0x000D);
    /// Slice Location (0020,1041).
    pub const SLICE_LOCATION: DicomTag = DicomTag::new(0x0020, 0x1041);
    /// Photometric Interpretation (0028,0004).
    pub const PHOTOMETRIC_INTERPRETATION: DicomTag = DicomTag::new(0x0028, 0x0004);
}

/// Typed attribute reads over a parsed DICOM object.
pub trait DicomAttributeRead {
    /// Read a required unsigned scalar attribute.
    ///
    /// # Errors
    /// Returns an error when the element is missing or cannot be decoded as an
    /// unsigned scalar.
    fn required_unsigned(&self, tag: DicomTag, name: &'static str) -> Result<u16>;

    /// Read an optional unsigned scalar attribute.
    ///
    /// # Errors
    /// Returns an error when the element is present but cannot be decoded as an
    /// unsigned scalar.
    fn optional_unsigned(&self, tag: DicomTag, name: &'static str) -> Result<Option<u16>>;

    /// Read an optional decimal scalar attribute.
    ///
    /// # Errors
    /// Returns an error when the element is present but cannot be decoded as a
    /// DICOM decimal string value.
    fn optional_decimal(&self, tag: DicomTag, name: &'static str) -> Result<Option<f64>>;

    /// Read an optional multi-valued decimal attribute.
    ///
    /// # Errors
    /// Returns an error when the element is present but cannot be decoded as a
    /// DICOM decimal string vector.
    fn optional_decimal_vec(&self, tag: DicomTag, name: &'static str) -> Result<Option<Vec<f64>>>;

    /// Read an optional element as raw bytes.
    ///
    /// Returns `None` when the element is absent.
    ///
    /// # Errors
    ///
    /// Returns an error when the element is present but its value cannot be
    /// decoded as a byte sequence.
    fn optional_bytes(&self, tag: DicomTag, name: &'static str) -> Result<Option<Vec<u8>>>;

    /// Return the transfer syntax UID recorded in the file meta table.
    fn transfer_syntax_uid(&self) -> &str;
}

fn require_scalar(value: &Value, tag: DicomTag, name: &str) -> Result<()> {
    let cardinality = value.cardinality();
    if cardinality != 1 {
        anyhow::bail!(
            "{name} ({:04X},{:04X}) has cardinality={cardinality}; expected exactly 1",
            tag.group,
            tag.element
        );
    }
    Ok(())
}

impl DicomAttributeRead for DicomFile {
    fn required_unsigned(&self, tag: DicomTag, name: &'static str) -> Result<u16> {
        let element = self
            .element(tag)
            .with_context(|| format!("missing {name}"))?;
        require_scalar(element.value(), tag, name)?;
        element
            .value()
            .to_num::<u16>()
            .with_context(|| format!("decode {name} as unsigned scalar"))
    }

    fn optional_unsigned(&self, tag: DicomTag, name: &'static str) -> Result<Option<u16>> {
        let Ok(element) = self.element(tag) else {
            return Ok(None);
        };
        require_scalar(element.value(), tag, name)?;
        element
            .value()
            .to_num::<u16>()
            .map(Some)
            .with_context(|| format!("decode {name} as unsigned scalar"))
    }

    fn optional_decimal(&self, tag: DicomTag, name: &'static str) -> Result<Option<f64>> {
        let Ok(element) = self.element(tag) else {
            return Ok(None);
        };
        element
            .value()
            .to_num::<f64>()
            .map(Some)
            .with_context(|| format!("decode {name} as decimal scalar"))
    }

    fn optional_decimal_vec(&self, tag: DicomTag, name: &'static str) -> Result<Option<Vec<f64>>> {
        let Ok(element) = self.element(tag) else {
            return Ok(None);
        };
        element
            .value()
            .to_numbers::<f64>()
            .map(Some)
            .with_context(|| format!("decode {name} as decimal vector"))
    }

    fn optional_bytes(&self, tag: DicomTag, _name: &'static str) -> Result<Option<Vec<u8>>> {
        let Ok(element) = self.element(tag) else {
            return Ok(None);
        };
        let bytes = element
            .value()
            .to_bytes()
            .with_context(|| "decode raw bytes".to_string())?
            .to_vec();
        Ok(Some(bytes))
    }

    #[inline]
    fn transfer_syntax_uid(&self) -> &str {
        self.meta().transfer_syntax.trim_end_matches('\0')
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;
    use crate::dataset::Dataset;
    use crate::dataset::SmallVec;
    use crate::dataset::{DataElement, PrimitiveValue, VR};

    fn object_with(elements: impl IntoIterator<Item = DataElement>) -> DicomFile {
        let mut obj = Dataset::new();
        for element in elements {
            obj.put(element);
        }
        obj.with_meta(
            crate::dataset::FileMetaBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                .media_storage_sop_instance_uid("2.25.1")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("invariant: synthetic object metadata is valid")
    }

    #[test]
    fn reads_unsigned_decimal_and_transfer_syntax_values() {
        let obj = object_with([
            DataElement::new(tags::ROWS, VR::US, PrimitiveValue::from(7_u16)),
            DataElement::new(
                tags::PIXEL_SPACING,
                VR::DS,
                PrimitiveValue::Strs(SmallVec::from_vec(vec![
                    "0.8".to_owned(),
                    "1.25".to_owned(),
                ])),
            ),
            DataElement::new(
                tags::IMAGE_ORIENTATION_PATIENT,
                VR::DS,
                PrimitiveValue::Strs(SmallVec::from_vec(vec![
                    "1".to_owned(),
                    "0".to_owned(),
                    "0".to_owned(),
                    "0".to_owned(),
                    "1".to_owned(),
                    "0".to_owned(),
                ])),
            ),
        ]);

        assert_eq!(obj.required_unsigned(tags::ROWS, "Rows").unwrap(), 7);
        assert_eq!(
            obj.optional_decimal_vec(tags::PIXEL_SPACING, "PixelSpacing")
                .unwrap(),
            Some(vec![0.8, 1.25])
        );
        assert_eq!(
            obj.optional_decimal_vec(tags::IMAGE_ORIENTATION_PATIENT, "ImageOrientationPatient")
                .unwrap(),
            Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        );
        assert_eq!(obj.transfer_syntax_uid(), "1.2.840.10008.1.2.1");
    }

    #[test]
    fn optional_attribute_distinguishes_absent_from_malformed() {
        let absent = object_with([]);
        assert_eq!(
            absent
                .optional_unsigned(tags::SAMPLES_PER_PIXEL, "SamplesPerPixel")
                .unwrap(),
            None
        );

        let malformed = object_with([DataElement::new(
            tags::SAMPLES_PER_PIXEL,
            VR::US,
            PrimitiveValue::from("not-an-integer"),
        )]);
        let err = malformed
            .optional_unsigned(tags::SAMPLES_PER_PIXEL, "SamplesPerPixel")
            .unwrap_err();
        assert!(
            err.to_string().contains("SamplesPerPixel"),
            "error must name malformed attribute, got {err:#}"
        );
    }

    #[test]
    fn required_unsigned_rejects_multiple_values() {
        let object = object_with([DataElement::new(
            tags::BITS_STORED,
            VR::US,
            PrimitiveValue::U16([12_u16, 16][..].into()),
        )]);

        let error = object
            .required_unsigned(tags::BITS_STORED, "BitsStored")
            .expect_err("multi-valued BitsStored must be rejected");

        assert_eq!(
            error.to_string(),
            "BitsStored (0028,0101) has cardinality=2; expected exactly 1"
        );
    }
}
