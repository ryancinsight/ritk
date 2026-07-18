//! Typed DICOM attribute access owned by RITK.
//!
//! This module keeps backend-specific object access behind the `ritk-dicom`
//! boundary so consumers do not depend on dicom-rs tags or object APIs.

use anyhow::{Context, Result};
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;

/// DICOM tag in `(group, element)` form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DicomTag {
    /// DICOM group number.
    pub group: u16,
    /// DICOM element number.
    pub element: u16 }

impl DicomTag {
    /// Create a DICOM tag.
    #[inline]
    #[must_use]
    pub const fn new(group: u16, element: u16) -> Self {
        Self { group, element }
    }
}

impl From<DicomTag> for Tag {
    #[inline]
    fn from(value: DicomTag) -> Self {
        Self(value.group, value.element)
    }
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

    /// Return the transfer syntax UID recorded in the file meta table.
    fn transfer_syntax_uid(&self) -> &str;
}

impl DicomAttributeRead for DefaultDicomObject {
    fn required_unsigned(&self, tag: DicomTag, name: &'static str) -> Result<u16> {
        self.element(Tag::from(tag))
            .with_context(|| format!("missing {name}"))?
            .value()
            .to_int::<u16>()
            .with_context(|| format!("decode {name} as unsigned scalar"))
    }

    fn optional_unsigned(&self, tag: DicomTag, name: &'static str) -> Result<Option<u16>> {
        let Ok(element) = self.element(Tag::from(tag)) else {
            return Ok(None);
        };
        element
            .value()
            .to_int::<u16>()
            .map(Some)
            .with_context(|| format!("decode {name} as unsigned scalar"))
    }

    fn optional_decimal(&self, tag: DicomTag, name: &'static str) -> Result<Option<f64>> {
        let Ok(element) = self.element(Tag::from(tag)) else {
            return Ok(None);
        };
        element
            .value()
            .to_float64()
            .map(Some)
            .with_context(|| format!("decode {name} as decimal scalar"))
    }

    fn optional_decimal_vec(&self, tag: DicomTag, name: &'static str) -> Result<Option<Vec<f64>>> {
        let Ok(element) = self.element(Tag::from(tag)) else {
            return Ok(None);
        };
        element
            .value()
            .to_multi_float64()
            .map(Some)
            .with_context(|| format!("decode {name} as decimal vector"))
    }

    #[inline]
    fn transfer_syntax_uid(&self) -> &str {
        self.meta().transfer_syntax.trim_end_matches('\0')
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom::core::smallvec::SmallVec;
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::object::InMemDicomObject;

    fn object_with(
        elements: impl IntoIterator<Item = DataElement<InMemDicomObject>>,
    ) -> DefaultDicomObject {
        let mut obj = InMemDicomObject::new_empty();
        for element in elements {
            obj.put(element);
        }
        obj.with_meta(
            dicom::object::meta::FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2")
                .media_storage_sop_instance_uid("2.25.1")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("invariant: synthetic object metadata is valid")
    }

    #[test]
    fn reads_unsigned_decimal_and_transfer_syntax_values() {
        let obj = object_with([
            DataElement::new(Tag::from(tags::ROWS), VR::US, PrimitiveValue::from(7_u16)),
            DataElement::new(
                Tag::from(tags::PIXEL_SPACING),
                VR::DS,
                PrimitiveValue::Strs(SmallVec::from_vec(vec![
                    "0.8".to_owned(),
                    "1.25".to_owned(),
                ])),
            ),
        ]);

        assert_eq!(obj.required_unsigned(tags::ROWS, "Rows").unwrap(), 7);
        assert_eq!(
            obj.optional_decimal_vec(tags::PIXEL_SPACING, "PixelSpacing")
                .unwrap(),
            Some(vec![0.8, 1.25])
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
            Tag::from(tags::SAMPLES_PER_PIXEL),
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
}
