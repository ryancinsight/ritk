//! Admission of classic image stacks to a shared patient coordinate system.
use super::types::{DicomReadMetadata, DicomSliceMetadata};
use crate::format::dicom::identity::uid_is_valid;
use ritk_dicom::DicomTag;

fn uid(slice: &DicomSliceMetadata, element: u16) -> Option<&str> {
    let node = slice
        .preservation
        .object
        .get(DicomTag::new(0x0020, element))?;
    let value = node.value.as_text()?.trim_end_matches('\0');
    (node.vr.as_deref() == Some("UI") && uid_is_valid(value)).then_some(value)
}

impl DicomReadMetadata {
    /// Return (Study Instance UID, Frame of Reference UID) when every instance
    /// explicitly supplies the same valid pair. First-seen catalog labels
    /// alone cannot establish a common patient coordinate system.
    #[must_use]
    pub fn common_spatial_reference(&self) -> Option<(&str, &str)> {
        let (first, rest) = self.slices.split_first()?;
        let reference = (uid(first, 0x000d)?, uid(first, 0x0052)?);
        rest.iter()
            .all(|slice| {
                uid(slice, 0x000d) == Some(reference.0) && uid(slice, 0x0052) == Some(reference.1)
            })
            .then_some(reference)
    }
}
