//! Series identity validation before metadata accumulation.

use anyhow::{bail, Context, Result};
use arrayvec::ArrayString;
use dicom::core::Tag;
use dicom::object::DefaultDicomObject;

use super::sop_class::{classify_sop_class, SopClassKind};

/// DICOM UID syntax from PS3.5 section 9.1.
pub(super) fn uid_is_valid(uid: &str) -> bool {
    !uid.is_empty()
        && uid.len() <= 64
        && uid.split('.').all(|component| {
            !component.is_empty()
                && component.bytes().all(|byte| byte.is_ascii_digit())
                && !(component.len() > 1 && component.starts_with('0'))
        })
}

/// Non-image storage objects do not participate in image-series selection.
pub(super) fn image_series_uid(obj: &DefaultDicomObject) -> Result<Option<ArrayString<64>>> {
    if let Ok(element) = obj.element(Tag(0x0008, 0x0016)) {
        let sop = element.to_str().context("invalid SOPClassUID")?;
        let kind = classify_sop_class(sop.trim_end_matches('\0').trim());
        if !kind.is_image_storage() && !matches!(kind, SopClassKind::Other(_)) {
            return Ok(None);
        }
    }
    let element = obj
        .element(Tag(0x0020, 0x000E))
        .context("image is missing SeriesInstanceUID")?;
    let value = element
        .to_str()
        .context("invalid SeriesInstanceUID value")?;
    let uid = value.trim_end_matches('\0');
    if !uid_is_valid(uid) {
        bail!("invalid SeriesInstanceUID syntax");
    }
    Ok(Some(ArrayString::from(uid).expect(
        "invariant: UID syntax validation enforces 64-byte capacity",
    )))
}
