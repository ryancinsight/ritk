//! Series identity validation before metadata accumulation.

use std::borrow::Cow;

use anyhow::{bail, Context, Result};
use arrayvec::ArrayString;
use ritk_dicom::dataset::DicomFile;
use ritk_dicom::{tags, DicomTag};

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
pub(super) fn image_series_uid(obj: &DicomFile) -> Result<Option<ArrayString<64>>> {
    image_series_uid_from(|tag| {
        obj.element(tag)
            .ok()
            .map(|element| element.to_str().context("invalid DICOM UID value"))
            .transpose()
    })
}

pub(super) fn image_series_uid_from<'a>(
    mut value: impl FnMut(DicomTag) -> Result<Option<Cow<'a, str>>>,
) -> Result<Option<ArrayString<64>>> {
    if let Some(sop) = value(tags::SOP_CLASS_UID).context("invalid SOPClassUID")? {
        let kind = classify_sop_class(sop.trim_end_matches('\0').trim());
        if !kind.is_image_storage() && !matches!(kind, SopClassKind::Other(_)) {
            return Ok(None);
        }
    }
    let series = value(tags::SERIES_INSTANCE_UID)
        .context("invalid SeriesInstanceUID value")?
        .context("image is missing SeriesInstanceUID")?;
    let uid = series.trim_end_matches('\0');
    if !uid_is_valid(uid) {
        bail!("invalid SeriesInstanceUID syntax");
    }
    Ok(Some(ArrayString::from(uid).expect(
        "invariant: UID syntax validation enforces 64-byte capacity",
    )))
}
