use anyhow::{Context, Result};
use dicom::core::Tag;

#[allow(dead_code)]
pub(super) const RGB_CHANNELS: usize = 3;

#[allow(dead_code)]
pub(super) fn required_usize(
    obj: &dicom::object::DefaultDicomObject,
    tag: Tag,
    name: &str,
) -> Result<usize> {
    obj.element(tag)
        .with_context(|| format!("{name} absent"))?
        .to_str()
        .with_context(|| format!("{name} unreadable"))?
        .trim()
        .parse::<usize>()
        .with_context(|| format!("{name} invalid"))
}

#[allow(dead_code)]
pub(super) fn optional_usize(obj: &dicom::object::DefaultDicomObject, tag: Tag) -> Option<usize> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

#[allow(dead_code)]
pub(super) fn optional_u16(obj: &dicom::object::DefaultDicomObject, tag: Tag) -> Option<u16> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

#[allow(dead_code)]
pub(super) fn required_string(
    obj: &dicom::object::DefaultDicomObject,
    tag: Tag,
    name: &str,
) -> Result<String> {
    Ok(obj
        .element(tag)
        .with_context(|| format!("{name} absent"))?
        .to_str()
        .with_context(|| format!("{name} unreadable"))?
        .to_string())
}
