use anyhow::{Context, Result};
use dicom::core::Tag;

pub(super) const RGB_CHANNELS: usize = 3;

pub(super) fn read_required<T>(
    obj: &dicom::object::DefaultDicomObject,
    tag: Tag,
    name: &str,
) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    obj.element(tag)
        .with_context(|| format!("{name} absent"))?
        .to_str()
        .with_context(|| format!("{name} unreadable"))?
        .trim()
        .parse::<T>()
        .with_context(|| format!("{name} invalid"))
}

pub(super) fn read_optional<T: std::str::FromStr>(
    obj: &dicom::object::DefaultDicomObject,
    tag: Tag,
) -> Option<T> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

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
