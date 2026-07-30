//! Shared utility helpers for DICOM module parsing.

use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::object::InMemDicomObject;

/// Navigate `item → seq_tag → items[0] → inner_tag` and parse the first
/// item's value as a single scalar `T` via DS string representation.
///
/// Returns `None` if the sequence is absent, has no items, or the value
/// fails to parse.
pub(in crate::format::dicom) fn read_nested_scalar<T: std::str::FromStr>(
    item: &InMemDicomObject,
    seq_tag: Tag,
    inner_tag: Tag,
) -> Option<T> {
    let elem = item.element(seq_tag).ok()?;
    if let Value::Sequence(seq) = elem.value() {
        let inner = seq.items().first()?;
        inner
            .element(inner_tag)
            .ok()?
            .to_str()
            .ok()
            .and_then(|s| s.trim().parse::<T>().ok())
    } else {
        None
    }
}
