//! Per-frame functional group extraction for Enhanced (multi-frame) DICOM objects.
//!
//! Parses Shared Functional Groups (5200,9229) and Per-Frame Functional Groups
//! (5200,9230) per DICOM PS3.3 C.7.6.16 into a `Vec<PerFrameInfo>`.

use dicom::core::value::Value;
use dicom::core::Tag;
use dicom::object::InMemDicomObject;

use super::reader::parse_ds_backslash;
use super::types::PerFrameInfo;
use crate::format::dicom::helpers::read_nested_f64;

/// Navigate item → seq_tag → items()\[0\] → inner_tag and parse as `\[f64; N\]` DS array.
///
/// Returns None if either sequence is absent, has no items, or the DS value fails to
/// parse with at least N components.
fn read_nested_ds<const N: usize>(
    item: &InMemDicomObject,
    seq_tag: Tag,
    inner_tag: Tag,
) -> Option<[f64; N]> {
    let elem = item.element(seq_tag).ok()?;
    if let Value::Sequence(seq) = elem.value() {
        let inner_item = seq.items().first()?;
        let inner_elem = inner_item.element(inner_tag).ok()?;
        inner_elem
            .to_str()
            .ok()
            .and_then(|s| parse_ds_backslash::<N>(&s))
    } else {
        None
    }
}

/// Extract per-frame metadata from functional group sequences per DICOM PS3.3 C.7.6.16.
///
/// Reads Shared Functional Groups (5200,9229) as fallback defaults, then per-frame
/// overrides from Per-Frame Functional Groups (5200,9230).
///
/// # Algorithm
///
/// 1. If neither (5200,9229) nor (5200,9230) is present, return `Vec::new()`.
/// 2. Parse shared groups (5200,9229)\[0\] into a `PerFrameInfo` template.
/// 3. For each frame index k in \[0, n_frames\):
///    a. Start from the shared template.
///    b. If (5200,9230)\[k\] exists, override non-None fields.
///    c. Push merged result.
///
/// # Fallback chain per attribute A
/// per_frame\[k\].A = per_frame_groups\[k\].A // if Some(...)
///               ?? shared_groups.A          // if Some(...)
///               ?? None
pub(crate) fn extract_functional_groups(
    obj: &InMemDicomObject,
    n_frames: usize,
) -> Vec<PerFrameInfo> {
    let has_shared = obj.element(Tag(0x5200, 0x9229)).is_ok();
    let has_per_frame = obj.element(Tag(0x5200, 0x9230)).is_ok();

    if !has_shared && !has_per_frame {
        tracing::debug!(
            "extract_functional_groups: (5200,9229) and (5200,9230) absent; per_frame=[]"
        );
        return Vec::new();
    }

    // Parse shared functional groups template.
    let shared_template = if let Ok(sg_elem) = obj.element(Tag(0x5200, 0x9229)) {
        if let Value::Sequence(seq) = sg_elem.value() {
            if let Some(item) = seq.items().first() {
                let tpl = PerFrameInfo {
                    image_position: read_nested_ds::<3>(
                        item,
                        Tag(0x0020, 0x9113),
                        Tag(0x0020, 0x0032),
                    ),
                    image_orientation: read_nested_ds::<6>(
                        item,
                        Tag(0x0020, 0x9116),
                        Tag(0x0020, 0x0037),
                    ),
                    pixel_spacing: read_nested_ds::<2>(
                        item,
                        Tag(0x0028, 0x9110),
                        Tag(0x0028, 0x0030),
                    ),
                    slice_thickness: read_nested_f64(
                        item,
                        Tag(0x0028, 0x9110),
                        Tag(0x0018, 0x0050),
                    ),
                    rescale_slope: read_nested_f64(item, Tag(0x0028, 0x9145), Tag(0x0028, 0x1053)),
                    rescale_intercept: read_nested_f64(
                        item,
                        Tag(0x0028, 0x9145),
                        Tag(0x0028, 0x1052),
                    ),
                };
                tracing::debug!(
                    has_image_position = tpl.image_position.is_some(),
                    has_pixel_spacing = tpl.pixel_spacing.is_some(),
                    has_rescale = tpl.rescale_slope.is_some(),
                    "extract_functional_groups: shared template parsed"
                );
                tpl
            } else {
                PerFrameInfo::default()
            }
        } else {
            PerFrameInfo::default()
        }
    } else {
        PerFrameInfo::default()
    };

    // Collect per-frame items.
    let pf_items: Vec<InMemDicomObject> = if let Ok(pf_elem) = obj.element(Tag(0x5200, 0x9230)) {
        if let Value::Sequence(seq) = pf_elem.value() {
            seq.items().to_vec()
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    tracing::debug!(
        n_frames,
        pf_items = pf_items.len(),
        "extract_functional_groups: merging per-frame metadata"
    );

    (0..n_frames)
        .map(|k| {
            let mut info = shared_template.clone();
            if let Some(item) = pf_items.get(k) {
                macro_rules! override_if_some {
                    ($field:ident, $expr:expr) => {
                        if let Some(v) = $expr {
                            info.$field = Some(v);
                        }
                    };
                }
                override_if_some!(
                    image_position,
                    read_nested_ds::<3>(item, Tag(0x0020, 0x9113), Tag(0x0020, 0x0032))
                );
                override_if_some!(
                    image_orientation,
                    read_nested_ds::<6>(item, Tag(0x0020, 0x9116), Tag(0x0020, 0x0037))
                );
                override_if_some!(
                    pixel_spacing,
                    read_nested_ds::<2>(item, Tag(0x0028, 0x9110), Tag(0x0028, 0x0030))
                );
                override_if_some!(
                    slice_thickness,
                    read_nested_f64(item, Tag(0x0028, 0x9110), Tag(0x0018, 0x0050))
                );
                override_if_some!(
                    rescale_slope,
                    read_nested_f64(item, Tag(0x0028, 0x9145), Tag(0x0028, 0x1053))
                );
                override_if_some!(
                    rescale_intercept,
                    read_nested_f64(item, Tag(0x0028, 0x9145), Tag(0x0028, 0x1052))
                );
            }
            info
        })
        .collect()
}
