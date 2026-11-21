//! Per-frame functional group extraction for Enhanced (multi-frame) DICOM objects.
//!
//! Parses Shared Functional Groups (5200,9229) and Per-Frame Functional Groups
//! (5200,9230) per DICOM PS3.3 C.7.6.16 into a `Vec<PerFrameInfo>`.

use ritk_dicom::dataset::Dataset;
use ritk_dicom::dataset::DicomTag;
use ritk_dicom::dataset::Value;

use super::super::value_units::{declared_rescale_unit, DicomValueUnit};
use super::geometry::{length_from_mm, parse_lengths_mm};
use super::reader::parse_ds_backslash;
use super::types::PerFrameInfo;
use crate::format::dicom::tag_parse::read_nested_scalar;

/// Navigate item → seq_tag → items()\[0\] → inner_tag and parse a DS tuple.
///
/// Returns None if either sequence is absent, has no items, or the DS value fails to
/// pass the supplied parser.
fn read_nested_ds<Parsed>(
    item: &Dataset,
    seq_tag: DicomTag,
    inner_tag: DicomTag,
    parse: impl FnOnce(&str) -> Option<Parsed>,
) -> Option<Parsed> {
    let elem = item.element(seq_tag).ok()?;
    if let Value::Sequence(seq) = elem.value() {
        let inner_item = seq.items().first()?;
        let inner_elem = inner_item.element(inner_tag).ok()?;
        inner_elem.to_str().ok().and_then(|s| parse(&s))
    } else {
        None
    }
}

fn value_unit(item: &Dataset) -> Option<DicomValueUnit> {
    let transform = item.element(DicomTag::new(0x0028, 0x9145)).ok()?;
    declared_rescale_unit(transform.value().items()?.first()?)
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
pub(crate) fn extract_functional_groups(obj: &Dataset, n_frames: usize) -> Vec<PerFrameInfo> {
    let has_shared = obj.element(DicomTag::new(0x5200, 0x9229)).is_ok();
    let has_per_frame = obj.element(DicomTag::new(0x5200, 0x9230)).is_ok();

    if !has_shared && !has_per_frame {
        tracing::debug!(
            "extract_functional_groups: (5200,9229) and (5200,9230) absent; per_frame=[]"
        );
        return Vec::new();
    }

    // Parse shared functional groups template.
    let shared_template = match obj.element(DicomTag::new(0x5200, 0x9229)) {
        Ok(sg_elem) => {
            if let Value::Sequence(seq) = sg_elem.value() {
                if let Some(item) = seq.items().first() {
                    let tpl = PerFrameInfo {
                        image_position: read_nested_ds(
                            item,
                            DicomTag::new(0x0020, 0x9113),
                            DicomTag::new(0x0020, 0x0032),
                            parse_lengths_mm::<3>,
                        ),
                        image_orientation: read_nested_ds(
                            item,
                            DicomTag::new(0x0020, 0x9116),
                            DicomTag::new(0x0020, 0x0037),
                            parse_ds_backslash::<6>,
                        ),
                        pixel_spacing: read_nested_ds(
                            item,
                            DicomTag::new(0x0028, 0x9110),
                            DicomTag::new(0x0028, 0x0030),
                            parse_lengths_mm::<2>,
                        ),
                        slice_thickness: read_nested_scalar::<f64>(
                            item,
                            DicomTag::new(0x0028, 0x9110),
                            DicomTag::new(0x0018, 0x0050),
                        )
                        .and_then(length_from_mm),
                        value_unit: value_unit(item),
                        rescale_slope: read_nested_scalar::<f64>(
                            item,
                            DicomTag::new(0x0028, 0x9145),
                            DicomTag::new(0x0028, 0x1053),
                        ),
                        rescale_intercept: read_nested_scalar::<f64>(
                            item,
                            DicomTag::new(0x0028, 0x9145),
                            DicomTag::new(0x0028, 0x1052),
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
        }
        _ => PerFrameInfo::default(),
    };

    // Collect per-frame items.
    let pf_items: Vec<Dataset> = match obj.element(DicomTag::new(0x5200, 0x9230)) {
        Ok(pf_elem) => {
            if let Value::Sequence(seq) = pf_elem.value() {
                seq.items().to_vec()
            } else {
                vec![]
            }
        }
        _ => {
            vec![]
        }
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
                    ($field:ident, $expr:expr_2021) => {
                        if let Some(v) = $expr {
                            info.$field = Some(v);
                        }
                    };
                }
                override_if_some!(
                    image_position,
                    read_nested_ds(
                        item,
                        DicomTag::new(0x0020, 0x9113),
                        DicomTag::new(0x0020, 0x0032),
                        parse_lengths_mm::<3>,
                    )
                );
                override_if_some!(
                    image_orientation,
                    read_nested_ds(
                        item,
                        DicomTag::new(0x0020, 0x9116),
                        DicomTag::new(0x0020, 0x0037),
                        parse_ds_backslash::<6>,
                    )
                );
                override_if_some!(
                    pixel_spacing,
                    read_nested_ds(
                        item,
                        DicomTag::new(0x0028, 0x9110),
                        DicomTag::new(0x0028, 0x0030),
                        parse_lengths_mm::<2>,
                    )
                );
                override_if_some!(
                    slice_thickness,
                    read_nested_scalar::<f64>(
                        item,
                        DicomTag::new(0x0028, 0x9110),
                        DicomTag::new(0x0018, 0x0050)
                    )
                    .and_then(length_from_mm)
                );
                override_if_some!(value_unit, value_unit(item));
                override_if_some!(
                    rescale_slope,
                    read_nested_scalar::<f64>(
                        item,
                        DicomTag::new(0x0028, 0x9145),
                        DicomTag::new(0x0028, 0x1053)
                    )
                );
                override_if_some!(
                    rescale_intercept,
                    read_nested_scalar::<f64>(
                        item,
                        DicomTag::new(0x0028, 0x9145),
                        DicomTag::new(0x0028, 0x1052)
                    )
                );
            }
            info
        })
        .collect()
}
