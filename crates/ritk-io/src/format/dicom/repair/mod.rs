//! Repair of encoding defects that stop conformant readers from using a
//! data set.
//!
//! [`repair_values`] rewrites date, time and UID values that are spelled
//! outside their value representation's grammar but still denote one value:
//! the ACR-NEMA date `YYYY.MM.DD`, the ACR-NEMA time `HH:MM:SS.frac`, and UIDs
//! with surrounding spaces or leading zeros in a component (PS3.5 Table 6.2-1
//! and section 9.1). A value with no known rewrite is left as it is; nothing
//! is guessed, removed or invented.

mod value;

#[cfg(test)]
mod tests;

use dicom::core::ops::{AttributeSelector, AttributeSelectorStep};
use dicom::core::value::Value;
use dicom::core::{PrimitiveValue, VR};
use dicom::object::InMemDicomObject;

/// Steps a selector may hold: `AttributeSelector::new` in dicom-core 0.10
/// asserts fewer than 256.
const MAX_SELECTOR_STEPS: usize = 255;

/// Values rewritten in one data set, nested items included.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RepairSummary {
    /// DA values rewritten as `YYYYMMDD`.
    pub dates: usize,
    /// TM values rewritten without colons.
    pub times: usize,
    /// UI values rewritten without spaces or leading zeros.
    pub uids: usize,
}

/// Why a data set was left unrepaired.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RepairError {
    /// A sequence nests deeper than an element address can express.
    #[error("sequences nest deeper than {limit} levels")]
    NestedTooDeep {
        /// Deepest nesting an address reaches.
        limit: usize,
    },
}

/// The rewrite of one element's value and how many of its values changed.
struct Rewrite {
    selector: AttributeSelector,
    text: String,
    values: usize,
    count: Count,
}

/// The summary field a value representation's rewrites are counted in.
type Count = fn(&mut RepairSummary) -> &mut usize;
type Repair = fn(&str) -> Option<String>;

/// Rewrite each backslash-separated value `repair` has a spelling for.
fn rewritten(text: &str, padding: char, repair: Repair) -> Option<(String, usize)> {
    let mut changed = 0;
    let values: Vec<String> = text
        .split('\\')
        .map(|raw| {
            let bare = raw.trim_end_matches(padding);
            repair(bare).map_or_else(
                || bare.to_owned(),
                |repaired| {
                    changed += 1;
                    repaired
                },
            )
        })
        .collect();
    (changed > 0).then(|| (values.join("\\"), changed))
}

/// Walk `object` and every nested item and list the rewrites its DA, TM and
/// UI elements need.
fn planned(object: &InMemDicomObject) -> Result<Vec<Rewrite>, RepairError> {
    let mut rewrites = Vec::new();
    let mut pending: Vec<(Vec<AttributeSelectorStep>, &InMemDicomObject)> =
        vec![(Vec::new(), object)];
    while let Some((path, data_set)) = pending.pop() {
        if path.len() >= MAX_SELECTOR_STEPS {
            return Err(RepairError::NestedTooDeep {
                limit: MAX_SELECTOR_STEPS - 1,
            });
        }
        for element in data_set {
            let tag = element.header().tag;
            if let Value::Sequence(sequence) = element.value() {
                pending.extend((0_u32..).zip(sequence.items()).map(|(item, nested)| {
                    let mut below = path.clone();
                    below.push(AttributeSelectorStep::Nested { tag, item });
                    (below, nested)
                }));
                continue;
            }
            let (padding, repair, count): (char, Repair, Count) = match element.vr() {
                VR::DA => (' ', value::date, |summary| &mut summary.dates),
                VR::TM => (' ', value::time, |summary| &mut summary.times),
                VR::UI => ('\0', value::uid, |summary| &mut summary.uids),
                _ => continue,
            };
            let Ok(text) = element.to_str() else {
                continue;
            };
            let Some((text, values)) = rewritten(&text, padding, repair) else {
                continue;
            };
            let steps = path
                .iter()
                .cloned()
                .chain([AttributeSelectorStep::Tag(tag)]);
            let selector = AttributeSelector::new(steps)
                .expect("invariant: the steps end in a tag step and are not empty");
            rewrites.push(Rewrite {
                selector,
                text,
                values,
                count,
            });
        }
    }
    Ok(rewrites)
}

/// Rewrite the nonconformant date, time and UID values of `object` and of
/// every item nested in it.
///
/// A UID rewrite changes the bytes other objects use to refer to this one, so
/// the objects of a study are repaired together; equal inputs always get the
/// same rewrite.
///
/// # Errors
/// Returns [`RepairError::NestedTooDeep`], with `object` unchanged, when a
/// sequence nests deeper than an element address can express.
///
/// # Examples
/// ```
/// use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
/// use dicom::object::InMemDicomObject;
/// use ritk_io::format::dicom::repair::repair_values;
///
/// let study_date = Tag(0x0008, 0x0020);
/// let mut object = InMemDicomObject::new_empty();
/// object.put(DataElement::new(study_date, VR::DA, PrimitiveValue::from("1994.07.21")));
///
/// let summary = repair_values(&mut object)?;
///
/// assert_eq!(summary.dates, 1);
/// assert_eq!(object.element(study_date)?.to_str()?, "19940721");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn repair_values(object: &mut InMemDicomObject) -> Result<RepairSummary, RepairError> {
    let mut summary = RepairSummary::default();
    for rewrite in planned(object)? {
        let Rewrite {
            selector,
            text,
            values,
            count,
        } = rewrite;
        object
            .update_value_at(selector, |value| {
                *value = Value::Primitive(PrimitiveValue::from(text.as_str()));
            })
            .expect("invariant: the selector was read from this object, unchanged since");
        *count(&mut summary) += values;
    }
    Ok(summary)
}
