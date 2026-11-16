//! Value rewrites applied through one data set and its nested items.

#[cfg(test)]
mod tests;

use super::value;
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
    value: PrimitiveValue,
    values: usize,
    count: Count,
}

/// The summary field a value representation's rewrites are counted in.
type Count = fn(&mut RepairSummary) -> &mut usize;
type Repair = fn(&str) -> Option<String>;

/// Rewrite each text value of `stored` that `repair` has a spelling for and
/// keep its multiplicity: joined text stays joined, separate values stay
/// separate.
fn rewritten(
    stored: &PrimitiveValue,
    padding: char,
    repair: Repair,
) -> Option<(PrimitiveValue, usize)> {
    let mut changed = 0;
    let mut spelled = |raw: &str| {
        repair(raw.trim_end_matches(padding)).map_or_else(
            || raw.to_owned(),
            |repaired| {
                changed += 1;
                repaired
            },
        )
    };
    let value = match stored {
        PrimitiveValue::Str(text) => {
            let values: Vec<String> = text.split('\\').map(&mut spelled).collect();
            PrimitiveValue::Str(values.join("\\"))
        }
        PrimitiveValue::Strs(values) => {
            PrimitiveValue::Strs(values.iter().map(|raw| spelled(raw)).collect())
        }
        _ => return None,
    };
    (changed > 0).then_some((value, changed))
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
            let Some((value, values)) = element
                .value()
                .primitive()
                .and_then(|stored| rewritten(stored, padding, repair))
            else {
                continue;
            };
            let steps = path
                .iter()
                .copied()
                .chain([AttributeSelectorStep::Tag(tag)]);
            let selector = AttributeSelector::new(steps)
                .expect("invariant: the steps end in a tag step and are not empty");
            rewrites.push(Rewrite {
                selector,
                value,
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
/// Only DA, TM and UI elements holding text are read. Trailing padding (space
/// for DA and TM, NUL for UI) is ignored when a value is matched and dropped
/// from a value that is rewritten. An element keeps its value representation
/// and its number of values.
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
            value,
            values,
            count,
        } = rewrite;
        object
            .update_value_at(selector, |stored| {
                *stored = Value::Primitive(value.clone());
            })
            .expect("invariant: the selector was read from this object, unchanged since");
        *count(&mut summary) += values;
    }
    Ok(summary)
}
