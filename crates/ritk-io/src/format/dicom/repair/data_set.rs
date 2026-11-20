//! Value rewrites applied through one data set and its nested items.

#[cfg(test)]
mod tests;

use super::value;
use ritk_dicom::dataset::Dataset;
use ritk_dicom::dataset::Value;
use ritk_dicom::dataset::{PrimitiveValue, VR};

const MAX_NESTING: usize = 254;

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
    /// A sequence nests beyond the supported repair depth.
    #[error("sequences nest deeper than {limit} levels")]
    NestedTooDeep {
        /// Deepest nesting an address reaches.
        limit: usize,
    },
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

// Check the whole tree before applying any changes. The existing limit
// keeps repair's all-or-nothing behavior for an in-memory object.
fn check_depth(root: &Dataset) -> Result<(), RepairError> {
    let mut pending = vec![(0, root)];
    while let Some((depth, dataset)) = pending.pop() {
        if depth > MAX_NESTING {
            return Err(RepairError::NestedTooDeep { limit: MAX_NESTING });
        }
        for element in dataset {
            if let Value::Sequence(sequence) = element.value() {
                pending.extend(sequence.items().iter().map(|item| (depth + 1, item)));
            }
        }
    }
    Ok(())
}

fn apply(dataset: &mut Dataset, summary: &mut RepairSummary) {
    for element in dataset.iter_mut() {
        let (padding, repair, count): (char, Repair, Count) = match element.vr() {
            VR::DA => (' ', value::date, |summary| &mut summary.dates),
            VR::TM => (' ', value::time, |summary| &mut summary.times),
            VR::UI => ('\0', value::uid, |summary| &mut summary.uids),
            _ => {
                if let Value::Sequence(sequence) = element.value_mut() {
                    for item in sequence.items_mut() {
                        apply(item, summary);
                    }
                }
                continue;
            }
        };
        if let Value::Primitive(stored) = element.value_mut()
            && let Some((value, values)) = rewritten(stored, padding, repair)
        {
            *stored = value;
            *count(summary) += values;
        }
    }
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
/// sequence nests beyond the supported repair depth.
///
/// # Examples
/// ```
/// use ritk_dicom::dataset::{DataElement, PrimitiveValue, DicomTag, VR};
/// use ritk_dicom::dataset::Dataset;
/// use ritk_io::format::dicom::repair::repair_values;
///
/// let study_date = DicomTag::new(0x0008, 0x0020);
/// let mut object = Dataset::new();
/// object.put(DataElement::new(study_date, VR::DA, PrimitiveValue::from("1994.07.21")));
///
/// let summary = repair_values(&mut object)?;
///
/// assert_eq!(summary.dates, 1);
/// assert_eq!(object.element(study_date)?.to_str()?, "19940721");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn repair_values(object: &mut Dataset) -> Result<RepairSummary, RepairError> {
    check_depth(object)?;
    let mut summary = RepairSummary::default();
    apply(object, &mut summary);
    Ok(summary)
}
