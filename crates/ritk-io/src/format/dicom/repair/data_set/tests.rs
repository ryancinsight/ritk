//! Rewrites applied through a data set and its nested items.

use super::{repair_values, RepairError, RepairSummary};
use ritk_dicom::dataset::Dataset;
use ritk_dicom::dataset::Sequence;
use ritk_dicom::dataset::{DataElement, DicomTag, PrimitiveValue, VR};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const STUDY_DATE: DicomTag = DicomTag::new(0x0008, 0x0020);
const STUDY_TIME: DicomTag = DicomTag::new(0x0008, 0x0030);
const STUDY_DESCRIPTION: DicomTag = DicomTag::new(0x0008, 0x1030);
const REFERENCED_SOP_UIDS: DicomTag = DicomTag::new(0x0008, 0x1155);
const REFERENCED_SERIES: DicomTag = DicomTag::new(0x0008, 0x1115);

fn text(tag: DicomTag, vr: VR, value: &str) -> DataElement {
    DataElement::new(tag, vr, PrimitiveValue::from(value))
}

fn sequence(tag: DicomTag, items: Vec<Dataset>) -> Dataset {
    let mut outer = Dataset::new();
    outer.put(DataElement::new(tag, VR::SQ, Sequence::from(items)));
    outer
}

/// The text an element stores, padding included: one entry for joined text,
/// one per value for separate values.
fn stored(object: &Dataset, tag: DicomTag) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    match object.element(tag)?.value().primitive() {
        Some(PrimitiveValue::Str(text)) => Ok(vec![text.clone()]),
        Some(PrimitiveValue::Strs(values)) => Ok(values.to_vec()),
        other => Err(format!("{tag} holds {other:?}, not text").into()),
    }
}

/// Every element of `object`, nested items included. Two data sets holding a
/// sequence never compare equal, because an undefined length equals nothing.
fn rendered(object: &Dataset) -> String {
    format!("{object:?}")
}

#[test]
fn values_are_rewritten_in_every_item_and_other_elements_are_untouched() -> TestResult {
    let mut first = Dataset::new();
    first.put(text(STUDY_TIME, VR::TM, "10:20:30.5"));
    first.put(text(REFERENCED_SOP_UIDS, VR::UI, "1.02.3\\1.2.4\\1.2.05"));
    let mut second = Dataset::new();
    second.put(text(STUDY_TIME, VR::TM, "11:00"));
    let mut object = sequence(REFERENCED_SERIES, vec![first, second]);
    object.put(text(STUDY_DATE, VR::DA, "1994.07.21 "));
    object.put(text(STUDY_DESCRIPTION, VR::LO, "1994.07.21"));

    let summary = repair_values(&mut object)?;

    assert_eq!(
        summary,
        RepairSummary {
            dates: 1,
            times: 2,
            uids: 2
        }
    );
    assert_eq!(stored(&object, STUDY_DATE)?, ["19940721"]);
    assert_eq!(object.element(STUDY_DATE)?.vr(), VR::DA);
    assert_eq!(stored(&object, STUDY_DESCRIPTION)?, ["1994.07.21"]);
    let items = object
        .element(REFERENCED_SERIES)?
        .items()
        .ok_or("the sequence lost its items")?;
    assert_eq!(stored(&items[0], STUDY_TIME)?, ["102030.5"]);
    assert_eq!(
        stored(&items[0], REFERENCED_SOP_UIDS)?,
        ["1.2.3\\1.2.4\\1.2.5"]
    );
    assert_eq!(stored(&items[1], STUDY_TIME)?, ["1100"]);
    Ok(())
}

#[test]
fn separate_values_stay_separate_and_unrewritten_ones_keep_their_bytes() -> TestResult {
    let mut object = Dataset::new();
    object.put(DataElement::new(
        REFERENCED_SOP_UIDS,
        VR::UI,
        PrimitiveValue::Strs(
            ["1.02.3", "1.2.4\0", "1.2.5 "]
                .into_iter()
                .map(String::from)
                .collect(),
        ),
    ));

    let summary = repair_values(&mut object)?;

    assert_eq!(summary.uids, 2);
    assert_eq!(
        stored(&object, REFERENCED_SOP_UIDS)?,
        ["1.2.3", "1.2.4\0", "1.2.5"]
    );
    Ok(())
}

#[test]
fn a_nontext_time_is_preserved() -> TestResult {
    let mut object = Dataset::new();
    let time = PrimitiveValue::from(102030u32);
    object.put(DataElement::new(STUDY_TIME, VR::TM, time.clone()));

    let summary = repair_values(&mut object)?;

    assert_eq!(summary, RepairSummary::default());
    assert_eq!(object.element(STUDY_TIME)?.value().primitive(), Some(&time));
    Ok(())
}

#[test]
fn a_conformant_data_set_is_returned_unchanged() -> TestResult {
    let mut item = Dataset::new();
    item.put(text(REFERENCED_SOP_UIDS, VR::UI, "1.2.3\\1.2.4"));
    let mut object = sequence(REFERENCED_SERIES, vec![item]);
    object.put(text(STUDY_DATE, VR::DA, "19940721 "));
    object.put(text(STUDY_TIME, VR::TM, "1020"));
    let before = rendered(&object);

    let summary = repair_values(&mut object)?;

    assert_eq!(summary, RepairSummary::default());
    assert_eq!(rendered(&object), before);
    Ok(())
}

/// A date held `depth` sequences below the root.
fn date_nested(depth: usize) -> Dataset {
    let mut object = Dataset::new();
    object.put(text(STUDY_DATE, VR::DA, "1994.07.21"));
    (0..depth).fold(object, |inner, _| sequence(REFERENCED_SERIES, vec![inner]))
}

#[test]
fn the_deepest_addressable_item_is_repaired_and_one_level_more_is_refused() -> TestResult {
    let mut deepest = date_nested(254);
    assert_eq!(
        repair_values(&mut deepest)?,
        RepairSummary {
            dates: 1,
            ..RepairSummary::default()
        }
    );

    let mut too_deep = date_nested(255);
    let before = rendered(&too_deep);
    assert_eq!(
        repair_values(&mut too_deep),
        Err(RepairError::NestedTooDeep { limit: 254 })
    );
    assert_eq!(rendered(&too_deep), before);
    Ok(())
}
