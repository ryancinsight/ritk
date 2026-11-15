//! Rewrites applied through a data set and its nested items.

use super::{repair_values, RepairError, RepairSummary};
use dicom::core::value::DataSetSequence;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::InMemDicomObject;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const STUDY_DATE: Tag = Tag(0x0008, 0x0020);
const STUDY_TIME: Tag = Tag(0x0008, 0x0030);
const STUDY_DESCRIPTION: Tag = Tag(0x0008, 0x1030);
const REFERENCED_SOP_UIDS: Tag = Tag(0x0008, 0x1155);
const REFERENCED_SERIES: Tag = Tag(0x0008, 0x1115);

fn text(tag: Tag, vr: VR, value: &str) -> DataElement<InMemDicomObject> {
    DataElement::new(tag, vr, PrimitiveValue::from(value))
}

fn nested_below(tag: Tag, item: InMemDicomObject) -> InMemDicomObject {
    let mut outer = InMemDicomObject::new_empty();
    outer.put(DataElement::new(
        tag,
        VR::SQ,
        DataSetSequence::from(vec![item]),
    ));
    outer
}

fn read(object: &InMemDicomObject, tag: Tag) -> Result<String, Box<dyn std::error::Error>> {
    Ok(object.element(tag)?.to_str()?.into_owned())
}

/// Every element of `object`, nested items included. Two data sets holding a
/// sequence never compare equal, because an undefined length equals nothing.
fn rendered(object: &InMemDicomObject) -> String {
    format!("{object:?}")
}

#[test]
fn values_are_rewritten_at_every_level_and_other_elements_are_untouched() -> TestResult {
    let mut item = InMemDicomObject::new_empty();
    item.put(text(STUDY_TIME, VR::TM, "10:20:30.5"));
    item.put(text(REFERENCED_SOP_UIDS, VR::UI, "1.02.3\\1.2.4\\1.2.05"));
    let mut object = nested_below(REFERENCED_SERIES, item);
    object.put(text(STUDY_DATE, VR::DA, "1994.07.21"));
    object.put(text(
        STUDY_DESCRIPTION,
        VR::LO,
        "seen 1994.07.21 at 10:20:30",
    ));

    let summary = repair_values(&mut object)?;

    assert_eq!(
        summary,
        RepairSummary {
            dates: 1,
            times: 1,
            uids: 2
        }
    );
    assert_eq!(read(&object, STUDY_DATE)?, "19940721");
    assert_eq!(
        read(&object, STUDY_DESCRIPTION)?,
        "seen 1994.07.21 at 10:20:30"
    );
    let items = object
        .element(REFERENCED_SERIES)?
        .items()
        .ok_or("the sequence lost its items")?;
    assert_eq!(read(&items[0], STUDY_TIME)?, "102030.5");
    assert_eq!(read(&items[0], REFERENCED_SOP_UIDS)?, "1.2.3\\1.2.4\\1.2.5");
    Ok(())
}

#[test]
fn a_conformant_data_set_is_returned_unchanged() -> TestResult {
    let mut item = InMemDicomObject::new_empty();
    item.put(text(REFERENCED_SOP_UIDS, VR::UI, "1.2.3\\1.2.4"));
    let mut object = nested_below(REFERENCED_SERIES, item);
    object.put(text(STUDY_DATE, VR::DA, "19940721"));
    object.put(text(STUDY_TIME, VR::TM, "1020"));
    let before = rendered(&object);

    let summary = repair_values(&mut object)?;

    assert_eq!(summary, RepairSummary::default());
    assert_eq!(rendered(&object), before);
    Ok(())
}

/// A date held `depth` sequences below the root.
fn date_nested(depth: usize) -> InMemDicomObject {
    let mut object = InMemDicomObject::new_empty();
    object.put(text(STUDY_DATE, VR::DA, "1994.07.21"));
    (0..depth).fold(object, |inner, _| nested_below(REFERENCED_SERIES, inner))
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
