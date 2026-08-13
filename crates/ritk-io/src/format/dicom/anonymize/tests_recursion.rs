//! Sequence-recursion tests for DICOM anonymization (PS 3.15 Annex E).
//!
//! A DICOM data set is a tree. Applying the profile only to top-level elements
//! leaves identifying attributes nested inside sequences untouched while
//! reporting success, which is the failure these tests pin down.
//!
//! Each leak test asserts on *every* string value at *every* depth rather than
//! on a single attribute, so an identifier that survives anywhere in the tree
//! fails the test regardless of which element it ended up in.
//!
//! # Test Coverage Index (this file)
//! 1. nested_patient_name_is_removed
//! 2. nested_identifiers_are_removed_at_depth_two
//! 3. nested_uid_maps_consistently_with_top_level
//! 4. original_attributes_sequence_contents_are_removed
//! 5. every_item_of_a_multi_item_sequence_is_anonymized
//! 6. nested_private_tags_are_removed
//! 7. private_tags_removed_counts_every_level_against_the_output
//! 8. nested_statistics_accumulate_across_levels
//! 9. nesting_at_the_traversal_bound_is_fully_anonymized
//! 10. nesting_past_the_traversal_bound_is_reported_as_failure_not_success
//! 11. clinical_attributes_inside_sequences_are_preserved
//! 12. objects_without_sequences_are_unaffected

use super::{
    anonymize_object, AnonymizationProfile, AnonymizeError, AnonymizeOptions, CleaningPolicy,
    MAX_SEQUENCE_DEPTH,
};
use dicom::core::header::Header;
use dicom::core::value::{DataSetSequence, PrimitiveValue, Value};
use dicom::core::{DataElement, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::{FileDicomObject, InMemDicomObject};

// ─── Helpers ──────────────────────────────────────────────────────────────────

const PATIENT_NAME: Tag = Tag(0x0010, 0x0010);
const PATIENT_ID: Tag = Tag(0x0010, 0x0020);
const INSTITUTION_NAME: Tag = Tag(0x0008, 0x0080);
const ACCESSION_NUMBER: Tag = Tag(0x0008, 0x0050);
const STUDY_INSTANCE_UID: Tag = Tag(0x0020, 0x000D);
const REFERENCED_SOP_INSTANCE_UID: Tag = Tag(0x0008, 0x1155);
/// A sequence the profile keeps, so its items must be reached by recursion.
const REFERENCED_IMAGE_SEQUENCE: Tag = Tag(0x0008, 0x1140);
/// A second kept sequence, used to build a second nesting level.
const ANATOMIC_REGION_SEQUENCE: Tag = Tag(0x0008, 0x2218);
const ORIGINAL_ATTRIBUTES_SEQUENCE: Tag = Tag(0x0400, 0x0561);
const ROWS: Tag = Tag(0x0028, 0x0010);
const SLICE_THICKNESS: Tag = Tag(0x0018, 0x0050);

/// A text data element.
fn text(tag: Tag, vr: VR, value: &str) -> DataElement<InMemDicomObject> {
    DataElement::new(tag, vr, PrimitiveValue::from(value))
}

/// A sequence data element holding `items`.
fn sequence(tag: Tag, items: Vec<InMemDicomObject>) -> DataElement<InMemDicomObject> {
    DataElement::new(tag, VR::SQ, Value::Sequence(DataSetSequence::from(items)))
}

/// Attach a minimal, valid file meta table to `dataset`.
fn with_meta(dataset: InMemDicomObject) -> FileDicomObject<InMemDicomObject> {
    dataset
        .with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.4")
                .media_storage_sop_instance_uid("1.2.3.4.5")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("invariant: synthetic meta table is valid")
}

/// Options that exercise the full profile, including private-tag removal.
fn enhanced_options() -> AnonymizeOptions {
    AnonymizeOptions {
        profile: AnonymizationProfile::Enhanced,
        clean_private_tags: CleaningPolicy::Clean,
        ..AnonymizeOptions::default()
    }
}

/// Collect every primitive string value in the tree, at every depth.
fn all_values(dataset: &InMemDicomObject, out: &mut Vec<String>) {
    for element in dataset.iter() {
        match element.value() {
            Value::Primitive(primitive) => out.push(primitive.to_str().trim().to_owned()),
            Value::Sequence(sequence) => {
                for item in sequence.items() {
                    all_values(item, out);
                }
            }
            Value::PixelSequence(_) => {}
        }
    }
}

/// Whether `needle` survives anywhere in the object, at any depth.
fn leaked(object: &FileDicomObject<InMemDicomObject>, needle: &str) -> bool {
    let mut values = Vec::new();
    all_values(object, &mut values);
    values.iter().any(|value| value.contains(needle))
}

/// The first item of the sequence at `tag`.
fn first_item(object: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<InMemDicomObject> {
    match object.element(tag).ok()?.value() {
        Value::Sequence(sequence) => sequence.items().first().cloned(),
        _ => None,
    }
}

/// Read an attribute of a nested data set as trimmed text.
fn read_item(item: &InMemDicomObject, tag: Tag) -> Option<String> {
    item.element(tag)
        .ok()
        .and_then(|element| element.value().to_str().ok())
        .map(|value| value.trim().to_owned())
}

/// Count private elements (odd group, excluding the file meta group) across the
/// whole tree. Measuring the data set itself is what makes the reported count
/// checkable against reality rather than against another counter.
fn private_element_count(dataset: &InMemDicomObject) -> usize {
    dataset
        .iter()
        .map(|element| {
            let group = element.tag().group();
            let here = usize::from(group & 1 == 1 && group != 0x0002);
            match element.value() {
                Value::Sequence(sequence) => {
                    here + sequence
                        .items()
                        .iter()
                        .map(private_element_count)
                        .sum::<usize>()
                }
                _ => here,
            }
        })
        .sum()
}

/// Wrap `leaf` in `levels` nested sequences, placing it at traversal depth
/// `levels`; the top-level data set is depth 0.
fn nest(leaf: InMemDicomObject, levels: u32) -> InMemDicomObject {
    (0..levels).fold(leaf, |inner, _| {
        InMemDicomObject::from_element_iter([sequence(REFERENCED_IMAGE_SEQUENCE, vec![inner])])
    })
}

// ─── Recursion: identifying attributes ────────────────────────────────────────

#[test]
fn nested_patient_name_is_removed() {
    let item = InMemDicomObject::from_element_iter([text(PATIENT_NAME, VR::PN, "Doe^John")]);
    let object = with_meta(InMemDicomObject::from_element_iter([sequence(
        REFERENCED_IMAGE_SEQUENCE,
        vec![item],
    )]));

    let (result, _) =
        anonymize_object(object, &enhanced_options()).expect("anonymization must succeed");

    assert!(
        !leaked(&result, "Doe^John"),
        "PatientName nested inside a retained sequence must not survive anonymization"
    );
}

#[test]
fn nested_identifiers_are_removed_at_depth_two() {
    // Depth 2 specifically: an implementation that descends only one level
    // passes a depth-1 test while still leaking.
    let deep = InMemDicomObject::from_element_iter([
        text(PATIENT_NAME, VR::PN, "Deep^Patient"),
        text(INSTITUTION_NAME, VR::LO, "Mercy General Hospital"),
    ]);
    let middle = InMemDicomObject::from_element_iter([
        text(ACCESSION_NUMBER, VR::SH, "ACC-778899"),
        sequence(ANATOMIC_REGION_SEQUENCE, vec![deep]),
    ]);
    let object = with_meta(InMemDicomObject::from_element_iter([sequence(
        REFERENCED_IMAGE_SEQUENCE,
        vec![middle],
    )]));

    let (result, _) =
        anonymize_object(object, &enhanced_options()).expect("anonymization must succeed");

    for identifier in ["Deep^Patient", "Mercy General Hospital", "ACC-778899"] {
        assert!(
            !leaked(&result, identifier),
            "{identifier:?} nested two levels deep must not survive anonymization"
        );
    }
}

// ─── Recursion: referential integrity ─────────────────────────────────────────

#[test]
fn nested_uid_maps_consistently_with_top_level() {
    // The same source UID at two depths must receive one replacement, or
    // nested references dangle while top-level ones are rewritten.
    let shared = "1.2.840.113619.2.55.3.604688119.1";

    let item =
        InMemDicomObject::from_element_iter([text(REFERENCED_SOP_INSTANCE_UID, VR::UI, shared)]);
    let object = with_meta(InMemDicomObject::from_element_iter([
        text(STUDY_INSTANCE_UID, VR::UI, shared),
        sequence(REFERENCED_IMAGE_SEQUENCE, vec![item]),
    ]));

    let options = AnonymizeOptions {
        profile: AnonymizationProfile::BasicReplaceUids,
        ..AnonymizeOptions::default()
    };
    let (result, _) = anonymize_object(object, &options).expect("anonymization must succeed");

    let top = result
        .element(STUDY_INSTANCE_UID)
        .ok()
        .and_then(|element| element.value().to_str().ok())
        .map(|value| value.trim().to_owned())
        .expect("StudyInstanceUID must be present");

    let nested = first_item(&result, REFERENCED_IMAGE_SEQUENCE)
        .as_ref()
        .and_then(|item| read_item(item, REFERENCED_SOP_INSTANCE_UID))
        .expect("nested ReferencedSOPInstanceUID must be present");

    assert_ne!(top, shared, "the source UID must actually be replaced");
    assert_eq!(
        top, nested,
        "one source UID must map to one replacement UID at every nesting depth"
    );
    assert!(
        nested.starts_with("2.25."),
        "nested replacement must use the 2.25 UUID arc, got {nested}"
    );
}

#[test]
fn original_attributes_sequence_contents_are_removed() {
    // OriginalAttributesSequence archives the values a previous
    // de-identification replaced, so leaving it behind ships exactly the data
    // that was meant to be destroyed.
    let archived = InMemDicomObject::from_element_iter([
        text(PATIENT_NAME, VR::PN, "Original^Patient"),
        text(PATIENT_ID, VR::LO, "MRN-12345"),
    ]);
    let object = with_meta(InMemDicomObject::from_element_iter([
        text(PATIENT_NAME, VR::PN, "ANONYMOUS"),
        sequence(ORIGINAL_ATTRIBUTES_SEQUENCE, vec![archived]),
    ]));

    let (result, _) =
        anonymize_object(object, &enhanced_options()).expect("anonymization must succeed");

    assert!(
        !leaked(&result, "Original^Patient"),
        "archived original PatientName must not survive"
    );
    assert!(
        !leaked(&result, "MRN-12345"),
        "archived original PatientID must not survive"
    );
}

#[test]
fn every_item_of_a_multi_item_sequence_is_anonymized() {
    // A sequence commonly holds one item per referenced instance; stopping
    // after the first would leave the rest identifying.
    let items: Vec<InMemDicomObject> = (0..4)
        .map(|index| {
            InMemDicomObject::from_element_iter([text(
                PATIENT_NAME,
                VR::PN,
                &format!("Item{index}^Patient"),
            )])
        })
        .collect();
    let object = with_meta(InMemDicomObject::from_element_iter([sequence(
        REFERENCED_IMAGE_SEQUENCE,
        items,
    )]));

    let (result, _) =
        anonymize_object(object, &enhanced_options()).expect("anonymization must succeed");

    for index in 0..4 {
        let identifier = format!("Item{index}^Patient");
        assert!(
            !leaked(&result, &identifier),
            "{identifier} in sequence item {index} must not survive"
        );
    }
}

// ─── Recursion: private tags and statistics ───────────────────────────────────

#[test]
fn nested_private_tags_are_removed() {
    let item = InMemDicomObject::from_element_iter([
        text(Tag(0x0029, 0x0010), VR::LO, "ACME PRIVATE BLOCK"),
        text(Tag(0x0029, 0x1010), VR::LO, "vendor-payload-with-identity"),
    ]);
    let object = with_meta(InMemDicomObject::from_element_iter([sequence(
        REFERENCED_IMAGE_SEQUENCE,
        vec![item],
    )]));

    let (result, _) =
        anonymize_object(object, &enhanced_options()).expect("anonymization must succeed");

    assert!(
        !leaked(&result, "vendor-payload-with-identity"),
        "private attributes nested inside a sequence must be removed"
    );

    let remaining =
        first_item(&result, REFERENCED_IMAGE_SEQUENCE).expect("the sequence item must still exist");
    assert!(
        remaining.element(Tag(0x0029, 0x1010)).is_err(),
        "the nested private element must be gone"
    );
}

#[test]
fn private_tags_removed_counts_every_level_against_the_output() {
    // The count is checked against the elements the output data set actually
    // lost. A count taken from the surviving candidates rather than from the
    // removals would credit work the object does not show.
    let deep =
        InMemDicomObject::from_element_iter([text(Tag(0x0029, 0x1010), VR::LO, "deep-payload")]);
    let middle = InMemDicomObject::from_element_iter([
        text(Tag(0x0019, 0x1001), VR::LO, "middle-payload"),
        sequence(ANATOMIC_REGION_SEQUENCE, vec![deep]),
    ]);
    let object = with_meta(InMemDicomObject::from_element_iter([
        text(Tag(0x0009, 0x0010), VR::LO, "top-payload"),
        sequence(REFERENCED_IMAGE_SEQUENCE, vec![middle]),
    ]));

    let before = private_element_count(&object);
    assert_eq!(
        before, 3,
        "the fixture carries one private element per level"
    );

    let (result, stats) =
        anonymize_object(object, &enhanced_options()).expect("anonymization must succeed");

    let after = private_element_count(&result);
    assert_eq!(after, 0, "no private element may survive at any depth");
    assert_eq!(
        stats.private_tags_removed,
        before - after,
        "the report must credit exactly the private elements the output lost"
    );
}

#[test]
fn nested_statistics_accumulate_across_levels() {
    let item = InMemDicomObject::from_element_iter([
        text(PATIENT_NAME, VR::PN, "Doe^John"),
        text(INSTITUTION_NAME, VR::LO, "Mercy General Hospital"),
    ]);
    let object = with_meta(InMemDicomObject::from_element_iter([
        text(PATIENT_NAME, VR::PN, "Doe^John"),
        sequence(REFERENCED_IMAGE_SEQUENCE, vec![item]),
    ]));

    let (_, stats) =
        anonymize_object(object, &enhanced_options()).expect("anonymization must succeed");

    // Two PatientName replacements (top level and nested) plus the nested
    // InstitutionName removal; counting only the top level would under-report.
    assert!(
        stats.tags_zeroed >= 2,
        "statistics must count nested replacements, got tags_zeroed={}",
        stats.tags_zeroed
    );
    assert!(
        stats.tags_deleted >= 1,
        "statistics must count nested removals, got tags_deleted={}",
        stats.tags_deleted
    );
}

// ─── Recursion: traversal bound ───────────────────────────────────────────────

#[test]
fn nesting_at_the_traversal_bound_is_fully_anonymized() {
    // Reaching the bound is not itself a truncation: nothing lies below this
    // leaf, so the walk is complete and must report success.
    let leaf = InMemDicomObject::from_element_iter([text(PATIENT_NAME, VR::PN, "Deep^Patient")]);
    let object = with_meta(nest(leaf, MAX_SEQUENCE_DEPTH));

    let (result, stats) = anonymize_object(object, &enhanced_options())
        .expect("nesting that reaches the bound without exceeding it must succeed");

    assert!(
        !leaked(&result, "Deep^Patient"),
        "the identifier sitting at the traversal bound must be replaced"
    );
    assert_eq!(
        stats.tags_zeroed, 1,
        "exactly the one pre-existing PatientName is suppressed; the placeholders \
         written into the intervening levels replaced nothing"
    );
}

#[test]
fn nesting_past_the_traversal_bound_is_reported_as_failure_not_success() {
    // The walk stops at the bound with a data set still below it, so the leaf
    // identifier survives. Reporting success would certify a data set that
    // still carries it — the report is the only evidence a caller has.
    let leaf = InMemDicomObject::from_element_iter([text(PATIENT_NAME, VR::PN, "Deep^Patient")]);
    let object = with_meta(nest(leaf, MAX_SEQUENCE_DEPTH + 1));

    let Err(error) = anonymize_object(object, &enhanced_options()) else {
        panic!("an object nested past the traversal bound must not report success");
    };

    assert_eq!(
        error.downcast_ref::<AnonymizeError>(),
        Some(&AnonymizeError::SequenceTooDeep {
            depth: MAX_SEQUENCE_DEPTH
        }),
        "the truncated traversal must surface as a typed error naming the depth"
    );
}

// ─── Non-regression ───────────────────────────────────────────────────────────

#[test]
fn clinical_attributes_inside_sequences_are_preserved() {
    // Recursion must not remove attributes the profile does not list, or the
    // output is de-identified but clinically useless.
    let item = InMemDicomObject::from_element_iter([
        text(PATIENT_NAME, VR::PN, "Doe^John"),
        DataElement::new(ROWS, VR::US, PrimitiveValue::from(512_u16)),
        text(SLICE_THICKNESS, VR::DS, "3.0"),
    ]);
    let object = with_meta(InMemDicomObject::from_element_iter([sequence(
        REFERENCED_IMAGE_SEQUENCE,
        vec![item],
    )]));

    let (result, _) =
        anonymize_object(object, &enhanced_options()).expect("anonymization must succeed");

    let preserved =
        first_item(&result, REFERENCED_IMAGE_SEQUENCE).expect("the sequence item must still exist");

    assert!(
        preserved.element(ROWS).is_ok(),
        "Rows inside a sequence must be preserved"
    );
    assert_eq!(
        read_item(&preserved, SLICE_THICKNESS).as_deref(),
        Some("3.0"),
        "SliceThickness inside a sequence must be preserved unchanged"
    );
    assert!(!leaked(&result, "Doe^John"));
}

#[test]
fn objects_without_sequences_are_unaffected() {
    // The recursive walk must behave exactly as the previous top-level-only
    // implementation did on objects that contain no sequences.
    let object = with_meta(InMemDicomObject::from_element_iter([
        text(PATIENT_NAME, VR::PN, "Doe^John"),
        text(INSTITUTION_NAME, VR::LO, "Mercy General Hospital"),
        text(SLICE_THICKNESS, VR::DS, "3.0"),
    ]));

    let (result, stats) =
        anonymize_object(object, &enhanced_options()).expect("anonymization must succeed");

    assert!(!leaked(&result, "Doe^John"));
    assert!(!leaked(&result, "Mercy General Hospital"));
    assert_eq!(
        result
            .element(SLICE_THICKNESS)
            .ok()
            .and_then(|e| e.value().to_str().ok())
            .map(|v| v.trim().to_owned())
            .as_deref(),
        Some("3.0"),
        "unlisted attributes must be preserved"
    );
    assert!(stats.tags_zeroed >= 1);
}
