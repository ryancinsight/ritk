use super::*;
use crate::overlay::Visibility;

#[test]
fn test_label_table_add_and_get() {
    let mut table = LabelTable::new();
    table
        .add_label(1, "Brain", RgbaBytes::new(255, 0, 0, 255))
        .expect("infallible: validated precondition");
    let entry = table.get_label(1).expect("entry must be present");
    assert_eq!(entry.id, 1);
    assert_eq!(entry.name, "Brain");
    assert_eq!(entry.color, RgbaBytes::new(255, 0, 0, 255));
    assert_eq!(entry.visible, Visibility::Visible);
}

#[test]
fn test_label_table_duplicate_id_error() {
    let mut table = LabelTable::new();
    table
        .add_label(1, "Brain", RgbaBytes::new(255, 0, 0, 255))
        .expect("infallible: validated precondition");
    let result = table.add_label(1, "Duplicate", RgbaBytes::new(0, 0, 0, 255));
    assert!(result.is_err(), "duplicate id must return Err");
    let msg = result.unwrap_err();
    assert!(msg.contains("1"), "error message must mention the id");
}

#[test]
fn test_label_table_remove_present() {
    let mut table = LabelTable::new();
    table
        .add_label(3, "Liver", RgbaBytes::new(0, 255, 0, 255))
        .expect("infallible: validated precondition");
    let removed = table.remove_label(3);
    assert!(removed, "remove must return true for present label");
    assert!(
        table.get_label(3).is_none(),
        "label must be absent after removal"
    );
}

#[test]
fn test_label_table_remove_absent() {
    let mut table = LabelTable::new();
    let removed = table.remove_label(99);
    assert!(!removed, "remove must return false for absent label");
}

#[test]
fn test_label_table_set_visibility() {
    let mut table = LabelTable::new();
    table
        .add_label(2, "Kidney", RgbaBytes::new(0, 0, 255, 255))
        .expect("infallible: validated precondition");
    let ok = table.set_visibility(2, Visibility::Hidden);
    assert!(ok, "set_visibility must return true for present label");
    assert_eq!(
        table
            .get_label(2)
            .expect("infallible: validated precondition")
            .visible,
        Visibility::Hidden,
        "label must be invisible after set_visibility(Hidden)"
    );
}

#[test]
fn test_label_table_next_free_id_empty() {
    let table = LabelTable::new();
    assert_eq!(
        table.next_free_id(),
        1,
        "next_free_id on empty table must be 1"
    );
}

#[test]
fn test_label_table_next_free_id_with_gaps() {
    let mut table = LabelTable::new();
    table
        .add_label(1, "A", RgbaBytes::new(0, 0, 0, 255))
        .expect("infallible: validated precondition");
    table
        .add_label(3, "B", RgbaBytes::new(0, 0, 0, 255))
        .expect("infallible: validated precondition");
    table
        .add_label(4, "C", RgbaBytes::new(0, 0, 0, 255))
        .expect("infallible: validated precondition");
    // IDs present: {1, 3, 4}; smallest positive absent: 2
    assert_eq!(
        table.next_free_id(),
        2,
        "next_free_id must return 2 when 1,3,4 are occupied"
    );
}

#[test]
fn test_label_table_len_and_is_empty() {
    let mut table = LabelTable::new();
    assert!(table.is_empty(), "new table must be empty");
    assert_eq!(table.len(), 0);
    table
        .add_label(5, "Spleen", RgbaBytes::new(128, 0, 128, 255))
        .expect("infallible: validated precondition");
    assert!(!table.is_empty(), "table must be non-empty after add");
    assert_eq!(table.len(), 1);
}
