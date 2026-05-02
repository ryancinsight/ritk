use super::*;

#[test]
fn new_editor_has_default_foreground_label_and_background_volume() {
    let editor = LabelEditor::new([2, 3, 4]);

    assert_eq!(editor.active_label_id(), 1);
    assert_eq!(editor.current_map().num_voxels(), 24);
    assert_eq!(editor.current_map().count_label(0), 24);
    assert_eq!(editor.current_map().count_label(1), 0);

    let entry = editor
        .current_map()
        .table
        .get_label(1)
        .expect("default label must be present");
    assert_eq!(entry.name, "Label 1");
    assert_eq!(entry.color, [255, 0, 0, 180]);
    assert!(entry.visible);
}

#[test]
fn paint_voxel_sets_active_label_and_supports_undo_redo() {
    let mut editor = LabelEditor::new([2, 3, 4]);

    let changed = editor.paint_voxel([1, 2, 3]).expect("paint must succeed");
    assert_eq!(changed, 1);
    assert_eq!(editor.current_map().label_at([1, 2, 3]), 1);
    assert_eq!(editor.current_map().count_label(1), 1);
    assert!(editor.can_undo());
    assert!(!editor.can_redo());

    assert!(editor.undo(), "undo must move to background state");
    assert_eq!(editor.current_map().label_at([1, 2, 3]), 0);
    assert_eq!(editor.current_map().count_label(1), 0);
    assert!(editor.can_redo());

    assert!(editor.redo(), "redo must restore painted voxel");
    assert_eq!(editor.current_map().label_at([1, 2, 3]), 1);
    assert_eq!(editor.current_map().count_label(0), 23);
}

#[test]
fn paint_sphere_radius_one_changes_center_and_axis_neighbors() {
    let mut editor = LabelEditor::new([3, 3, 3]);

    let changed = editor
        .paint_sphere([1, 1, 1], 1)
        .expect("radius-one paint must succeed");

    assert_eq!(changed, 7, "closed integer ball r=1 in 3-D has 7 voxels");
    for idx in [
        [1, 1, 1],
        [0, 1, 1],
        [2, 1, 1],
        [1, 0, 1],
        [1, 2, 1],
        [1, 1, 0],
        [1, 1, 2],
    ] {
        assert_eq!(editor.current_map().label_at(idx), 1, "{idx:?}");
    }
    assert_eq!(
        editor.current_map().label_at([0, 0, 0]),
        0,
        "diagonal distance sqrt(3) exceeds radius 1"
    );
    assert_eq!(editor.current_map().count_label(1), 7);
    assert_eq!(editor.current_map().count_label(0), 20);
}

#[test]
fn erase_sphere_restores_background_inside_brush() {
    let mut editor = LabelEditor::new([3, 3, 3]);
    assert_eq!(editor.paint_sphere([1, 1, 1], 1).unwrap(), 7);

    let erased = editor.erase_voxel([1, 1, 1]).expect("erase must succeed");

    assert_eq!(erased, 1);
    assert_eq!(editor.current_map().label_at([1, 1, 1]), 0);
    assert_eq!(editor.current_map().count_label(1), 6);
    assert_eq!(editor.current_map().count_label(0), 21);
}

#[test]
fn add_label_uses_next_free_id_sets_active_and_updates_visibility() {
    let mut editor = LabelEditor::new([1, 2, 3]);

    let id = editor
        .add_label("Tumor", [0, 255, 0, 200])
        .expect("adding a second label must succeed");
    assert_eq!(id, 2);
    assert_eq!(editor.active_label_id(), 2);
    assert_eq!(editor.current_map().table.len(), 2);

    let changed = editor.paint_voxel([0, 1, 2]).expect("paint must succeed");
    assert_eq!(changed, 1);
    assert_eq!(editor.current_map().label_at([0, 1, 2]), 2);
    assert_eq!(editor.label_counts(), vec![(0, 5), (2, 1)]);

    editor
        .set_label_visibility(2, false)
        .expect("visibility update must succeed");
    let entry = editor
        .current_map()
        .table
        .get_label(2)
        .expect("label 2 must remain present");
    assert!(!entry.visible);
}

#[test]
fn custom_table_rejects_background_or_absent_active_label() {
    let mut table = LabelTable::new();
    table.add_label(7, "Kidney", [0, 0, 255, 180]).unwrap();

    assert!(LabelEditor::with_table([1, 1, 1], table.clone(), 0).is_err());
    assert!(LabelEditor::with_table([1, 1, 1], table.clone(), 8).is_err());

    let editor = LabelEditor::with_table([1, 1, 1], table, 7).expect("label 7 exists");
    assert_eq!(editor.active_label_id(), 7);
    assert_eq!(
        editor.current_map().table.get_label(7).unwrap().name,
        "Kidney"
    );
}

#[test]
fn out_of_bounds_paint_returns_error_without_history_change() {
    let mut editor = LabelEditor::new([2, 2, 2]);
    let depth_before = editor.history_depth();

    let result = editor.paint_voxel([2, 0, 0]);

    assert!(result.is_err());
    assert_eq!(editor.history_depth(), depth_before);
    assert_eq!(editor.current_map().count_label(1), 0);
    assert_eq!(editor.current_map().count_label(0), 8);
}

#[test]
fn repeat_paint_noop_does_not_create_history_entry() {
    let mut editor = LabelEditor::new([1, 1, 1]);

    assert_eq!(editor.paint_voxel([0, 0, 0]).unwrap(), 1);
    let depth_after_first_paint = editor.history_depth();
    assert_eq!(editor.paint_voxel([0, 0, 0]).unwrap(), 0);

    assert_eq!(editor.history_depth(), depth_after_first_paint);
    assert_eq!(editor.current_map().label_at([0, 0, 0]), 1);
}
