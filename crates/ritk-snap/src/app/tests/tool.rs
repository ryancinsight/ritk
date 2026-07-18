//! Tool interaction tests: zoom, pan, label undo/redo, window/level, shortcuts.

use super::*;
use crate::ui::tool_kind_for_key;

#[test]
fn zoom_tool_drag_updates_zoom_from_pointer_delta() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Zoom;
    app.zoom = 1.0;
    app.on_drag_start(Some(egui::pos2(100.0, 100.0)));
    app.on_drag(Some(egui::pos2(100.0, 80.0)));
    assert!(app.zoom > 1.0, "expected drag-up zoom-in, got {}", app.zoom);
    assert!(app.status_message.starts_with("Zoom:"));
}

#[test]
fn pan_tool_drag_updates_offset_via_ssot() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Pan;
    app.pan_offset = egui::Vec2::ZERO;
    app.on_drag_start(Some(egui::pos2(100.0, 100.0)));
    app.on_drag(Some(egui::pos2(130.0, 80.0)));
    // delta = (30.0, -20.0)
    // new_offset = (0, 0) + (30, -20) = (30, -20)
    assert_eq!(app.pan_offset.x, 30.0);
    assert_eq!(app.pan_offset.y, -20.0);
}

#[test]
fn pan_tool_drag_with_nonzero_starting_offset() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Pan;
    app.pan_offset = egui::Vec2::new(50.0, 75.0);
    app.on_drag_start(Some(egui::pos2(200.0, 150.0)));
    app.on_drag(Some(egui::pos2(220.0, 130.0)));
    // delta = (20.0, -20.0)
    // new_offset = (50, 75) + (20, -20) = (70, 55)
    assert_eq!(app.pan_offset.x, 70.0);
    assert_eq!(app.pan_offset.y, 55.0);
}

#[test]
fn pan_tool_drag_zero_delta_preserves_offset() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Pan;
    app.pan_offset = egui::Vec2::new(100.0, 100.0);
    app.on_drag_start(Some(egui::pos2(200.0, 150.0)));
    app.on_drag(Some(egui::pos2(200.0, 150.0)));
    // delta = (0.0, 0.0)
    // new_offset = (100, 100) + (0, 0) = (100, 100)
    assert_eq!(app.pan_offset.x, 100.0);
    assert_eq!(app.pan_offset.y, 100.0);
}

#[test]
fn label_shortcut_undo_redo_updates_map_and_status() {
    let mut app = SnapApp::default();
    let mut editor = crate::label::LabelEditor::new([2, 2, 2]);
    let _ = editor.paint_voxel([0, 0, 0]).expect("paint must succeed");
    app.label_editor = Some(editor);

    app.undo_label_edit_shortcut();
    let label_after_undo = app
        .label_editor
        .as_ref()
        .expect("editor")
        .current_map()
        .label_at([0, 0, 0]);
    assert_eq!(label_after_undo, 0);
    assert_eq!(app.status_message, "Segmentation undo.");

    app.redo_label_edit_shortcut();
    let label_after_redo = app
        .label_editor
        .as_ref()
        .expect("editor")
        .current_map()
        .label_at([0, 0, 0]);
    assert_eq!(label_after_redo, 1);
    assert_eq!(app.status_message, "Segmentation redo.");
}

/// Window/Level drag updates center and width through the SSOT mapping.
///
/// Drag (dx=+10, dy=-5) with default sensitivity 4.0:
/// new_width = 400 + 10*4 = 440
/// new_center = 40 âˆ’ (âˆ’5)*4 = 60
#[test]
fn window_level_drag_updates_center_and_width_via_ssot() {
    use crate::tools::interaction::ToolState;
    use egui::Pos2;

    let mut app = SnapApp::default();
    app.viewer_state.window_center = Some(40.0);
    app.viewer_state.window_width = Some(400.0);
    app.tool_state = ToolState::WindowLevelDrag {
        start: Pos2::new(100.0, 100.0),
        original_center: 40.0,
        original_width: 400.0 };
    app.on_drag(Some(Pos2::new(110.0, 95.0)));

    let new_center = app.viewer_state.window_center.expect("center set");
    let new_width = app.viewer_state.window_width.expect("width set");
    // Analytical: center = 40 âˆ’ (âˆ’5)*4 = 60, width = 400 + 10*4 = 440
    assert_eq!(new_center, 60.0_f32, "center mismatch");
    assert_eq!(new_width, 440.0_f32, "width mismatch");
    assert!(app.texture_dirty, "axial dirty not set");
    assert!(app.coronal_dirty, "coronal dirty not set");
    assert!(app.sagittal_dirty, "sagittal dirty not set");
    assert!(app.mip_dirty, "mip dirty not set");
}

/// Tool shortcut 'L' selects MeasureLength tool.
#[test]
fn tool_shortcut_l_selects_measure_length() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Pan; // different tool initially
    if let Some(tool) = tool_kind_for_key(egui::Key::L) {
        app.active_tool = tool;
    }
    assert_eq!(app.active_tool, ToolKind::MeasureLength);
}

/// Tool shortcut 'A' selects MeasureAngle tool.
#[test]
fn tool_shortcut_a_selects_measure_angle() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Pan;
    if let Some(tool) = tool_kind_for_key(egui::Key::A) {
        app.active_tool = tool;
    }
    assert_eq!(app.active_tool, ToolKind::MeasureAngle);
}

/// Tool shortcut 'R' selects RoiRect tool.
#[test]
fn tool_shortcut_r_selects_roi_rect() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Zoom;
    if let Some(tool) = tool_kind_for_key(egui::Key::R) {
        app.active_tool = tool;
    }
    assert_eq!(app.active_tool, ToolKind::RoiRect);
}

/// Tool shortcut 'E' selects RoiEllipse tool.
#[test]
fn tool_shortcut_e_selects_roi_ellipse() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::MeasureLength;
    if let Some(tool) = tool_kind_for_key(egui::Key::E) {
        app.active_tool = tool;
    }
    assert_eq!(app.active_tool, ToolKind::RoiEllipse);
}

/// Tool shortcut 'H' selects PointHu tool.
#[test]
fn tool_shortcut_h_selects_point_hu() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::WindowLevel;
    if let Some(tool) = tool_kind_for_key(egui::Key::H) {
        app.active_tool = tool;
    }
    assert_eq!(app.active_tool, ToolKind::PointHu);
}

/// Tool shortcut 'P' selects Pan tool.
#[test]
fn tool_shortcut_p_selects_pan() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Zoom;
    if let Some(tool) = tool_kind_for_key(egui::Key::P) {
        app.active_tool = tool;
    }
    assert_eq!(app.active_tool, ToolKind::Pan);
}

/// Tool shortcut 'Z' selects Zoom tool.
#[test]
fn tool_shortcut_z_selects_zoom() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::WindowLevel;
    if let Some(tool) = tool_kind_for_key(egui::Key::Z) {
        app.active_tool = tool;
    }
    assert_eq!(app.active_tool, ToolKind::Zoom);
}

/// Tool shortcut 'W' selects WindowLevel tool.
#[test]
fn tool_shortcut_w_selects_window_level() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Pan;
    if let Some(tool) = tool_kind_for_key(egui::Key::W) {
        app.active_tool = tool;
    }
    assert_eq!(app.active_tool, ToolKind::WindowLevel);
}

/// Tool shortcut 'B' selects LabelPaint tool.
#[test]
fn tool_shortcut_b_selects_label_paint() {
    let mut app = SnapApp::default();
    app.active_tool = ToolKind::Zoom;
    if let Some(tool) = tool_kind_for_key(egui::Key::B) {
        app.active_tool = tool;
    }
    assert_eq!(app.active_tool, ToolKind::LabelPaint);
}
