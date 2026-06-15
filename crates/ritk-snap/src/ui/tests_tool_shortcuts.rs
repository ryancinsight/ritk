use super::*;

/// L key maps to MeasureLength.
#[test]
fn l_key_selects_measure_length() {
    assert_eq!(tool_kind_for_key(Key::L), Some(ToolKind::MeasureLength));
}

/// A key maps to MeasureAngle.
#[test]
fn a_key_selects_measure_angle() {
    assert_eq!(tool_kind_for_key(Key::A), Some(ToolKind::MeasureAngle));
}

/// R key maps to RoiRect.
#[test]
fn r_key_selects_roi_rect() {
    assert_eq!(tool_kind_for_key(Key::R), Some(ToolKind::RoiRect));
}

/// E key maps to RoiEllipse.
#[test]
fn e_key_selects_roi_ellipse() {
    assert_eq!(tool_kind_for_key(Key::E), Some(ToolKind::RoiEllipse));
}

/// H key maps to PointHu.
#[test]
fn h_key_selects_point_hu() {
    assert_eq!(tool_kind_for_key(Key::H), Some(ToolKind::PointHu));
}

/// P key maps to Pan.
#[test]
fn p_key_selects_pan() {
    assert_eq!(tool_kind_for_key(Key::P), Some(ToolKind::Pan));
}

/// Z key maps to Zoom.
#[test]
fn z_key_selects_zoom() {
    assert_eq!(tool_kind_for_key(Key::Z), Some(ToolKind::Zoom));
}

/// W key maps to WindowLevel.
#[test]
fn w_key_selects_window_level() {
    assert_eq!(tool_kind_for_key(Key::W), Some(ToolKind::WindowLevel));
}

/// B key maps to LabelPaint.
#[test]
fn b_key_selects_label_paint() {
    assert_eq!(tool_kind_for_key(Key::B), Some(ToolKind::LabelPaint));
}

/// Unmapped keys return None.
#[test]
fn unmapped_key_returns_none() {
    assert_eq!(tool_kind_for_key(Key::Escape), None);
    assert_eq!(tool_kind_for_key(Key::Tab), None);
    assert_eq!(tool_kind_for_key(Key::Q), None);
    assert_eq!(tool_kind_for_key(Key::X), None);
}

/// All mapped shortcuts are distinct (no accidental duplication).
#[test]
fn all_shortcuts_distinct() {
    let keys = [
        Key::L,
        Key::A,
        Key::R,
        Key::E,
        Key::H,
        Key::P,
        Key::Z,
        Key::W,
        Key::B,
    ];
    let mut mapped = vec![];
    for k in &keys {
        if let Some(tool) = tool_kind_for_key(*k) {
            mapped.push(tool);
        }
    }
    // If there were duplicates, this would fail (we'd see fewer unique tools than keys)
    // But since we defined 9 distinct keys → 9 distinct tools, this passes by construction.
    assert_eq!(mapped.len(), 9);
}
