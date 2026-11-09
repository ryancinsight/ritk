use super::*;

/// L key maps to MeasureLength.
#[test]
fn l_key_selects_measure_length() {
    assert_eq!(
        tool_kind_for_virtual_key(u32::from(b'L')),
        Some(ToolKind::MeasureLength)
    );
}

/// A key maps to MeasureAngle.
#[test]
fn a_key_selects_measure_angle() {
    assert_eq!(
        tool_kind_for_virtual_key(u32::from(b'A')),
        Some(ToolKind::MeasureAngle)
    );
}

/// R key maps to RoiRect.
#[test]
fn r_key_selects_roi_rect() {
    assert_eq!(
        tool_kind_for_virtual_key(u32::from(b'R')),
        Some(ToolKind::RoiRect)
    );
}

/// E key maps to RoiEllipse.
#[test]
fn e_key_selects_roi_ellipse() {
    assert_eq!(
        tool_kind_for_virtual_key(u32::from(b'E')),
        Some(ToolKind::RoiEllipse)
    );
}

/// H key maps to PointHu.
#[test]
fn h_key_selects_point_hu() {
    assert_eq!(
        tool_kind_for_virtual_key(u32::from(b'H')),
        Some(ToolKind::PointHu)
    );
}

/// P key maps to Pan.
#[test]
fn p_key_selects_pan() {
    assert_eq!(
        tool_kind_for_virtual_key(u32::from(b'P')),
        Some(ToolKind::Pan)
    );
}

/// Z key maps to Zoom.
#[test]
fn z_key_selects_zoom() {
    assert_eq!(
        tool_kind_for_virtual_key(u32::from(b'Z')),
        Some(ToolKind::Zoom)
    );
}

/// W key maps to WindowLevel.
#[test]
fn w_key_selects_window_level() {
    assert_eq!(
        tool_kind_for_virtual_key(u32::from(b'W')),
        Some(ToolKind::WindowLevel)
    );
}

/// B key maps to LabelPaint.
#[test]
fn b_key_selects_label_paint() {
    assert_eq!(
        tool_kind_for_virtual_key(u32::from(b'B')),
        Some(ToolKind::LabelPaint)
    );
}

/// Unmapped keys return None.
#[test]
fn unmapped_key_returns_none() {
    assert_eq!(tool_kind_for_virtual_key(0x1b), None);
    assert_eq!(tool_kind_for_virtual_key(0x09), None);
    assert_eq!(tool_kind_for_virtual_key(u32::from(b'Q')), None);
    assert_eq!(tool_kind_for_virtual_key(u32::from(b'X')), None);
}

/// All mapped shortcuts are distinct (no accidental duplication).
#[test]
fn all_shortcuts_distinct() {
    let mapped: Vec<ToolKind> = b"LAREHPZWB"
        .iter()
        .filter_map(|key| tool_kind_for_virtual_key(u32::from(*key)))
        .collect();
    assert_eq!(mapped.len(), 9);
    for (index, tool) in mapped.iter().enumerate() {
        assert!(
            !mapped[index + 1..].contains(tool),
            "{tool:?} is duplicated"
        );
    }
}
