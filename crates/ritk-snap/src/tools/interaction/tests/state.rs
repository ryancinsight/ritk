use super::*;
// ── ToolState ─────────────────────────────────────────────────────────────

/// `Idle` must report `is_idle() == true` and `tool_kind() == None`.
#[test]

fn test_tool_state_idle() {
    let state = ToolState::Idle;
    assert!(state.is_idle(), "Idle must report is_idle() = true");
    assert_eq!(state.tool_kind(), None, "Idle must have tool_kind() = None");
}

/// Each non-idle variant must report `is_idle() == false` and the
/// correct `ToolKind`.
#[test]
fn test_tool_state_non_idle_variants() {
    let cases: &[(ToolState, ToolKind)] = &[
        (
            ToolState::Panning {
                start: ImagePoint::new(0.0, 0.0),
                viewport_origin: ViewportOffset::new(0.0, 0.0),
            },
            ToolKind::Pan,
        ),
        (
            ToolState::Zooming {
                start: ImagePoint::new(0.0, 0.0),
                original_zoom: 1.0,
            },
            ToolKind::Zoom,
        ),
        (
            ToolState::WindowLevelDrag {
                start: ImagePoint::new(0.0, 0.0),
                original_center: 0.0,
                original_width: 1.0,
            },
            ToolKind::WindowLevel,
        ),
        (
            ToolState::MeasureLength1 {
                p1: ImagePoint::new(0.0, 0.0),
            },
            ToolKind::MeasureLength,
        ),
        (
            ToolState::PatientLength1 {
                p1: crate::geometry::PatientPointMm::try_new([0.0; 3])
                    .expect("finite patient point"),
            },
            ToolKind::MeasureLength,
        ),
        (
            ToolState::MeasureAngle2 {
                p1: ImagePoint::new(0.0, 0.0),
                p2: ImagePoint::new(1.0, 0.0),
            },
            ToolKind::MeasureAngle,
        ),
        (
            ToolState::RoiDrag {
                start: ImagePoint::new(0.0, 0.0),
                current: ImagePoint::new(1.0, 1.0),
                kind: RoiKind::Rect,
            },
            ToolKind::RoiRect,
        ),
        (
            ToolState::RoiDrag {
                start: ImagePoint::new(0.0, 0.0),
                current: ImagePoint::new(1.0, 1.0),
                kind: RoiKind::Ellipse,
            },
            ToolKind::RoiEllipse,
        ),
    ];

    for (state, expected_kind) in cases {
        assert!(!state.is_idle(), "{:?} must not be idle", state.tool_kind());
        assert_eq!(
            state.tool_kind(),
            Some(*expected_kind),
            "tool_kind() must return {:?}",
            expected_kind
        );
    }
}
