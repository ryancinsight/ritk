use crate::app::browser_semantics::{BrowserCanvasSemantics, BrowserLoadState};
use crate::presentation::{PresentationFrame, PresentationSnapshot};
use crate::render::WindowLevel;
use crate::tools::interaction::ViewportOffset;

fn snapshot(
    loaded: bool,
    axis: usize,
    slice_indices: [usize; 3],
    slice_counts: [usize; 3],
    window_preset_index: Option<usize>,
) -> PresentationSnapshot {
    PresentationSnapshot::from_parts(
        7,
        loaded,
        axis,
        slice_indices,
        slice_counts,
        WindowLevel::new(128.0, 256.0),
        false,
        12.0,
        1.0,
        ViewportOffset::new(0.0, 0.0),
        window_preset_index,
        2,
        "W/L",
    )
}

#[test]
fn empty_canvas_publishes_bounded_empty_state() {
    let state =
        BrowserCanvasSemantics::from_snapshot(snapshot(false, 0, [0, 0, 0], [1, 1, 1], None), None);

    assert_eq!(state.load_state(), BrowserLoadState::Empty);
    assert_eq!(state.load_state_value(), "empty");
    assert_eq!(state.frame_state_value(), "empty");
    assert_eq!(state.frame_dimensions_or_zero(), (0, 0));
    assert!(!state.snapshot.cine_enabled());
    assert_eq!(state.cine_enabled_value(), "false");
    assert_eq!(state.cine_fps_value(), "12");
    assert_eq!(state.window_center_value(), "128");
    assert_eq!(state.window_width_value(), "256");
    assert_eq!(state.window_preset_index_value(), "");
    assert_eq!(state.active_tool_index_value(), "2");
    assert_eq!(state.active_tool_name(), "W/L");
    assert_eq!(
        (state.axis(), state.slice_index(), state.slice_count()),
        (0, 0, 1)
    );
}

#[test]
fn presented_canvas_preserves_axis_slice_and_frame_dimensions() {
    let frame = PresentationFrame::from_rgba(4, 3, &[17; 4 * 3 * 4]).expect("valid frame");
    let state = BrowserCanvasSemantics::from_snapshot(
        PresentationSnapshot::from_parts(
            8,
            true,
            2,
            [0, 0, 5],
            [4, 6, 9],
            WindowLevel::new(500.0, 800.0),
            true,
            18.0,
            1.0,
            ViewportOffset::new(0.0, 0.0),
            Some(0),
            2,
            "W/L",
        ),
        Some(&frame),
    );

    assert_eq!(state.load_state(), BrowserLoadState::Ready);
    assert_eq!(state.load_state_value(), "ready");
    assert_eq!(state.frame_state_value(), "presented");
    assert_eq!(state.frame_dimensions_or_zero(), (4, 3));
    assert!(state.snapshot.cine_enabled());
    assert_eq!(state.cine_enabled_value(), "true");
    assert_eq!(state.cine_fps_value(), "18");
    assert_eq!(state.window_center_value(), "500");
    assert_eq!(state.window_width_value(), "800");
    assert_eq!(state.window_preset_index_value(), "0");
    assert_eq!(state.active_tool_index_value(), "2");
    assert_eq!(state.active_tool_name(), "W/L");
    assert_eq!(
        (state.axis(), state.slice_index(), state.slice_count()),
        (2, 5, 9)
    );
}
