use crate::app::browser_semantics::{BrowserCanvasSemantics, BrowserLoadState};
use crate::presentation::PresentationFrame;

#[test]
fn empty_canvas_publishes_bounded_empty_state() {
    let state =
        BrowserCanvasSemantics::from_state(false, 0, 0, 1, None, false, 12.0, 128.0, 256.0, None);

    assert_eq!(state.load_state, BrowserLoadState::Empty);
    assert_eq!(state.load_state_value(), "empty");
    assert_eq!(state.frame_state_value(), "empty");
    assert_eq!(state.frame_dimensions_or_zero(), (0, 0));
    assert!(!state.cine_enabled);
    assert_eq!(state.cine_enabled_value(), "false");
    assert_eq!(state.cine_fps_value(), "12");
    assert_eq!(state.window_center_value(), "128");
    assert_eq!(state.window_width_value(), "256");
    assert_eq!(state.window_preset_index_value(), "");
    assert_eq!(
        (state.axis, state.slice_index, state.slice_count),
        (0, 0, 1)
    );
}

#[test]
fn presented_canvas_preserves_axis_slice_and_frame_dimensions() {
    let frame = PresentationFrame::from_rgba(4, 3, &[17; 4 * 3 * 4]).expect("valid frame");
    let state = BrowserCanvasSemantics::from_state(
        true,
        2,
        5,
        9,
        Some(&frame),
        true,
        18.0,
        500.0,
        800.0,
        Some(0),
    );

    assert_eq!(state.load_state, BrowserLoadState::Ready);
    assert_eq!(state.load_state_value(), "ready");
    assert_eq!(state.frame_state_value(), "presented");
    assert_eq!(state.frame_dimensions_or_zero(), (4, 3));
    assert!(state.cine_enabled);
    assert_eq!(state.cine_enabled_value(), "true");
    assert_eq!(state.cine_fps_value(), "18");
    assert_eq!(state.window_center_value(), "500");
    assert_eq!(state.window_width_value(), "800");
    assert_eq!(state.window_preset_index_value(), "0");
    assert_eq!(
        (state.axis, state.slice_index, state.slice_count),
        (2, 5, 9)
    );
}
