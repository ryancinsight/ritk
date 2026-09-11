use crate::app::browser_semantics::{BrowserCanvasSemantics, BrowserLoadState};
use crate::presentation::PresentationFrame;

#[test]
fn empty_canvas_publishes_bounded_empty_state() {
    let state = BrowserCanvasSemantics::from_state(false, 0, 0, 1, None);

    assert_eq!(state.load_state, BrowserLoadState::Empty);
    assert_eq!(state.load_state_value(), "empty");
    assert_eq!(state.frame_state_value(), "empty");
    assert_eq!(state.frame_dimensions_or_zero(), (0, 0));
    assert_eq!(
        (state.axis, state.slice_index, state.slice_count),
        (0, 0, 1)
    );
}

#[test]
fn presented_canvas_preserves_axis_slice_and_frame_dimensions() {
    let frame = PresentationFrame::from_rgba(4, 3, &[17; 4 * 3 * 4]).expect("valid frame");
    let state = BrowserCanvasSemantics::from_state(true, 2, 5, 9, Some(&frame));

    assert_eq!(state.load_state, BrowserLoadState::Ready);
    assert_eq!(state.load_state_value(), "ready");
    assert_eq!(state.frame_state_value(), "presented");
    assert_eq!(state.frame_dimensions_or_zero(), (4, 3));
    assert_eq!(
        (state.axis, state.slice_index, state.slice_count),
        (2, 5, 9)
    );
}
