//! Reference cases for pan displacement in host presentation units.

use super::super::test_volume;
use super::apply_app_event;
use crate::app::browser_geometry::viewport_for_display_with_zoom_pan;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};
use crate::tools::interaction::ViewportOffset;
use crate::tools::kind::ToolKind;

fn pointer_down(x: f64, y: f64) -> PresentationEvent {
    PresentationEvent::PointerDown {
        x,
        y,
        button: PointerButton::Left,
        modifiers: PresentationModifiers::NONE,
    }
}

fn pointer_move(x: f64, y: f64) -> PresentationEvent {
    PresentationEvent::PointerMove { x, y }
}

fn pointer_up(x: f64, y: f64) -> PresentationEvent {
    PresentationEvent::PointerUp {
        x,
        y,
        button: PointerButton::Left,
    }
}

#[test]
fn browser_identity_pan_matches_backing_raster_displacement() {
    const BACKING_SIZE: [u32; 2] = [100, 100];
    let viewport = viewport_for_display_with_zoom_pan(
        0,
        [1.0; 2],
        BACKING_SIZE,
        1.0,
        ViewportOffset::new(0.0, 0.0),
    )
    .expect("identity browser viewport");
    let before = viewport.project([0.0, 0.0]);
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 100, 100]));
    app.active_tool = ToolKind::Pan;
    for event in [
        pointer_down(0.5, 0.5),
        pointer_move(0.74, 0.38),
        pointer_up(0.74, 0.38),
    ] {
        apply_app_event(&mut app, &viewport, event);
    }
    assert_eq!(app.pan_offset, ViewportOffset::new(24.0, -12.0));
    let after =
        viewport_for_display_with_zoom_pan(0, [1.0; 2], BACKING_SIZE, app.zoom, app.pan_offset)
            .expect("panned browser viewport")
            .project([0.0, 0.0]);
    let rendered = [
        (after[0] - before[0]) * f64::from(BACKING_SIZE[0]),
        (after[1] - before[1]) * f64::from(BACKING_SIZE[1]),
    ];
    assert_eq!(rendered, [24.0, -12.0]);
}
