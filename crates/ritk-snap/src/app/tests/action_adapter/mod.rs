//! End-to-end checks for the format-neutral presentation-to-viewer seam.

use crate::app::action_adapter::ViewerViewport;
use crate::app::SnapApp;
use crate::presentation::{PresentationDispatcher, PresentationEvent};
use crate::ui::ViewTransform;

mod coordinates;
mod gestures;
mod lifecycle;
mod wheel;

fn viewport(source_size: [usize; 2]) -> ViewerViewport {
    ViewerViewport::new(
        0,
        [0.0, 0.0],
        [1.0, 1.0],
        source_size,
        ViewTransform::default(),
    )
    .expect("test viewport geometry is valid")
}

fn apply_events(
    app: &mut SnapApp,
    dispatcher: &mut PresentationDispatcher,
    viewport: &ViewerViewport,
    event: PresentationEvent,
) {
    let actions = dispatcher
        .dispatch(&[event])
        .expect("test presentation event is valid");
    for action in actions.iter() {
        app.apply_viewer_action(action, Some(viewport))
            .expect("test viewer action is supported");
    }
}

fn apply_app_event(app: &mut SnapApp, viewport: &ViewerViewport, event: PresentationEvent) {
    app.apply_presentation_events(&[event], Some(viewport))
        .expect("test presentation event is supported");
}
