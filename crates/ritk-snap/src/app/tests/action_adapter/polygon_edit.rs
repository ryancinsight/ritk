use super::{apply_app_event, viewport};
use crate::app::tests::test_volume;
use crate::app::viewer_viewport::ViewerViewport;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};
use crate::tools::interaction::{Annotation, AnnotationLocation};
use crate::tools::kind::ToolKind;

fn app_with_polygon() -> SnapApp {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([2, 64, 64]));
    app.select_tool(ToolKind::Pan);
    app.annotations.push(Annotation::OpenPolygon {
        location: Some(AnnotationLocation { plane: 0, slice: 0 }),
        points: vec![[0.0, 0.0], [0.0, 3.0], [4.0, 3.0]],
        length_mm: 7.0,
    });
    app
}

fn native_viewport() -> ViewerViewport {
    viewport([64, 64]).with_polygon_handles()
}

fn send(app: &mut SnapApp, event: PresentationEvent) {
    apply_app_event(app, &native_viewport(), event);
}

fn press(app: &mut SnapApp, x: f64, y: f64) {
    send(
        app,
        PresentationEvent::PointerDown {
            x,
            y,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
    );
}

fn release(app: &mut SnapApp, x: f64, y: f64) {
    send(
        app,
        PresentationEvent::PointerUp {
            x,
            y,
            button: PointerButton::Left,
        },
    );
}

#[test]
fn selected_vertex_drag_commits_recomputed_length_on_release() {
    let mut app = app_with_polygon();
    press(&mut app, 1.5, 0.0);
    release(&mut app, 1.5, 0.0);
    press(&mut app, 3.0, 0.0);
    send(&mut app, PresentationEvent::PointerMove { x: 6.0, y: 0.0 });
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon { points, .. }] if points[1] == [0.0, 3.0]
    ));
    release(&mut app, 6.0, 0.0);
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon { points, length_mm, .. }]
            if points[1] == [0.0, 6.0] && *length_mm == 11.0
    ));
}

fn begin_vertex_drag(app: &mut SnapApp) {
    press(app, 1.5, 0.0);
    release(app, 1.5, 0.0);
    press(app, 3.0, 0.0);
}

fn assert_unchanged(app: &SnapApp) {
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon { points, .. }] if points[1] == [0.0, 3.0]
    ));
}

#[test]
fn selected_vertex_drag_cancellation_modes_preserve_geometry() {
    let mut app = app_with_polygon();
    begin_vertex_drag(&mut app);
    app.set_slice_for_axis(0, 1);
    send(&mut app, PresentationEvent::PointerMove { x: 6.0, y: 0.0 });
    release(&mut app, 6.0, 0.0);
    assert_unchanged(&app);

    let mut app = app_with_polygon();
    begin_vertex_drag(&mut app);
    send(
        &mut app,
        PresentationEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
    );
    assert_unchanged(&app);

    let mut app = app_with_polygon();
    begin_vertex_drag(&mut app);
    send(
        &mut app,
        PresentationEvent::PointerCancel {
            x: 6.0,
            y: 0.0,
            button: PointerButton::Left,
        },
    );
    assert_unchanged(&app);
}
