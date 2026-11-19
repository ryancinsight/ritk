use super::{apply_app_event, viewport};
use crate::app::pointer_bindings::{AnnotationMoveKind, PendingAnnotationMove};
use crate::app::tests::test_volume;
use crate::app::viewer_viewport::ViewerViewport;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers, ViewportPoint};
use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint};
use crate::tools::kind::ToolKind;

fn app_with_polygon() -> SnapApp {
    app_with_annotation(Annotation::OpenPolygon {
        location: Some(AnnotationLocation { plane: 0, slice: 0 }),
        points: vec![[0.0, 0.0], [0.0, 3.0], [4.0, 3.0]],
        length_mm: 7.0,
    })
}
fn app_with_annotation(annotation: Annotation) -> SnapApp {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([2, 64, 64]));
    app.select_tool(ToolKind::Pan);
    app.annotations.push(annotation);
    app
}
fn app_with_long_open_polygon() -> SnapApp {
    app_with_annotation(Annotation::OpenPolygon {
        location: Some(AnnotationLocation { plane: 0, slice: 0 }),
        points: vec![[0.0, 0.0], [0.0, 20.0], [20.0, 20.0]],
        length_mm: 40.0,
    })
}
fn native_viewport() -> ViewerViewport {
    viewport([64, 64]).with_polygon_handles()
}
fn send(app: &mut SnapApp, event: PresentationEvent) {
    apply_app_event(app, &native_viewport(), event)
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
    assert_unchanged(&app);
    release(&mut app, 6.0, 0.0);
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon { points, length_mm, .. }]
            if points[1] == [0.0, 6.0] && *length_mm == 11.0
    ));
}
#[test]
fn selected_open_edge_click_inserts_a_projected_node() {
    let mut app = app_with_long_open_polygon();
    press(&mut app, 10.0, 0.0);
    release(&mut app, 10.0, 0.0);
    press(&mut app, 10.0, 0.0);
    release(&mut app, 10.0, 0.0);
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon { points, length_mm, .. }]
            if points.as_slice() == [[0.0, 0.0], [0.0, 10.0], [0.0, 20.0], [20.0, 20.0]]
                && *length_mm == 40.0
    ));
}
#[test]
fn pending_edge_insert_cancellation_preserves_geometry() {
    for event in [
        Some(PresentationEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        }),
        Some(PresentationEvent::PointerCancel {
            x: 5.0,
            y: 0.0,
            button: PointerButton::Left,
        }),
        None,
    ] {
        let mut app = app_with_long_open_polygon();
        press(&mut app, 10.0, 0.0);
        release(&mut app, 10.0, 0.0);
        press(&mut app, 10.0, 0.0);
        send(&mut app, PresentationEvent::PointerMove { x: 5.0, y: 0.0 });
        if let Some(event) = event {
            send(&mut app, event);
        } else {
            app.set_slice_for_axis(0, 1);
        }
        release(&mut app, 5.0, 0.0);
        assert!(matches!(
            app.annotations.as_slice(),
            [Annotation::OpenPolygon { points, .. }] if points.len() == 3
        ));
    }
}
#[test]
fn selected_closed_edge_drag_inserts_and_moves_a_node() {
    let mut app = app_with_annotation(Annotation::ClosedPolygon {
        location: Some(AnnotationLocation { plane: 0, slice: 0 }),
        points: vec![[0.0, 0.0], [0.0, 20.0], [20.0, 20.0], [20.0, 0.0]],
        area_mm2: 400.0,
        perimeter_mm: 80.0,
    });
    press(&mut app, 0.0, 10.0);
    release(&mut app, 0.0, 10.0);
    press(&mut app, 0.0, 10.0);
    send(&mut app, PresentationEvent::PointerMove { x: 0.0, y: 15.0 });
    release(&mut app, 0.0, 15.0);
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::ClosedPolygon { points, area_mm2, .. }]
            if points.as_slice() == [[0.0, 0.0], [0.0, 20.0], [20.0, 20.0], [20.0, 0.0], [15.0, 0.0]]
                && *area_mm2 == 400.0
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
    for event in [
        Some(PresentationEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        }),
        Some(PresentationEvent::PointerCancel {
            x: 6.0,
            y: 0.0,
            button: PointerButton::Left,
        }),
        None,
    ] {
        let mut app = app_with_polygon();
        begin_vertex_drag(&mut app);
        send(&mut app, PresentationEvent::PointerMove { x: 6.0, y: 0.0 });
        if let Some(event) = event {
            send(&mut app, event);
        } else {
            app.set_slice_for_axis(0, 1);
            app.set_slice_for_axis(0, 0);
        }
        release(&mut app, 6.0, 0.0);
        assert_unchanged(&app);
    }
}

#[test]
fn failed_admission_preserves_pending_transaction() {
    let mut app = app_with_polygon();
    app.pointer.selected_annotation = Some(0);
    let point = ImagePoint::new(3.0, 0.0);
    assert!(app.begin_annotation_move(native_viewport(), ViewportPoint::new(3.0, 0.0), point,));
    let original = app.annotations[0].clone();
    let pending = PendingAnnotationMove {
        annotation_index: 0,
        source_location: AnnotationLocation { plane: 0, slice: 0 },
        original: original.clone(),
        kind: AnnotationMoveKind::Polygon {
            node_index: 1,
            inserted: false,
        },
        candidate: original,
        start: point,
        current: point,
    };
    assert_eq!(app.pointer.pending_annotation_move, Some(pending.clone()));
    assert!(!app.begin_annotation_move(
        native_viewport(),
        ViewportPoint::new(63.0, 63.0),
        ImagePoint::new(63.0, 63.0),
    ));
    assert_eq!(app.pointer.pending_annotation_move, Some(pending));
}
