use super::{apply_app_event, viewport};
use crate::app::tests::test_volume;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};
use crate::tools::interaction::{Annotation, AnnotationLocation};

fn app_with_polygon() -> SnapApp {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    app.annotations.push(Annotation::OpenPolygon {
        location: Some(AnnotationLocation { plane: 0, slice: 0 }),
        points: vec![[10.0, 10.0], [10.0, 30.0], [30.0, 30.0]],
        length_mm: 40.0,
    });
    app
}

fn press(app: &mut SnapApp, x: f64, y: f64) {
    apply_app_event(
        app,
        &viewport([64, 64]),
        PresentationEvent::PointerDown {
            x,
            y,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
    );
}

fn release(app: &mut SnapApp, x: f64, y: f64) {
    apply_app_event(
        app,
        &viewport([64, 64]),
        PresentationEvent::PointerUp {
            x,
            y,
            button: PointerButton::Left,
        },
    );
}

#[test]
fn selected_segment_click_inserts_and_recomputes_on_release() {
    let mut app = app_with_polygon();
    press(&mut app, 20.0, 10.0);
    release(&mut app, 20.0, 10.0);
    press(&mut app, 20.0, 10.0);
    release(&mut app, 20.0, 10.0);

    match app.annotations.as_slice() {
        [Annotation::OpenPolygon {
            points, length_mm, ..
        }] => {
            assert_eq!(points.len(), 4);
            assert_eq!(points[1], [10.0, 20.0]);
            assert_eq!(*length_mm, 40.0);
        }
        other => panic!("expected one edited open polygon, found {other:?}"),
    }
}

#[test]
fn selected_node_drag_commits_only_at_release() {
    let mut app = app_with_polygon();
    press(&mut app, 20.0, 10.0);
    release(&mut app, 20.0, 10.0);
    press(&mut app, 30.0, 10.0);
    apply_app_event(
        &mut app,
        &viewport([64, 64]),
        PresentationEvent::PointerMove { x: 35.0, y: 15.0 },
    );
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon { points, .. }] if points[1] == [10.0, 30.0]
    ));
    release(&mut app, 35.0, 15.0);
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon { points, .. }] if points[1] == [15.0, 35.0]
    ));
}

#[test]
fn polygon_edit_cancel_does_not_mutate_the_annotation() {
    let mut app = app_with_polygon();
    press(&mut app, 20.0, 10.0);
    release(&mut app, 20.0, 10.0);
    press(&mut app, 30.0, 10.0);
    apply_app_event(
        &mut app,
        &viewport([64, 64]),
        PresentationEvent::PointerMove { x: 35.0, y: 15.0 },
    );
    apply_app_event(
        &mut app,
        &viewport([64, 64]),
        PresentationEvent::PointerCancel {
            x: 35.0,
            y: 15.0,
            button: PointerButton::Left,
        },
    );
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon { points, .. }] if points[1] == [10.0, 30.0]
    ));
}

#[test]
fn escape_during_a_node_drag_ignores_the_trailing_release() {
    let mut app = app_with_polygon();
    let original = app.annotations.clone();
    press(&mut app, 20.0, 10.0);
    release(&mut app, 20.0, 10.0);
    press(&mut app, 30.0, 10.0);
    apply_app_event(
        &mut app,
        &viewport([64, 64]),
        PresentationEvent::PointerMove { x: 35.0, y: 15.0 },
    );
    apply_app_event(
        &mut app,
        &viewport([64, 64]),
        PresentationEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
    );
    release(&mut app, 35.0, 15.0);
    assert_eq!(app.annotations, original);
}

#[test]
fn losing_pointer_during_insertion_leaves_the_next_click_independent() {
    let mut app = app_with_polygon();
    let original = app.annotations.clone();
    press(&mut app, 20.0, 10.0);
    release(&mut app, 20.0, 10.0);
    press(&mut app, 20.0, 10.0);
    app.cancel_presentation_gesture();
    press(&mut app, 52.0, 52.0);
    release(&mut app, 52.0, 52.0);
    assert_eq!(app.annotations, original);
}
