use super::{apply_app_event, viewport};
use crate::app::SnapApp;
use crate::app::tests::test_volume;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};
use crate::tools::{
    interaction::{Annotation, ImagePoint},
    kind::ToolKind,
};
use metis_ui_lang::PathCapture;

#[test]
fn closed_polygon_commits_area_perimeter_and_slice_location() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    app.select_tool(ToolKind::RoiClosedPolygon);
    let viewport = viewport([64, 64]);
    let nodes = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)];
    let first = nodes[0];

    for (x, y) in [first].into_iter().chain(nodes).chain([first]) {
        apply_app_event(
            &mut app,
            &viewport,
            PresentationEvent::PointerDown {
                x,
                y,
                button: PointerButton::Left,
                modifiers: PresentationModifiers::NONE,
            },
        );
        apply_app_event(
            &mut app,
            &viewport,
            PresentationEvent::PointerUp {
                x,
                y,
                button: PointerButton::Left,
            },
        );
    }

    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::ClosedPolygon {
            points,
            area_mm2,
            perimeter_mm,
            location: Some(location),
        }] if points.len() == 4
            && *area_mm2 == 400.0
            && *perimeter_mm == 80.0
            && *location == crate::tools::interaction::AnnotationLocation { plane: 0, slice: 0 }
    ));
}

#[test]
fn pending_polygon_anchor_is_discarded_after_slice_change() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([2, 64, 64]));
    app.select_tool(ToolKind::RoiClosedPolygon);
    let viewport = viewport([64, 64]);
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerDown {
            x: 10.0,
            y: 10.0,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
    );
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerUp {
            x: 10.0,
            y: 10.0,
            button: PointerButton::Left,
        },
    );
    app.viewer_state.slice_index = 1;
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerDown {
            x: 30.0,
            y: 10.0,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
    );
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::PointerUp {
            x: 30.0,
            y: 10.0,
            button: PointerButton::Left,
        },
    );
    assert!(app.annotations.is_empty());
    assert_eq!(
        app.tool_state,
        crate::tools::interaction::ToolState::Polygon {
            points: PathCapture::new(ImagePoint::new(30.0, 10.0)),
            kind: crate::tools::interaction::PolygonKind::Closed,
        }
    );
}

fn polygon_press(x: f64, y: f64) -> PresentationEvent {
    PresentationEvent::PointerDown {
        x,
        y,
        button: PointerButton::Left,
        modifiers: PresentationModifiers::NONE,
    }
}

fn polygon_release(x: f64, y: f64) -> PresentationEvent {
    PresentationEvent::PointerUp {
        x,
        y,
        button: PointerButton::Left,
    }
}

#[test]
fn open_polygon_enter_commits_physical_length_and_consumes_release() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    app.select_tool(ToolKind::RoiOpenPolygon);
    let viewport = viewport([64, 64]);
    for event in [
        polygon_press(10.0, 10.0),
        polygon_release(10.0, 10.0),
        polygon_press(13.0, 14.0),
        polygon_release(13.0, 14.0),
        PresentationEvent::KeyDown {
            virtual_key: 0x0d,
            repeated: true,
            modifiers: PresentationModifiers::NONE,
        },
    ] {
        apply_app_event(&mut app, &viewport, event);
    }
    assert_eq!(app.annotations.as_slice(), &[]);
    for event in [
        polygon_press(30.0, 30.0),
        PresentationEvent::KeyDown {
            virtual_key: 0x0d,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
        polygon_release(30.0, 30.0),
    ] {
        apply_app_event(&mut app, &viewport, event);
    }
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon {
            points,
            length_mm,
            location: Some(location),
        }] if points == &[[10.0, 10.0], [14.0, 13.0]]
            && *length_mm == 5.0
            && *location == crate::tools::interaction::AnnotationLocation { plane: 0, slice: 0 }
    ));
    assert_eq!(app.tool_state, crate::tools::interaction::ToolState::Idle);
}

#[test]
fn escape_cancels_open_polygon_without_annotation() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    app.select_tool(ToolKind::RoiOpenPolygon);
    let viewport = viewport([64, 64]);
    for event in [
        polygon_press(10.0, 10.0),
        polygon_release(10.0, 10.0),
        polygon_press(20.0, 20.0),
    ] {
        apply_app_event(&mut app, &viewport, event);
    }
    assert_eq!(
        app.tool_state,
        crate::tools::interaction::ToolState::Polygon {
            points: PathCapture::new(ImagePoint::new(10.0, 10.0)),
            kind: crate::tools::interaction::PolygonKind::Open,
        }
    );
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
    );
    apply_app_event(&mut app, &viewport, polygon_release(20.0, 20.0));
    assert_eq!(app.annotations.as_slice(), &[]);
    assert_eq!(app.tool_state, crate::tools::interaction::ToolState::Idle);
}

#[test]
fn open_polygon_anchor_is_discarded_after_slice_change() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([2, 64, 64]));
    app.select_tool(ToolKind::RoiOpenPolygon);
    let viewport = viewport([64, 64]);
    apply_app_event(&mut app, &viewport, polygon_press(10.0, 10.0));
    apply_app_event(&mut app, &viewport, polygon_release(10.0, 10.0));
    app.viewer_state.slice_index = 1;
    apply_app_event(&mut app, &viewport, polygon_press(30.0, 10.0));
    apply_app_event(&mut app, &viewport, polygon_release(30.0, 10.0));
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::KeyDown {
            virtual_key: 0x0d,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
    );
    assert_eq!(app.annotations.as_slice(), &[]);
    assert_eq!(
        app.tool_state,
        crate::tools::interaction::ToolState::Polygon {
            points: PathCapture::new(ImagePoint::new(30.0, 10.0)),
            kind: crate::tools::interaction::PolygonKind::Open,
        }
    );
}
