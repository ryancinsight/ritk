use super::{apply_app_event, viewport};
use crate::app::tests::test_volume;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};
use crate::tools::{interaction::Annotation, kind::ToolKind};

#[test]
fn closed_polygon_commits_area_perimeter_and_slice_location() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    app.select_tool(ToolKind::RoiClosedPolygon);
    let viewport = viewport([64, 64]);
    let nodes = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)];
    let first = nodes[0];

    for (x, y) in nodes.into_iter().chain([first]) {
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
            points: metis_ui_lang::PathCapture::new(crate::tools::interaction::ImagePoint::new(
                10.0, 30.0
            ),),
            kind: crate::tools::interaction::PolygonKind::Closed,
        }
    );
}

#[test]
fn open_polygon_enter_commits_physical_length_and_location() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    app.select_tool(ToolKind::RoiOpenPolygon);
    let viewport = viewport([64, 64]);
    for (x, y) in [(10.0, 10.0), (13.0, 14.0)] {
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
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::KeyDown {
            virtual_key: 0x0d,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
    );
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::OpenPolygon {
            points,
            length_mm,
            location: Some(location),
        }] if points.len() == 2
            && *length_mm == 5.0
            && *location == crate::tools::interaction::AnnotationLocation { plane: 0, slice: 0 }
    ));
}

#[test]
fn escape_cancels_open_polygon_without_annotation() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    app.select_tool(ToolKind::RoiOpenPolygon);
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
        PresentationEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
    );
    assert!(app.annotations.is_empty());
    assert!(app.tool_state.is_idle());
}

#[test]
fn open_polygon_anchor_is_discarded_after_slice_change() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([2, 64, 64]));
    app.select_tool(ToolKind::RoiOpenPolygon);
    let viewport = viewport([64, 64]);
    for (x, y) in [(10.0, 10.0), (30.0, 10.0)] {
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
        if y == 10.0 && x == 10.0 {
            app.viewer_state.slice_index = 1;
        }
    }
    apply_app_event(
        &mut app,
        &viewport,
        PresentationEvent::KeyDown {
            virtual_key: 0x0d,
            repeated: false,
            modifiers: PresentationModifiers::NONE,
        },
    );
    assert!(app.annotations.is_empty());
}
