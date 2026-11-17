use super::{apply_app_event, viewport};
use crate::app::tests::test_volume;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};
use crate::tools::{
    interaction::{Annotation, AnnotationLocation},
    kind::ToolKind,
};

#[test]
fn polygon_gestures_commit_length_and_area_on_the_current_slice() {
    let viewport = viewport([64, 64]);
    let nodes = [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)];
    let mut annotations = Vec::new();
    for (tool, finish) in [
        (ToolKind::RoiClosedPolygon, nodes[0]),
        (ToolKind::RoiOpenPolygon, nodes[3]),
    ] {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([64, 64, 1]));
        app.select_tool(tool);
        for (x, y) in nodes.into_iter().chain([finish]) {
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
        annotations.extend(app.annotations);
    }
    assert!(matches!(
        annotations.as_slice(),
        [
            Annotation::ClosedPolygon { area_mm2, perimeter_mm, .. },
            Annotation::OpenPolygon { length_mm, .. },
        ] if *area_mm2 == 400.0 && *perimeter_mm == 80.0 && *length_mm == 60.0
    ));
}

#[test]
fn dragging_a_completed_mark_moves_the_annotation_without_remeasuring() {
    let viewport = viewport([64, 64]);
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    let location = Some(AnnotationLocation { plane: 0, slice: 0 });
    app.annotations.push(Annotation::Length {
        plane: 0,
        location,
        p1: [10.0, 10.0],
        p2: [10.0, 30.0],
        length_mm: 20.0,
    });
    for event in [
        PresentationEvent::PointerDown {
            x: 20.0,
            y: 10.0,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
        PresentationEvent::PointerMove { x: 17.0, y: 14.0 },
        PresentationEvent::PointerUp {
            x: 17.0,
            y: 14.0,
            button: PointerButton::Left,
        },
    ] {
        apply_app_event(&mut app, &viewport, event);
    }
    assert_eq!(
        app.annotations.as_slice(),
        &[Annotation::Length {
            plane: 0,
            location,
            p1: [14.0, 7.0],
            p2: [14.0, 27.0],
            length_mm: 20.0,
        }]
    );
}
