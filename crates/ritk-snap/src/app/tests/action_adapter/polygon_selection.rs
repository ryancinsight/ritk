use super::{apply_app_event, viewport};
use crate::app::tests::test_volume;
use crate::app::SnapApp;
use crate::presentation::{PointerButton, PresentationEvent, PresentationModifiers};
use crate::tools::interaction::{Annotation, AnnotationLocation};
use crate::tools::kind::ToolKind;

fn click(app: &mut SnapApp, view: &crate::app::viewer_viewport::ViewerViewport, x: f64, y: f64) {
    for event in [
        PresentationEvent::PointerDown {
            x,
            y,
            button: PointerButton::Left,
            modifiers: PresentationModifiers::NONE,
        },
        PresentationEvent::PointerUp {
            x,
            y,
            button: PointerButton::Left,
        },
    ] {
        apply_app_event(app, view, event);
    }
}

#[test]
fn pan_click_selects_a_visible_closing_edge_and_delete_keeps_other_slices() {
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([2, 64, 64]));
    app.select_tool(ToolKind::Pan);
    let view = viewport([64, 64]);
    let points = vec![[10.0, 10.0], [10.0, 48.0], [48.0, 48.0]];
    let adjacent = Annotation::OpenPolygon {
        location: Some(AnnotationLocation { plane: 0, slice: 1 }),
        points: points.clone(),
        length_mm: 76.0,
    };
    app.annotations.push(Annotation::ClosedPolygon {
        location: Some(AnnotationLocation { plane: 0, slice: 0 }),
        points,
        area_mm2: 722.0,
        perimeter_mm: 76.0 + 38.0_f32.hypot(38.0),
    });
    app.annotations.push(adjacent.clone());

    click(&mut app, &view, 29.0, 29.0);
    assert_eq!(app.pointer.selected_polygon, None);
    let view = view.with_polygon_handles();
    for modifiers in [
        PresentationModifiers::new(false, false, true, false),
        PresentationModifiers::new(false, false, false, true),
    ] {
        apply_app_event(
            &mut app,
            &view,
            PresentationEvent::PointerDown {
                x: 29.0,
                y: 29.0,
                button: PointerButton::Left,
                modifiers,
            },
        );
        apply_app_event(
            &mut app,
            &view,
            PresentationEvent::PointerUp {
                x: 29.0,
                y: 29.0,
                button: PointerButton::Left,
            },
        );
        assert_eq!(app.pointer.selected_polygon, None);
    }

    click(&mut app, &view, 29.0, 29.0);
    assert_eq!(app.pointer.selected_polygon, Some(0));

    let delete = || PresentationEvent::KeyDown {
        virtual_key: 0x2e,
        repeated: false,
        modifiers: PresentationModifiers::NONE,
    };
    app.set_slice_for_axis(0, 1);
    apply_app_event(&mut app, &view, delete());
    assert_eq!(app.annotations.len(), 2);
    assert_eq!(app.pointer.selected_polygon, None);

    app.set_slice_for_axis(0, 0);
    click(&mut app, &view, 29.0, 29.0);
    apply_app_event(&mut app, &view, delete());
    assert_eq!(app.annotations, [adjacent]);
}

#[test]
fn polygon_pick_radius_is_measured_in_projected_screen_pixels() {
    use crate::app::viewer_viewport::ViewerViewport;
    use crate::tools::interaction::ViewportOffset;
    use crate::ui::ViewTransform;

    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([1, 64, 64]));
    app.select_tool(ToolKind::Pan);
    app.annotations.push(Annotation::OpenPolygon {
        location: Some(AnnotationLocation { plane: 0, slice: 0 }),
        points: vec![[24.0, 24.0], [24.0, 40.0]],
        length_mm: 16.0,
    });
    let view = ViewerViewport::new_with_zoom_pan(
        0,
        [0.0; 2],
        [2.0, 1.0],
        [64, 64],
        ViewTransform::default(),
        2.0,
        ViewportOffset::new(0.0, 0.0),
    )
    .expect("test viewport geometry")
    .with_polygon_handles();
    // The horizontal edge projects to y = 32 + 2 * (24 - 32) = 16.
    click(&mut app, &view, 48.0, 23.0);
    assert_eq!(app.pointer.selected_polygon, None);
    click(&mut app, &view, 48.0, 22.0);
    assert_eq!(app.pointer.selected_polygon, Some(0));
}
