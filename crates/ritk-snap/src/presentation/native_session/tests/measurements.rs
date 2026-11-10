//! Native measurement line drawing and plane binding.

use super::super::layout::{measurement_overlay, MEASUREMENT_COLOR};
use super::*;
use crate::tools::interaction::{Annotation, AnnotationPlane, PlacedAnnotation};
use crate::tools::kind::ToolKind;
use metis_platform::native::MouseButton;
use metis_ui_lang::DisplayCommand;

fn drag(session: &mut NativeViewerSession, button: MouseButton, from: (i32, i32), to: (i32, i32)) {
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: from.0,
                y: from.1,
                button,
            },
            WindowEvent::PointerMove { x: to.0, y: to.1 },
            WindowEvent::PointerUp {
                x: to.0,
                y: to.1,
                button,
            },
        ])
        .expect("measurement gesture");
}

fn line_colors(commands: &[DisplayCommand]) -> Vec<metis_platform::Color> {
    commands
        .iter()
        .filter_map(|command| match command {
            DisplayCommand::DrawLine { color, .. } => Some(*color),
            _ => None,
        })
        .collect()
}

fn length_on(plane: AnnotationPlane) -> PlacedAnnotation {
    PlacedAnnotation::on_plane(
        plane,
        Annotation::Length {
            p1: [1.0, 1.0],
            p2: [1.0, 3.0],
            length_mm: 2.0,
        },
    )
}

#[test]
fn native_annotation_length_drag_records_its_plane_and_changes_the_frame() {
    let (mut session, _root) = session();
    session.app.active_tool = ToolKind::MeasureLength;
    let slice = session.app.axis_slice_info(0).0;
    let before = session.framebuffer.clone();
    let (x, y) = session.viewports[0].center();

    drag(&mut session, MouseButton::Left, (x, y), (x + 40, y));

    match session.app.annotations.as_slice() {
        [placed] => {
            assert_eq!(placed.plane, Some(AnnotationPlane { axis: 0, slice }));
            assert!(matches!(placed.annotation, Annotation::Length { .. }));
        }
        other => panic!("expected one placed annotation, found {other:?}"),
    }
    assert_ne!(session.framebuffer.pixels(), before.pixels());
}

#[test]
fn native_annotation_is_drawn_only_on_its_own_slice_and_axis() {
    let (session, _root) = session();
    let view = &session.views[0];
    let viewport = &session.viewports[0];
    let here = AnnotationPlane {
        axis: view.axis,
        slice: view.slice_index,
    };
    let other_slice = AnnotationPlane {
        axis: view.axis,
        slice: view.slice_index + 1,
    };
    let other_axis = AnnotationPlane {
        axis: (view.axis + 1) % 3,
        slice: view.slice_index,
    };

    let shown = measurement_overlay(view, viewport, &[length_on(here)]).expect("overlay");
    let hidden = measurement_overlay(
        view,
        viewport,
        &[length_on(other_slice), length_on(other_axis)],
    )
    .expect("overlay");

    assert_eq!(line_colors(shown.commands()), vec![MEASUREMENT_COLOR]);
    assert!(line_colors(hidden.commands()).is_empty());
}

#[test]
fn native_annotation_endpoints_use_the_pixel_projection() {
    let (session, _root) = session();
    let view = &session.views[0];
    let viewport = &session.viewports[0];
    let placed = length_on(AnnotationPlane {
        axis: view.axis,
        slice: view.slice_index,
    });

    let overlay = measurement_overlay(view, viewport, &[placed]).expect("overlay");

    let mapping = viewport.mapping();
    let image_point = |(x, y): (i32, i32)| {
        mapping
            .map(crate::presentation::ViewportPoint::new(
                f64::from(x),
                f64::from(y),
            ))
            .expect("the endpoint lies inside the image")
    };
    match overlay.commands() {
        [DisplayCommand::DrawLine { start, end, .. }] => {
            for (screen, column) in [(*start, 1.0_f32), (*end, 3.0)] {
                let image = image_point(screen);
                assert!((image.x() - column).abs() <= 0.5, "{image:?}");
                assert!((image.y() - 1.0).abs() <= 0.5, "{image:?}");
            }
        }
        other => panic!("expected one line, found {other:?}"),
    }
}
