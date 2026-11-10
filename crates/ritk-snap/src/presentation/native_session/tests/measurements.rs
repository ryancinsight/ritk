//! Native measurement drawing and plane binding.

use super::super::layout::{
    measurement_label, measurement_overlay, LENGTH_LABEL_DISTANCE, MEASUREMENT_COLOR, ROI_COLOR,
};
use super::*;
use crate::tools::interaction::{Annotation, AnnotationPlane, PlacedAnnotation, ToolState};
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

    let shown =
        measurement_overlay(view, viewport, &[length_on(here)], &ToolState::Idle).expect("overlay");
    let hidden = measurement_overlay(
        view,
        viewport,
        &[length_on(other_slice), length_on(other_axis)],
        &ToolState::Idle,
    )
    .expect("overlay");

    assert_eq!(line_colors(shown.lines.commands()), vec![MEASUREMENT_COLOR]);
    assert_eq!(shown.labels.len(), 1);
    assert_eq!(shown.labels[0].text, "2.0 mm");
    assert!(line_colors(hidden.lines.commands()).is_empty());
    assert!(hidden.labels.is_empty());
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

    let overlay =
        measurement_overlay(view, viewport, &[placed], &ToolState::Idle).expect("overlay");

    let mapping = viewport.mapping();
    let image_point = |(x, y): (i32, i32)| {
        mapping
            .map(crate::presentation::ViewportPoint::new(
                f64::from(x),
                f64::from(y),
            ))
            .expect("the endpoint lies inside the image")
    };
    match overlay.lines.commands() {
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

#[test]
fn native_length_label_sits_beside_the_midpoint_of_its_line() {
    let (session, _root) = session();
    let view = &session.views[0];
    let viewport = &session.viewports[0];
    let placed = length_on(AnnotationPlane {
        axis: view.axis,
        slice: view.slice_index,
    });

    let overlay =
        measurement_overlay(view, viewport, &[placed], &ToolState::Idle).expect("overlay");

    // The line runs along one image row, drawn left to right, so its
    // counter-clockwise perpendicular points down the screen. Each endpoint
    // and the label round to whole pixels once, so each axis is within one.
    let distance = 12_i32;
    assert_eq!(f64::from(distance), LENGTH_LABEL_DISTANCE);
    match (overlay.lines.commands(), overlay.labels.as_slice()) {
        ([DisplayCommand::DrawLine { start, end, .. }], [label]) => {
            assert_eq!(start.1, end.1, "the line is horizontal on screen");
            assert!((label.x - (start.0 + end.0) / 2).abs() <= 1, "{label:?}");
            assert!((label.y - (start.1 + distance)).abs() <= 1, "{label:?}");
            assert_eq!(label.text, "2.0 mm");
        }
        other => panic!("expected one line and one label, found {other:?}"),
    }
}

#[test]
fn native_annotation_in_progress_roi_is_previewed_on_the_active_pane() {
    let (mut session, _root) = session();
    session.app.active_tool = ToolKind::RoiRect;
    let (x, y) = session.viewports[0].center();
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerMove {
                x: x + 30,
                y: y + 20,
            },
        ])
        .expect("begin ROI drag");

    assert!(matches!(session.app.tool_state, ToolState::RoiDrag { .. }));
    let overlay = measurement_overlay(
        &session.views[0],
        &session.viewports[0],
        &[],
        &session.app.tool_state,
    )
    .expect("overlay");
    assert_eq!(line_colors(overlay.lines.commands()), vec![ROI_COLOR; 4]);
}

#[test]
fn native_annotation_labels_state_their_values() {
    let angle = Annotation::Angle {
        p1: [0.0, 1.0],
        p2: [0.0, 0.0],
        p3: [1.0, 0.0],
        angle_deg: 90.0,
    };
    // The length a 3-by-4 pixel drag records at unit spacing
    // (`a_drag_release_contributes_its_final_position`).
    let length = Annotation::Length {
        p1: [1.0, 1.0],
        p2: [5.0, 4.0],
        length_mm: 5.0,
    };
    let point = Annotation::HuPoint {
        pos: [0.0, 0.0],
        value: -1000.0,
    };

    assert_eq!(measurement_label(&length).as_deref(), Some("5.0 mm"));
    assert_eq!(measurement_label(&angle).as_deref(), Some("90.0 deg"));
    assert_eq!(measurement_label(&point).as_deref(), Some("-1000"));
}
