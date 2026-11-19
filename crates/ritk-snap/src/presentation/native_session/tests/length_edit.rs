use super::measurement_overlay::{axial_panel, draw_lines};
use super::support::{control_center, session};
use super::*;
use crate::app::pointer_bindings::AnnotationMoveKind;
use crate::presentation::native_session::layout::{append_measurement_overlay, OverlayCommands};
use crate::presentation::native_session::window_controls::WindowAction;
use crate::tools::interaction::{Annotation, AnnotationLocation};
use crate::tools::kind::ToolKind;
use eunomia::FloatElement;
use metis_platform::native::MouseButton;
fn pointer((x, y): (i32, i32), down: bool) -> WindowEvent {
    if down {
        WindowEvent::PointerDown {
            x,
            y,
            button: MouseButton::Left,
        }
    } else {
        WindowEvent::PointerUp {
            x,
            y,
            button: MouseButton::Left,
        }
    }
}
fn drag(start: (i32, i32), end: (i32, i32)) -> [WindowEvent; 3] {
    [
        pointer(start, true),
        WindowEvent::PointerMove { x: end.0, y: end.1 },
        pointer(end, false),
    ]
}
fn apply(session: &mut NativeViewerSession, events: &[WindowEvent]) {
    session.handle_events(events).expect("length edit event");
}
fn overlay_lines(session: &NativeViewerSession) -> Vec<((i32, i32), (i32, i32))> {
    let mut overlay = OverlayCommands::with_capacity(16).expect("reserve edit overlay");
    append_measurement_overlay(
        &mut overlay,
        &session.views,
        &session.viewports,
        &session.app.annotations,
        session.app.pointer.selected_annotation,
        session
            .app
            .pointer
            .pending_annotation_move
            .as_ref()
            .map(|pending| (pending.annotation_index, &pending.candidate)),
        None,
        |location| session.app.length_scale(location),
    )
    .expect("compose edit overlay");
    draw_lines(overlay.into_display_list().commands())
}
#[derive(Clone, Copy)]
enum Cancellation {
    Escape,
    CaptureLoss,
    SliceReturn,
}
#[test]
fn native_pan_edits_both_length_endpoints_transactionally() {
    let (mut session, _root) = session();
    session.app.axis = 0;
    let grid = super::layout::PanelGrid::new(1, 1).expect("single-panel layout");
    session
        .set_workspace_layout(super::layout::WorkspaceLayout::Panels(grid))
        .expect("show the measured plane");
    session.app.loaded.as_mut().expect("fixture volume").spacing = [1.0, 2.0, 3.0];
    session
        .refresh_frame()
        .expect("refresh anisotropic fixture");
    let (pane, origin_x, origin_y, sx, sy) = axial_panel(&session);
    let (x, y) = session.viewports[pane].center();
    let physical = |x, y| {
        [
            (f64::from(y) - origin_y) / sy,
            (f64::from(x) - origin_x) / sx,
        ]
    };
    let stored = |x, y| physical(x, y).map(<f32 as FloatElement>::from_f64);
    let spacing = session.app.slice_plane_spacing().expect("spacing");
    let [row_spacing, column_spacing] = spacing;
    // A combines screen displacement, viewport scale, and anisotropic spacing;
    // endpoint two doubles A, then moving endpoint one opposite produces 3A.
    let a = (30.0 * column_spacing / sx).hypot(40.0 * row_spacing / sy);
    let raw = <f32 as FloatElement>::from_f64(a);
    let location = AnnotationLocation {
        plane: 0,
        slice: session.app.axis_slice_info(0).0,
    };
    let assert_length = |annotation: &Annotation, p1: [f64; 2], p2: [f64; 2], length: f64| {
        let Annotation::Length {
            plane,
            location: actual_location,
            p1: actual_p1,
            p2: actual_p2,
            length_mm,
        } = annotation
        else {
            panic!("expected a length annotation");
        };
        assert_eq!((*plane, *actual_location), (location.plane, Some(location)));
        assert_eq!(*actual_p1, p1.map(<f32 as FloatElement>::from_f64));
        assert_eq!(*actual_p2, p2.map(<f32 as FloatElement>::from_f64));
        // Endpoint narrowing precedes eight f32 operations in the physical norm.
        let unit = f64::from(f32::EPSILON) / 2.0;
        let stored = |value: f64| unit * value.abs().max(1.0);
        let endpoint = ((stored(p1[0]) + stored(p2[0])) * spacing[0])
            .hypot((stored(p1[1]) + stored(p2[1])) * spacing[1]);
        let roundings = 8.0;
        let gamma = roundings * unit / (1.0 - roundings * unit);
        let bound = endpoint + gamma * (length + endpoint);
        assert!((f64::from(*length_mm) - length).abs() <= bound);
    };
    let initial = Annotation::Length {
        plane: 0,
        location: Some(location),
        p1: stored(x, y),
        p2: stored(x + 30, y + 40),
        length_mm: raw,
    };
    let other_source = initial.clone().with_location(AnnotationLocation {
        slice: location.slice + 1,
        ..location
    });
    session.app.annotations = vec![initial, other_source.clone()];
    let reference = session.app.length_calibration_reference();
    assert!(session
        .app
        .set_length_calibration(reference, Some(raw * 2.0)));
    let pan = control_center(&session, None, WindowAction::SelectTool(ToolKind::Pan));
    apply(&mut session, &[pointer(pan, true), pointer(pan, false)]);
    assert_eq!(session.app.active_tool, ToolKind::Pan);
    let before = session.app.annotations.clone();
    apply(&mut session, &drag((x + 30, y + 40), (x + 60, y + 80))[..2]);
    assert_eq!(session.app.annotations, before);
    apply(&mut session, &[pointer((x + 60, y + 80), false)]);
    assert_length(
        &session.app.annotations[0],
        physical(x, y),
        physical(x + 60, y + 80),
        2.0 * a,
    );
    apply(&mut session, &drag((x, y), (x - 30, y - 40)));
    assert_length(
        &session.app.annotations[0],
        physical(x - 30, y - 40),
        physical(x + 60, y + 80),
        3.0 * a,
    );
    assert_eq!(session.app.length_scale(Some(location)), 2.0);
    assert_eq!(session.app.annotations[1], other_source);
    assert_eq!(overlay_lines(&session).len(), 5);
    for cancellation in [
        Cancellation::Escape,
        Cancellation::CaptureLoss,
        Cancellation::SliceReturn,
    ] {
        let committed = session.app.annotations.clone();
        apply(&mut session, &drag((x + 60, y + 80), (x + 70, y + 90))[..2]);
        assert_eq!(session.app.annotations, committed);
        assert!(overlay_lines(&session).contains(&((x - 30, y - 40), (x + 70, y + 90))));
        let event = match cancellation {
            Cancellation::Escape => WindowEvent::KeyDown {
                virtual_key: 0x1b,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
            Cancellation::CaptureLoss => WindowEvent::PointerCaptureLost,
            Cancellation::SliceReturn => {
                let (slice, count) = session.app.axis_slice_info(0);
                session.app.set_slice_for_axis(0, (slice + 1) % count);
                session.app.set_slice_for_axis(0, slice);
                pointer((x + 70, y + 90), false)
            }
        };
        apply(&mut session, &[event]);
        assert_eq!(session.app.annotations, committed);
        assert_eq!(session.app.pointer.pending_annotation_move, None);
    }
    let committed = session.app.annotations[0].clone();
    apply(
        &mut session,
        &[WindowEvent::KeyDown {
            virtual_key: crate::ui::tool_shortcuts::VIRTUAL_KEY_ARROW,
            repeated: false,
            modifiers: ModifierState::NONE,
        }],
    );
    apply(&mut session, &drag((x + 60, y + 80), (x + 70, y + 90)));
    assert_eq!(session.app.annotations[0], committed);
    assert!(matches!(
        session.app.annotations.last(),
        Some(Annotation::Arrow { .. })
    ));
    let Annotation::Length { p1, p2, .. } = committed.clone() else {
        unreachable!("invariant: committed annotation is a length");
    };
    let polygon = Annotation::OpenPolygon {
        location: Some(location),
        points: vec![p1, p2],
        length_mm: <f32 as FloatElement>::from_f64(3.0 * a),
    };
    session.app.annotations = vec![committed.clone(), polygon.clone()];
    session.app.pointer.selected_annotation = None;
    apply(&mut session, &[pointer(pan, true), pointer(pan, false)]);
    let pan_before = session.app.pan_offset;
    let raw = session.app.annotations.clone();
    apply(&mut session, &drag((x - 30, y - 40), (x - 25, y - 40))[..2]);
    assert_eq!(session.app.annotations, raw);
    assert_eq!(session.app.pointer.selected_annotation, Some(1));
    assert!(
        matches!(session.app.pointer.pending_annotation_move.as_ref(), Some(edit) if edit.annotation_index == 1 && matches!(edit.kind, AnnotationMoveKind::Polygon { .. }))
    );
    assert_eq!(session.app.pan_offset, pan_before);
    apply(&mut session, &[pointer((x - 25, y - 40), false)]);
    let Annotation::OpenPolygon { points, .. } = &session.app.annotations[1] else {
        unreachable!("invariant: topmost annotation remains a polygon");
    };
    assert_eq!(points[0], stored(x - 25, y - 40));
    assert_eq!(session.app.pan_offset, pan_before);
    session.app.annotations = vec![polygon, committed.clone()];
    session.app.pointer.selected_annotation = Some(0);
    apply(&mut session, &drag((x - 30, y - 40), (x - 25, y - 40))[..2]);
    assert!(
        matches!(session.app.pointer.pending_annotation_move.as_ref(), Some(edit) if edit.annotation_index == 1 && matches!(edit.kind, AnnotationMoveKind::Length(_)))
    );
    session.app.pointer.selected_annotation = Some(0);
    assert!(overlay_lines(&session)
        .iter()
        .any(|&(start, end)| start == (x - 25, y - 40) || end == (x - 25, y - 40)));
    assert_eq!(session.app.annotations[1], committed);
}
