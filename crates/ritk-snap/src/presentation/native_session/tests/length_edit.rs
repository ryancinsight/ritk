use super::measurement_overlay::{axial_panel, draw_lines};
use super::support::{control_center, session};
use super::*;
use crate::presentation::native_session::layout::{append_measurement_overlay, OverlayCommands};
use crate::presentation::native_session::window_controls::WindowAction;
use crate::tools::interaction::Annotation;
use crate::tools::kind::ToolKind;
use eunomia::FloatElement;
use metis_platform::native::MouseButton;

const COORDINATE_ROUNDINGS: f64 = 8.0;
const LENGTH_ROUNDINGS: f64 = 16.0;

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

fn assert_length(annotation: &Annotation, p1: [f64; 2], p2: [f64; 2], length: f64) {
    let Annotation::Length {
        p1: actual_p1,
        p2: actual_p2,
        length_mm,
        ..
    } = annotation
    else {
        panic!("expected a length annotation");
    };
    for (actual, expected) in actual_p1
        .iter()
        .chain(actual_p2)
        .zip(p1.into_iter().chain(p2))
    {
        let bound = COORDINATE_ROUNDINGS * f64::from(f32::EPSILON) * expected.abs().max(1.0);
        assert!((f64::from(*actual) - expected).abs() <= bound);
    }
    let bound = LENGTH_ROUNDINGS * f64::from(f32::EPSILON) * length.abs().max(1.0);
    assert!((f64::from(*length_mm) - length).abs() <= bound);
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

#[test]
fn native_pan_edits_both_length_endpoints_transactionally() {
    let (mut session, _root) = session();
    session.app.axis = 0;
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
    let [row_spacing, column_spacing] = session.app.slice_plane_spacing().expect("spacing");
    // Screen 30-by-40 maps to column 30/sx and row 40/sy before anisotropic
    // physical spacing. Doubling endpoint two gives 2A; moving endpoint one
    // to the opposite displacement gives 3A.
    let a = (30.0 * column_spacing / sx).hypot(40.0 * row_spacing / sy);
    let location = session.app.annotation_location();
    let initial = Annotation::Length {
        plane: 0,
        location: Some(location),
        p1: stored(x, y),
        p2: stored(x + 30, y + 40),
        length_mm: Annotation::compute_length_checked(
            stored(x, y),
            stored(x + 30, y + 40),
            [row_spacing, column_spacing],
        )
        .expect("finite source length"),
    };
    session.app.annotations = vec![initial];
    let reference = session.app.length_calibration_reference();
    let raw = match session.app.annotations[0] {
        Annotation::Length { length_mm, .. } => length_mm,
        _ => unreachable!("invariant: fixture stores a length"),
    };
    assert!(session
        .app
        .set_length_calibration(reference, Some(raw * 2.0)));

    let pan = control_center(&session, None, WindowAction::SelectTool(ToolKind::Pan));
    session
        .handle_events(&[pointer(pan, true), pointer(pan, false)])
        .expect("select Pan through the native toolbar");
    assert_eq!(session.app.active_tool, ToolKind::Pan);
    session
        .handle_events(&drag((x + 30, y + 40), (x + 60, y + 80)))
        .expect("move endpoint two");
    assert_length(
        &session.app.annotations[0],
        physical(x, y),
        physical(x + 60, y + 80),
        2.0 * a,
    );
    session
        .handle_events(&drag((x, y), (x - 30, y - 40)))
        .expect("move endpoint one");
    assert_length(
        &session.app.annotations[0],
        physical(x - 30, y - 40),
        physical(x + 60, y + 80),
        3.0 * a,
    );
    assert_eq!(session.app.length_scale(Some(location)), 2.0);
    assert_eq!(
        overlay_lines(&session).len(),
        5,
        "selected length has two handles"
    );

    let committed = session.app.annotations.clone();
    session
        .handle_events(&[
            pointer((x + 60, y + 80), true),
            WindowEvent::PointerMove {
                x: x + 70,
                y: y + 90,
            },
        ])
        .expect("preview endpoint before Escape");
    assert_eq!(session.app.annotations, committed);
    assert!(overlay_lines(&session).contains(&((x - 30, y - 40), (x + 70, y + 90))));
    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x1b,
            repeated: false,
            modifiers: ModifierState::NONE,
        }])
        .expect("Escape abandons preview");
    assert_eq!(session.app.annotations, committed);
}
