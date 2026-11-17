//! Native measurement overlay: plane-aware length projection.
//!
//! Completed length annotations draw on the pane matching their recorded
//! plane, projecting endpoints through the same plane mapping the crosshair
//! uses, with the value label beside the segment.

use super::super::layout::{append_measurement_overlay, OverlayCommands};
use super::support::session;
use super::*;
use crate::geometry::affine::AffineTransform;
use crate::geometry::PatientPointMm;
use crate::tools::interaction::{Annotation, AnnotationLocation, PatientLength};
use metis_platform::Color;
use metis_ui_lang::DisplayCommand;

const IDENTITY_DIRECTION: [f64; 9] = [
    1.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, //
    0.0, 0.0, 1.0,
];

#[test]
fn grid_measurement_draws_on_its_slice_and_disappears_on_the_adjacent_slice() {
    let (mut session, _root) = session();
    let grid = super::layout::PanelGrid::new(2, 1).expect("two-panel layout");
    session
        .set_workspace_layout(super::layout::WorkspaceLayout::Panels(grid))
        .expect("place the loaded series in the grid");
    session.app.annotations.push(Annotation::Angle {
        location: Some(AnnotationLocation { plane: 0, slice: 1 }),
        p1: [0.25, 0.25],
        p2: [0.25, 1.25],
        p3: [1.25, 1.25],
        angle_deg: 90.0,
    });
    session.refresh_frame().expect("compose the measured slice");
    let (_, x, y, dx, dy) = axial_panel(&session);
    let x = super::layout::screen_coordinate(x + 0.25 * dx, "sample x").expect("sample fits");
    let y = super::layout::screen_coordinate(y + 0.25 * dy, "sample y").expect("sample fits");
    let measurement_color = Color::rgba(255, 255, 0, 255);
    assert_eq!(session.framebuffer.get_pixel(x, y), measurement_color);

    session.app.set_slice_for_axis(0, 0);
    session.refresh_frame().expect("compose the adjacent slice");
    assert_ne!(session.framebuffer.get_pixel(x, y), measurement_color);
}

fn draw_lines(commands: &[DisplayCommand]) -> Vec<((i32, i32), (i32, i32))> {
    commands
        .iter()
        .filter_map(|command| match command {
            DisplayCommand::DrawLine { start, end, .. } => Some((*start, *end)),
            _ => None,
        })
        .collect()
}

fn draw_texts(commands: &[DisplayCommand]) -> Vec<String> {
    commands
        .iter()
        .filter_map(|command| match command {
            DisplayCommand::DrawText { text, .. } => Some(text.to_string()),
            _ => None,
        })
        .collect()
}

fn axial_panel(session: &NativeViewerSession) -> (usize, f64, f64, f64, f64) {
    let index = session
        .views
        .iter()
        .position(|view| view.axis == 0)
        .expect("session composes an axial pane");
    let viewport = &session.viewports[index];
    let width = f64::from(session.views[index].frame.width());
    let height = f64::from(session.views[index].frame.height());
    (
        index,
        viewport.image.x,
        viewport.image.y,
        viewport.image.width / width,
        viewport.image.height / height,
    )
}

fn composed(session: &NativeViewerSession, affine: Option<AffineTransform>) -> Vec<DisplayCommand> {
    let mut overlay = OverlayCommands::with_capacity(session.app.annotations.len() * 2 + 8)
        .expect("reserve the measurement overlay commands");
    append_measurement_overlay(
        &mut overlay,
        &session.views,
        &session.viewports,
        &session.app.annotations,
        affine,
    )
    .expect("compose the measurement overlay");
    overlay.into_display_list().commands().to_vec()
}

#[test]
fn measurement_length_draws_projected_line_and_value_on_matching_plane() {
    let (mut session, _root) = session();
    session.app.annotations.push(Annotation::Length {
        plane: 0,
        location: None,
        p1: [0.0, 0.0],
        p2: [3.0, 4.0],
        length_mm: 5.0,
    });
    let (index, origin_x, origin_y, texel_x, texel_y) = axial_panel(&session);
    let commands = composed(&session, None);
    let lines = draw_lines(&commands);
    assert_eq!(lines.len(), 1, "one segment per matching length: {lines:?}");
    let ((x1, y1), (x2, y2)) = lines[0];
    assert_eq!(
        (f64::from(x1), f64::from(y1)),
        (origin_x.round(), origin_y.round()),
        "segment starts where the image origin projects"
    );
    assert_eq!(
        (f64::from(x2), f64::from(y2)),
        (
            (origin_x + 4.0 * texel_x).round(),
            (origin_y + 3.0 * texel_y).round()
        ),
        "segment ends where the [row, col] endpoint projects"
    );
    let texts = draw_texts(&commands);
    assert!(
        texts.iter().any(|text| text == "5.0 mm"),
        "length value labels the segment: {texts:?}"
    );
    let _ = index;
}

#[test]
fn measurement_length_skips_panes_beside_its_recorded_plane() {
    let (mut session, _root) = session();
    session.app.annotations.push(Annotation::Length {
        plane: 1,
        location: None,
        p1: [0.0, 0.0],
        p2: [3.0, 4.0],
        length_mm: 5.0,
    });
    let (index, ..) = axial_panel(&session);
    let views = [session.views[index].clone()];
    let viewports = [session.viewports[index]];
    let mut overlay = OverlayCommands::with_capacity(8).expect("reserve overlay commands");
    append_measurement_overlay(
        &mut overlay,
        &views,
        &viewports,
        &session.app.annotations,
        None,
    )
    .expect("compose the measurement overlay");
    let commands = overlay.into_display_list().commands().to_vec();
    assert!(
        draw_lines(&commands).is_empty(),
        "a coronal length must not draw on the axial pane"
    );
    assert!(
        draw_texts(&commands).is_empty(),
        "a coronal length must not label the axial pane"
    );
}

#[test]
fn measurement_patient_length_projects_through_affine_to_same_segment() {
    let (mut session, _root) = session();
    let start = PatientPointMm::try_new([0.0, 0.0, 0.0]).expect("finite patient point");
    let end = PatientPointMm::try_new([0.0, 3.0, 4.0]).expect("finite patient point");
    let length = PatientLength::try_new(start, end).expect("3-4-5 segment is representable");
    assert_eq!(length.length_mm(), 5.0, "acceptance segment measures 5 mm");
    session.app.annotations.push(Annotation::PatientLength {
        plane: 0,
        location: None,
        length,
    });
    let affine = AffineTransform::from_parts([0.0, 0.0, 0.0], IDENTITY_DIRECTION, [1.0, 1.0, 1.0])
        .expect("unit affine is valid");
    let (index, origin_x, origin_y, texel_x, texel_y) = axial_panel(&session);
    let commands = composed(&session, Some(affine));
    let lines = draw_lines(&commands);
    assert_eq!(lines.len(), 1, "one segment per matching length: {lines:?}");
    let ((x1, y1), (x2, y2)) = lines[0];
    assert_eq!(
        (f64::from(x1), f64::from(y1)),
        (origin_x.round(), origin_y.round()),
        "patient segment starts where the voxel origin projects"
    );
    assert_eq!(
        (f64::from(x2), f64::from(y2)),
        (
            (origin_x + 4.0 * texel_x).round(),
            (origin_y + 3.0 * texel_y).round()
        ),
        "patient segment ends where the mapped voxel projects"
    );
    let texts = draw_texts(&commands);
    assert!(
        texts.iter().any(|text| text == "5.0 mm"),
        "patient length value labels the segment: {texts:?}"
    );
    let _ = index;
}

#[test]
fn native_toolbar_length_then_drag_records_measurement() {
    let (mut session, _root) = session();
    let (tool_x, tool_y) = super::support::control_center(
        &session,
        None,
        WindowAction::SelectTool(crate::tools::kind::ToolKind::MeasureLength),
    );
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: tool_x,
                y: tool_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x: tool_x,
                y: tool_y,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("toolbar length selection");
    assert_eq!(
        session.app.active_tool,
        crate::tools::kind::ToolKind::MeasureLength
    );

    let (pane, _, _, width, height) = axial_panel(&session);
    let (start_x, start_y) = session.viewports[pane].center();
    let [row_spacing, col_spacing] = session.views[pane].frame.display_spacing().values();
    let expected_mm = (3.0 * col_spacing / width).hypot(4.0 * row_spacing / height);
    let bound = f64::from(f32::EPSILON)
        * (f64::from(session.views[pane].frame.width()) * col_spacing
            + f64::from(session.views[pane].frame.height()) * row_spacing);
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: start_x,
                y: start_y,
                button: metis_platform::native::MouseButton::Left,
            },
            WindowEvent::PointerMove {
                x: start_x + 3,
                y: start_y + 4,
            },
            WindowEvent::PointerUp {
                x: start_x + 3,
                y: start_y + 4,
                button: metis_platform::native::MouseButton::Left,
            },
        ])
        .expect("native length drag");
    assert!(matches!(
        session.app.annotations.as_slice(),
        [crate::tools::interaction::Annotation::Length { length_mm, .. }]
            if (f64::from(*length_mm) - expected_mm).abs() <= bound
    ));
}

#[test]
fn native_delete_targets_the_mark_and_image_under_the_pointer() {
    use metis_platform::native::MouseButton;

    let (mut session, _root) = session();
    let (pane, x, y, dx, dy) = axial_panel(&session);
    let slice = session.views[pane].slice_index;
    let other_slice = (slice + 1) % session.views[pane].slice_count;
    assert_ne!(slice, other_slice);
    for (position, recorded) in [
        ([0.5, 0.5], slice),
        ([1.5, 1.5], slice),
        ([0.5, 0.5], other_slice),
    ] {
        session.app.annotations.push(Annotation::HuPoint {
            location: Some(AnnotationLocation {
                plane: 0,
                slice: recorded,
            }),
            pos: position,
            value: 42.0,
        });
    }
    let x = super::layout::screen_coordinate(x + dx * 0.5, "point x").expect("bounded test x");
    let y = super::layout::screen_coordinate(y + dy * 0.5, "point y").expect("bounded test y");
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::KeyDown {
                virtual_key: 0x2e,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("select and delete one mark");
    assert_eq!(session.app.annotations.len(), 2);
    assert!(
        matches!(session.app.annotations.first(), Some(Annotation::HuPoint { pos, .. }) if *pos == [1.5, 1.5])
    );
    session
        .handle_events(&[WindowEvent::KeyDown {
            virtual_key: 0x2e,
            repeated: false,
            modifiers: ModifierState::CONTROL,
        }])
        .expect("delete only marks on the displayed image");
    assert_eq!(
        session.app.annotations.as_slice(),
        &[Annotation::HuPoint {
            location: Some(AnnotationLocation {
                plane: 0,
                slice: other_slice
            }),
            pos: [0.5, 0.5],
            value: 42.0,
        }]
    );
}

#[test]
fn native_calibration_uses_the_selected_reference_on_one_image() {
    use metis_platform::native::MouseButton;

    let (mut session, _root) = session();
    let (pane, x, y, dx, dy) = axial_panel(&session);
    let slice = session.views[pane].slice_index;
    let p1 = [0.5, 0.5];
    let p2 = [0.5, 1.5];
    let raw = session
        .app
        .measured_length(p1, p2)
        .expect("reference has physical spacing");
    session.app.annotations.push(Annotation::Length {
        plane: 0,
        location: Some(AnnotationLocation { plane: 0, slice }),
        p1,
        p2,
        length_mm: raw,
    });
    session.app.annotations.push(Annotation::Length {
        plane: 0,
        location: Some(AnnotationLocation { plane: 0, slice }),
        p1: [1.5, 0.5],
        p2: [1.5, 2.5],
        length_mm: raw * 2.0,
    });
    let other_slice = (slice + 1) % session.views[pane].slice_count;
    session.app.annotations.push(Annotation::Length {
        plane: 0,
        location: Some(AnnotationLocation {
            plane: 0,
            slice: other_slice,
        }),
        p1,
        p2,
        length_mm: raw,
    });
    let x = super::layout::screen_coordinate(x + dx, "reference x").expect("bounded test x");
    let y = super::layout::screen_coordinate(y + dy * 0.5, "reference y").expect("bounded test y");
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x,
                y,
                button: MouseButton::Left,
            },
        ])
        .expect("select the reference length");
    session
        .apply_window_action(WindowAction::OpenCalibration)
        .expect("open calibration");
    session
        .handle_events(&[
            WindowEvent::TextInput { character: '8' },
            WindowEvent::KeyDown {
                virtual_key: 0x0d,
                repeated: false,
                modifiers: ModifierState::NONE,
            },
        ])
        .expect("enter the known calibration distance");
    assert!(matches!(session.app.annotations.as_slice(), [
            Annotation::Length { length_mm: reference, .. },
            Annotation::Length { length_mm: existing, .. }
        ] if *reference == 8.0 && (*existing - 16.0).abs() <= f32::EPSILON * 16.0));
    let corrected = session
        .app
        .measured_length(p1, p2)
        .expect("calibrated length");
    assert!((corrected - 8.0).abs() <= f32::EPSILON * 8.0);
    session.app.set_slice_for_axis(0, other_slice);
    assert_eq!(
        session
            .app
            .measured_length(p1, p2)
            .expect("adjacent image retains spacing"),
        raw
    );
    session.app.set_slice_for_axis(0, slice);
    session.app.pointer.selected_annotation = None;
    let reset = WindowAction::ResetCalibration;
    super::support::control_center(
        &session,
        Some(super::super::window_controls::Menu::Tools),
        reset,
    );
    session
        .apply_window_action(reset)
        .expect("reset calibration");
    assert!(matches!(session.app.annotations.as_slice(), [
        Annotation::Length { length_mm: reference, .. },
        Annotation::Length { length_mm: existing, .. },
        Annotation::Length { length_mm: other, .. }
    ] if (*reference - raw).abs() <= f32::EPSILON * raw
        && (*existing - raw * 2.0).abs() <= f32::EPSILON * raw * 2.0
        && (*other - raw).abs() <= f32::EPSILON * raw));
}
