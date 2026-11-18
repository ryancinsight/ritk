//! Native measurement overlay: plane-aware length projection.
//!
//! Completed length annotations draw on the pane matching their recorded
//! plane, projecting endpoints through the same plane mapping the crosshair
//! uses, with the value label beside the segment.

use super::super::layout::{
    append_measurement_overlay, append_measurement_preview, OverlayCommands,
};
use super::support::session;
use super::*;
use crate::geometry::affine::AffineTransform;
use crate::geometry::PatientPointMm;
use crate::tools::interaction::{
    Annotation, AnnotationLocation, ImagePoint, PatientLength, ToolState,
};
use crate::ui::{RotationSteps, ViewTransform};
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
        session.app.pointer.selected_polygon,
        None,
        affine,
        |location| session.app.length_scale(location),
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
        session.app.pointer.selected_polygon,
        None,
        None,
        |_| 1.0,
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
fn native_length_preview_draws_only_on_its_owning_slice() {
    let (mut session, _root) = session();
    session.app.view_transform = ViewTransform {
        flip_h: true,
        flip_v: false,
        rotation: RotationSteps::Ninety,
    };
    session
        .refresh_frame()
        .expect("compose transformed preview view");
    let index = session
        .views
        .iter()
        .position(|view| view.axis == 0)
        .expect("session composes an axial pane");
    let view = &session.views[index];
    let viewport = &session.viewports[index];
    let tool_state = ToolState::LengthDrag {
        start: ImagePoint::new(0.5, 0.25),
        current: ImagePoint::new(3.25, 1.5),
    };
    let location = Some(AnnotationLocation {
        plane: view.axis,
        slice: view.slice_index,
    });
    let mut overlay = OverlayCommands::with_capacity(2).expect("reserve preview commands");
    append_measurement_preview(&mut overlay, view, viewport, &tool_state, location)
        .expect("compose native length preview");
    let commands = overlay.into_display_list().commands().to_vec();
    let lines = draw_lines(&commands);
    assert_eq!(lines.len(), 1, "preview draws one segment");
    let output_size = view.transform.output_size(view.source_size);
    let output_width = f64::from(u32::try_from(output_size[0]).expect("output width fits"));
    let output_height = f64::from(u32::try_from(output_size[1]).expect("output height fits"));
    let expected = [[1.75, 3.5], [0.5, 0.75]].map(|[x, y]| {
        (
            super::layout::screen_coordinate(
                viewport.image.x + x / output_width * viewport.image.width,
                "preview endpoint x",
            )
            .expect("preview x fits"),
            super::layout::screen_coordinate(
                viewport.image.y + y / output_height * viewport.image.height,
                "preview endpoint y",
            )
            .expect("preview y fits"),
        )
    });
    assert_eq!(
        lines[0],
        (expected[0], expected[1]),
        "transformed endpoints"
    );

    let mut overlay = OverlayCommands::with_capacity(2).expect("reserve preview commands");
    append_measurement_preview(
        &mut overlay,
        view,
        viewport,
        &tool_state,
        Some(AnnotationLocation {
            plane: view.axis,
            slice: view.slice_index + 1,
        }),
    )
    .expect("compose mismatched native preview");
    let commands = overlay.into_display_list().commands().to_vec();
    assert!(
        draw_lines(&commands).is_empty(),
        "preview skips another slice"
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
            WindowEvent::PointerUp {
                x: start_x,
                y: start_y,
                button: metis_platform::native::MouseButton::Left,
            },
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
fn selected_polygon_draws_one_handle_at_each_projected_vertex() {
    let (mut session, _root) = session();
    let (index, origin_x, origin_y, texel_x, texel_y) = axial_panel(&session);
    let view = &session.views[index];
    session.app.annotations.push(Annotation::OpenPolygon {
        location: Some(AnnotationLocation {
            plane: view.axis,
            slice: view.slice_index,
        }),
        points: vec![[0.0, 0.0], [1.0, 2.0]],
        length_mm: 3.0,
    });
    let unselected = draw_lines(&composed(&session, None));
    assert_eq!(unselected.len(), 1);
    session.app.pointer.selected_polygon = Some(0);
    let selected = draw_lines(&composed(&session, None));
    let first = (origin_x.round(), origin_y.round());
    let second = (
        (origin_x + 2.0 * texel_x).round(),
        (origin_y + texel_y).round(),
    );
    assert_eq!(selected.len(), 5);
    for (x, y) in [first, second] {
        assert!(selected.iter().any(|&(start, end)| {
            (f64::from(start.0), f64::from(start.1)) == (x - 2.0, y)
                && (f64::from(end.0), f64::from(end.1)) == (x + 2.0, y)
        }));
        assert!(selected.iter().any(|&(start, end)| {
            (f64::from(start.0), f64::from(start.1)) == (x, y - 2.0)
                && (f64::from(end.0), f64::from(end.1)) == (x, y + 2.0)
        }));
    }
}

#[test]
fn native_length_calibration_scales_labels_and_resets() {
    use super::super::window_controls::Menu;
    use super::support::control_center;
    use metis_platform::native::MouseButton;

    let (mut session, _root) = session();
    session.app.axis = 0;
    let (_, slice_count) = session.app.axis_slice_info(0);
    let (pane, ..) = axial_panel(&session);
    let slice = session.views[pane].slice_index;
    let other_slice = (slice + 1) % slice_count;
    session.app.loaded.as_mut().expect("fixture volume").spacing = [1.0, 4.0, 3.0];
    let measure = |app: &mut SnapApp, end| {
        app.on_drag_start(
            crate::tools::kind::ToolKind::MeasureLength,
            Some(ImagePoint::new(0.0, 0.0)),
        );
        app.on_drag_end(Some(end));
    };
    for (index, end) in [
        (slice, ImagePoint::new(1.0, 0.0)),
        (other_slice, ImagePoint::new(1.0, 1.0)),
        (slice, ImagePoint::new(1.0, 1.0)),
    ] {
        session.app.set_slice_for_axis(0, index);
        measure(&mut session.app, end);
    }
    let raw = session.app.annotations.clone();
    let click_row = |session: &NativeViewerSession| {
        let (x, y) = control_center(session, None, WindowAction::OpenMenu(Menu::Tools));
        let (rx, ry) = control_center(session, Some(Menu::Tools), WindowAction::LengthCalibration);
        let button = MouseButton::Left;
        [
            WindowEvent::PointerDown { x, y, button },
            WindowEvent::PointerUp { x, y, button },
            WindowEvent::PointerDown {
                x: rx,
                y: ry,
                button,
            },
            WindowEvent::PointerUp {
                x: rx,
                y: ry,
                button,
            },
        ]
    };
    let key = |virtual_key| WindowEvent::KeyDown {
        virtual_key,
        repeated: false,
        modifiers: ModifierState::NONE,
    };
    assert_eq!(draw_texts(&composed(&session, None)), ["3.0 mm", "5.0 mm"]);
    let mut calibrate = click_row(&session).to_vec();
    calibrate.extend([
        WindowEvent::TextInput { character: '0' },
        key(0x0d),
        key(0x08),
        WindowEvent::TextInput { character: '1' },
        WindowEvent::TextInput { character: '0' },
        key(0x0d),
    ]);
    session
        .handle_events(&calibrate)
        .expect("calibration dialog owns input in the menu event batch");
    assert_eq!(draw_texts(&composed(&session, None)), ["6.0 mm", "10.0 mm"]);
    assert_eq!(session.app.annotations, raw);

    let reference = session.app.length_calibration_reference();
    session.app.set_slice_for_axis(0, other_slice);
    assert!(!session.app.set_length_calibration(reference, Some(20.0)));
    session.refresh_frame().expect("compose other image");
    assert_eq!(draw_texts(&composed(&session, None)), ["5.0 mm"]);
    session.app.set_slice_for_axis(0, slice);
    session.refresh_frame().expect("compose calibrated image");
    session
        .handle_events(&click_row(&session))
        .expect("reset calibration");
    assert_eq!(draw_texts(&composed(&session, None)), ["3.0 mm", "5.0 mm"]);
}
