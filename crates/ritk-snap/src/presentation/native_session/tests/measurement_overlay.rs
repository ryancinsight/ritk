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
use crate::tools::interaction::{Annotation, PatientLength};
use metis_ui_lang::DisplayCommand;

const IDENTITY_DIRECTION: [f64; 9] = [
    1.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, //
    0.0, 0.0, 1.0,
];

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
    session
        .app
        .annotations
        .push(Annotation::PatientLength { plane: 0, length });
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
