//! Completed length projection for the native overlay.
//!
//! Each length draws on the pane matching its recorded plane. Endpoints pass
//! through the same plane projection the linked cursor uses: the view
//! transform maps source coordinates to output pixels and the viewport image
//! rectangle maps those to screen pixels. The value label sits at the segment
//! midpoint, offset perpendicular to the segment exactly like the eframe
//! measurement layer, so both shells read the same placement.

use anyhow::{bail, Result};
use metis_platform::Color;

use super::super::frame::RenderedView;
use super::crosshair::screen_coordinate;
use super::geometry::NativeViewport;
use super::text::{text_command, text_style, PANEL_DETAIL_SIZE};
use super::OverlayCommands;
use crate::geometry::affine::AffineTransform;
use crate::geometry::PatientPointMm;
use crate::tools::interaction::Annotation;

pub(crate) const MEASUREMENT_COLOR: Color = Color::rgba(255, 255, 0, 255);

/// Label offset perpendicular to the measured segment, in screen pixels.
///
/// Matches the eframe measurement layer so the native value label clears the
/// segment by the same distance on both shells.
const LABEL_OFFSET_PX: f64 = 10.0;

/// Append every completed length on its recorded plane to the overlay.
///
/// Orthogonal lengths already store image `[row, col]` endpoints; patient
/// lengths map through `affine` to the viewed plane first. Annotations for
/// other planes, degenerate views, and unmappable endpoints are skipped so one
/// misplaced measurement cannot fail the whole overlay.
pub(crate) fn append_measurement_overlay(
    overlay: &mut OverlayCommands,
    views: &[RenderedView],
    viewports: &[NativeViewport],
    annotations: &[Annotation],
    affine: Option<AffineTransform>,
) -> Result<()> {
    for (view, viewport) in views.iter().zip(viewports.iter()) {
        let output_size = view.transform.output_size(view.source_size);
        if output_size.contains(&0) {
            continue;
        }
        let output_width = f64::from(u32::try_from(output_size[0])?);
        let output_height = f64::from(u32::try_from(output_size[1])?);
        for annotation in annotations {
            let Some((p1, p2, length_mm)) = annotation_endpoints(annotation, view.axis, affine)
            else {
                continue;
            };
            let (start, end) = (
                project(view, viewport, p1, output_width, output_height)?,
                project(view, viewport, p2, output_width, output_height)?,
            );
            overlay.push(metis_ui_lang::DisplayCommand::DrawLine {
                start,
                end,
                color: MEASUREMENT_COLOR,
            });
            let label = format!("{length_mm:.1} mm");
            let (label_x, label_y) = label_position(start, end);
            let style = text_style(MEASUREMENT_COLOR, PANEL_DETAIL_SIZE)?;
            overlay.push(text_command(
                label,
                screen_coordinate(label_x, "measurement label x")?,
                screen_coordinate(label_y, "measurement label y")?,
                style,
            ));
        }
    }
    Ok(())
}

/// Image `[col, row]` endpoints and the value for one annotation on `axis`.
///
/// Returns `None` for annotations placed on another plane, non-length
/// annotations, and patient lengths without a volume affine.
fn annotation_endpoints(
    annotation: &Annotation,
    axis: usize,
    affine: Option<AffineTransform>,
) -> Option<([f64; 2], [f64; 2], f64)> {
    match annotation {
        Annotation::Length {
            plane,
            p1,
            p2,
            length_mm,
        } => {
            if *plane != axis {
                return None;
            }
            Some((
                [f64::from(p1[1]), f64::from(p1[0])],
                [f64::from(p2[1]), f64::from(p2[0])],
                f64::from(*length_mm),
            ))
        }
        Annotation::PatientLength { plane, length } => {
            if *plane != axis {
                return None;
            }
            let affine = affine?;
            let p1 = plane_row_col(axis, affine, length.start_mm())?;
            let p2 = plane_row_col(axis, affine, length.end_mm())?;
            Some((p1, p2, length.length_mm()))
        }
        _ => None,
    }
}

/// Continuous `[col, row]` image coordinates of a patient point on `axis`.
///
/// The voxel keeps its fractional position: rounding here would move
/// sub-voxel endpoints the way the integer cursor mapping must not.
fn plane_row_col(axis: usize, affine: AffineTransform, point: PatientPointMm) -> Option<[f64; 2]> {
    let voxel = affine.patient_to_voxel(point.coordinates());
    if voxel.iter().any(|value| !value.is_finite()) {
        return None;
    }
    // Mirror the component selection of `map_voxel_to_view_row_col` without
    // the integer rounding that cursor voxels require.
    match axis {
        0 => Some([voxel[2], voxel[1]]),
        1 => Some([voxel[2], voxel[0]]),
        2 => Some([voxel[1], voxel[0]]),
        _ => None,
    }
}

/// Screen pixels of one image `[col, row]` point through the view projection.
fn project(
    view: &RenderedView,
    viewport: &NativeViewport,
    point: [f64; 2],
    output_width: f64,
    output_height: f64,
) -> Result<(i32, i32)> {
    if view.source_size.contains(&0) {
        bail!("measurement source has a degenerate extent");
    }
    let output = view
        .transform
        .source_to_output_coordinates(point, view.source_size);
    let point_x = viewport.image.x + output[0] / output_width * viewport.image.width;
    let point_y = viewport.image.y + output[1] / output_height * viewport.image.height;
    Ok((
        screen_coordinate(point_x, "measurement point x")?,
        screen_coordinate(point_y, "measurement point y")?,
    ))
}

/// Label anchor at the segment midpoint, offset perpendicular to the segment.
///
/// The perpendicular runs 90° counter-clockwise from the segment direction,
/// matching the eframe measurement layer; a degenerate segment labels straight
/// above its point. The caller converts through `screen_coordinate` so label
/// placement shares the overlay's host-range contract.
fn label_position(start: (i32, i32), end: (i32, i32)) -> (f64, f64) {
    // Each endpoint converts exactly (|i32| < 2^53) and their difference stays
    // below 2^32, so the f64 subtraction is exact with no host rounding.
    let (dx, dy) = (
        f64::from(end.0) - f64::from(start.0),
        f64::from(end.1) - f64::from(start.1),
    );
    let length = dx.hypot(dy);
    let (offset_x, offset_y) = if length < 1e-4 {
        (0.0, -LABEL_OFFSET_PX)
    } else {
        (
            -dy / length * LABEL_OFFSET_PX,
            dx / length * LABEL_OFFSET_PX,
        )
    };
    (
        (f64::from(start.0) + f64::from(end.0)) * 0.5 + offset_x,
        (f64::from(start.1) + f64::from(end.1)) * 0.5 + offset_y,
    )
}
