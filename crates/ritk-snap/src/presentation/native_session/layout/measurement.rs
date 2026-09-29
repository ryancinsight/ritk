//! Patient-space measurement rendering for the native oblique panel.

use anyhow::{Result, anyhow};
use metis_platform::Color;
use metis_ui_lang::{DisplayCommand, DisplayList};

use super::geometry::screen_coordinate;
use super::text::{MEASUREMENT_LABEL_SIZE, text_style};
use crate::app::ObliqueViewport;
use crate::geometry::PatientPointMm;
use crate::render::{ResliceError, ReslicePlane};
use crate::tools::interaction::{Annotation, PatientLength, ToolState};

const MEASUREMENT_COLOR: Color = Color::rgba(255, 235, 59, 255);
const LABEL_MARGIN: f64 = 2.0;
/// Project patient-space lengths only when their endpoints remain on this plane.
pub(crate) fn patient_measurement_overlay(
    annotations: &[Annotation],
    tool_state: &ToolState,
    viewport: ObliqueViewport,
    plane: &ReslicePlane,
) -> Result<DisplayList> {
    let mut overlay = DisplayList::default();
    let Some(clip_bounds) = viewport.visible_pixel_bounds() else {
        return Ok(overlay);
    };
    for annotation in annotations {
        if let Annotation::PatientLength(length) = annotation {
            append_length(&mut overlay, *length, viewport, plane, clip_bounds)?;
        }
    }
    if let ToolState::PatientLength1 { p1 } = tool_state {
        if let Some(point) = projected_point(*p1, viewport, plane)? {
            append_marker(&mut overlay, point, clip_bounds)?;
        }
    }
    Ok(overlay)
}

fn append_length(
    overlay: &mut DisplayList,
    length: PatientLength,
    viewport: ObliqueViewport,
    plane: &ReslicePlane,
    clip_bounds: [f64; 4],
) -> Result<()> {
    let Some(start) = projected_point(length.start_mm(), viewport, plane)? else {
        return Ok(());
    };
    let Some(end) = projected_point(length.end_mm(), viewport, plane)? else {
        return Ok(());
    };
    let Some((line_start, line_end)) = clip_segment(start, end, clip_bounds) else {
        return Ok(());
    };
    overlay.append_line(
        (
            screen_coordinate(line_start[0], "measurement start x")?,
            screen_coordinate(line_start[1], "measurement start y")?,
        ),
        (
            screen_coordinate(line_end[0], "measurement end x")?,
            screen_coordinate(line_end[1], "measurement end y")?,
        ),
        MEASUREMENT_COLOR,
    )?;
    append_marker(overlay, start, clip_bounds)?;
    append_marker(overlay, end, clip_bounds)?;
    append_length_label(
        overlay,
        length.length_mm(),
        line_start,
        line_end,
        clip_bounds,
    )?;
    Ok(())
}

fn projected_point(
    patient: PatientPointMm,
    viewport: ObliqueViewport,
    plane: &ReslicePlane,
) -> Result<Option<[f64; 2]>> {
    let projection = match plane.project_patient(patient.coordinates()) {
        Ok(projection) => projection,
        Err(ResliceError::PixelOutOfBounds { .. }) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let pixel_radius =
        0.5 * vector_norm(plane.horizontal_step()).hypot(vector_norm(plane.vertical_step()));
    if projection.distance_mm().abs() > pixel_radius {
        return Ok(None);
    }
    Ok(viewport.screen_point(projection.pixel()))
}

fn append_marker(overlay: &mut DisplayList, point: [f64; 2], clip_bounds: [f64; 4]) -> Result<()> {
    const MARKER_HALF_SIZE: f64 = 3.0;
    let [left, right, top, bottom] = clip_bounds;
    if point[0] < left || point[0] > right || point[1] < top || point[1] > bottom {
        return Ok(());
    }
    let x = screen_coordinate(point[0], "measurement marker x")?;
    let y = screen_coordinate(point[1], "measurement marker y")?;
    overlay.append_line(
        (
            screen_coordinate(
                (point[0] - MARKER_HALF_SIZE).max(left),
                "measurement marker left",
            )?,
            y,
        ),
        (
            screen_coordinate(
                (point[0] + MARKER_HALF_SIZE).min(right),
                "measurement marker right",
            )?,
            y,
        ),
        MEASUREMENT_COLOR,
    )?;
    overlay.append_line(
        (
            x,
            screen_coordinate(
                (point[1] - MARKER_HALF_SIZE).max(top),
                "measurement marker top",
            )?,
        ),
        (
            x,
            screen_coordinate(
                (point[1] + MARKER_HALF_SIZE).min(bottom),
                "measurement marker bottom",
            )?,
        ),
        MEASUREMENT_COLOR,
    )?;
    Ok(())
}

fn append_length_label(
    overlay: &mut DisplayList,
    length_mm: f64,
    start: [f64; 2],
    end: [f64; 2],
    clip_bounds: [f64; 4],
) -> Result<()> {
    let text = format!("{length_mm:.1} mm");
    let style = text_style(MEASUREMENT_COLOR, MEASUREMENT_LABEL_SIZE)?;
    let [left, right, top, bottom] = clip_bounds;
    let text_width = style.advance(&text);
    let text_height = style.line_height();
    let x = ((start[0] + end[0]) * 0.5 + LABEL_MARGIN).clamp(
        left + LABEL_MARGIN,
        (right - text_width - LABEL_MARGIN).max(left + LABEL_MARGIN),
    );
    let y = ((start[1] + end[1]) * 0.5 - text_height - LABEL_MARGIN).clamp(
        top + LABEL_MARGIN,
        (bottom - text_height - LABEL_MARGIN).max(top + LABEL_MARGIN),
    );
    overlay
        .commands
        .try_reserve(1)
        .map_err(|_| anyhow!("native patient measurement label allocation failed"))?;
    overlay.commands.push(DisplayCommand::DrawText {
        text,
        x: screen_coordinate(x, "measurement label x")?,
        y: screen_coordinate(y, "measurement label y")?,
        style,
    });
    Ok(())
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    vector
        .into_iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

fn clip_segment(
    start: [f64; 2],
    end: [f64; 2],
    [left, right, top, bottom]: [f64; 4],
) -> Option<([f64; 2], [f64; 2])> {
    let delta = [end[0] - start[0], end[1] - start[1]];
    if !delta.into_iter().all(f64::is_finite) {
        return None;
    }
    let mut entering = 0.0_f64;
    let mut leaving = 1.0_f64;
    for (direction, distance) in [
        (-delta[0], start[0] - left),
        (delta[0], right - start[0]),
        (-delta[1], start[1] - top),
        (delta[1], bottom - start[1]),
    ] {
        if direction == 0.0 {
            if distance < 0.0 {
                return None;
            }
            continue;
        }
        let ratio = distance / direction;
        if direction < 0.0 {
            if ratio > leaving {
                return None;
            }
            entering = entering.max(ratio);
        } else {
            if ratio < entering {
                return None;
            }
            leaving = leaving.min(ratio);
        }
    }
    (entering <= leaving).then_some((
        [
            start[0] + entering * delta[0],
            start[1] + entering * delta[1],
        ],
        [start[0] + leaving * delta[0], start[1] + leaving * delta[1]],
    ))
}

#[cfg(test)]
mod tests {
    use super::clip_segment;

    #[test]
    fn patient_length_segment_is_clipped_to_visible_pane_pixels() {
        assert_eq!(
            clip_segment([-20.0, 20.0], [40.0, 20.0], [10.0, 29.0, 5.0, 35.0]),
            Some(([10.0, 20.0], [29.0, 20.0]))
        );
        assert_eq!(
            clip_segment([-20.0, 2.0], [40.0, 2.0], [10.0, 29.0, 5.0, 35.0]),
            None
        );
    }
}
