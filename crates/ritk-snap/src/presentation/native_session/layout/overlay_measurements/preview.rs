//! Measurements in progress on the slice that owns the pointer gesture.

use super::{image_point, NativeViewport, OverlayCommands, Pane, RenderedView, MEASUREMENT_COLOR};
use crate::app::SnapApp;
use crate::tools::interaction::{AnnotationLocation, ImagePoint, PathKind, ToolState};
use anyhow::Result;

pub(in crate::presentation::native_session) fn append_measurement_preview(
    overlay: &mut OverlayCommands,
    view: &RenderedView,
    viewport: &NativeViewport,
    app: &SnapApp,
) -> Result<()> {
    let location = AnnotationLocation {
        plane: view.axis,
        slice: view.slice_index,
    };
    let is_drag = app.pointer.drag_measurement_location == Some(location);
    let is_pending = app.pointer.pending_measurement_location == Some(location);
    if !is_drag && !is_pending {
        return Ok(());
    }
    let Some(pane) = Pane::new(view, viewport)? else {
        return Ok(());
    };
    let cursor = app
        .pointer
        .annotation_cursor
        .filter(|(recorded, _)| *recorded == location)
        .map(|(_, point)| point);

    match &app.tool_state {
        ToolState::LengthDrag { start, current } if is_drag => {
            length(&pane, overlay, app, *start, *current)?
        }
        ToolState::MeasureLength1 { p1 } if is_pending => {
            if let Some(current) = cursor {
                if app.active_tool == crate::tools::kind::ToolKind::MeasureLength {
                    length(&pane, overlay, app, *p1, current)?;
                } else {
                    segment(&pane, overlay, *p1, current)?;
                }
            }
        }
        ToolState::MeasureAngle2 { p1, p2 } if is_pending => {
            segment(&pane, overlay, *p1, *p2)?;
            if let Some(current) = cursor {
                segment(&pane, overlay, *p2, current)?;
            }
        }
        ToolState::Polygon { points, .. } if is_pending => path(&pane, overlay, points, cursor)?,
        ToolState::Cobb { points } if is_pending => path(&pane, overlay, points, None)?,
        ToolState::CobbDrag { points } if is_drag => {
            for pair in points.chunks_exact(2) {
                segment(&pane, overlay, pair[0], pair[1])?;
            }
        }
        ToolState::DeviationDrag { start, current } if is_drag => {
            segment(&pane, overlay, *start, *current)?;
        }
        ToolState::PathDrag {
            points,
            kind: PathKind::Pencil,
            ..
        } if is_drag => path(&pane, overlay, points, None)?,
        ToolState::PathDrag {
            start,
            points,
            kind: PathKind::Arrow,
        } if is_drag => {
            if let Some(end) = points.last() {
                pane.arrow(
                    overlay,
                    image_point([start.y(), start.x()]),
                    image_point([end.y(), end.x()]),
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn path(
    pane: &Pane<'_>,
    overlay: &mut OverlayCommands,
    points: &[ImagePoint],
    cursor: Option<ImagePoint>,
) -> Result<()> {
    for pair in points.windows(2) {
        segment(pane, overlay, pair[0], pair[1])?;
    }
    if let (Some(last), Some(cursor)) = (points.last(), cursor) {
        segment(pane, overlay, *last, cursor)?;
    }
    Ok(())
}

fn segment(
    pane: &Pane<'_>,
    overlay: &mut OverlayCommands,
    start: ImagePoint,
    end: ImagePoint,
) -> Result<()> {
    pane.line(
        overlay,
        image_point([start.y(), start.x()]),
        image_point([end.y(), end.x()]),
        MEASUREMENT_COLOR,
    )
}

fn length(
    pane: &Pane<'_>,
    overlay: &mut OverlayCommands,
    app: &SnapApp,
    start: ImagePoint,
    end: ImagePoint,
) -> Result<()> {
    segment(pane, overlay, start, end)?;
    let a = [start.y(), start.x()];
    let b = [end.y(), end.x()];
    if let Ok(mm) = app.measured_length(a, b) {
        pane.length_label(
            overlay,
            image_point(a),
            image_point(b),
            format!("{mm:.1} mm"),
        )?;
    }
    Ok(())
}
