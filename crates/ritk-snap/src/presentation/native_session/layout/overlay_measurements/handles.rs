use super::{image_point, OverlayCommands, Pane, POINT_COLOR};
use crate::tools::interaction::Annotation;
use anyhow::Result;
use metis_ui_lang::DisplayCommand;

pub(super) fn append(
    pane: &Pane<'_>,
    annotation: &Annotation,
    overlay: &mut OverlayCommands,
) -> Result<()> {
    let (Annotation::OpenPolygon { points, .. } | Annotation::ClosedPolygon { points, .. }) =
        annotation
    else {
        return Ok(());
    };
    append_points(pane, points.iter().copied(), overlay)
}

fn append_points(
    pane: &Pane<'_>,
    points: impl IntoIterator<Item = [f32; 2]>,
    overlay: &mut OverlayCommands,
) -> Result<()> {
    for point in points {
        let (x, y) = pane.point(image_point(point))?;
        for (start, end) in [
            ((x.saturating_sub(2), y), (x.saturating_add(2), y)),
            ((x, y.saturating_sub(2)), (x, y.saturating_add(2))),
        ] {
            overlay.push(DisplayCommand::DrawLine {
                start,
                end,
                color: POINT_COLOR,
            });
        }
    }
    Ok(())
}
