//! Linked voxel cursor projection for the native Métis display list.

use anyhow::{bail, Result};
use metis_platform::Color;
use metis_ui_lang::DisplayList;

use super::super::frame::RenderedView;
use super::geometry::NativeViewport;
use crate::ui::map_voxel_to_view_row_col;

pub(crate) const CROSSHAIR_COLOR: Color = Color::rgba(0, 229, 255, 255);

/// Build host-neutral linked-cursor lines over the three image rectangles.
///
/// The cursor is stored in volume order `[z, y, x]`. Each plane maps that
/// coordinate to source row/column using the same inverse used by pointer
/// interactions, then applies the exact pixel orientation transform before
/// projecting into its native image rectangle. The returned commands are
/// clipped to the panel so the overlay cannot paint application chrome.
pub(crate) fn crosshair_overlay(
    views: &[RenderedView; 3],
    viewports: &[NativeViewport; 3],
    shape: Option<[usize; 3]>,
    cursor: Option<[usize; 3]>,
    visible: bool,
) -> Result<DisplayList> {
    let mut overlay = DisplayList::default();
    let (Some(shape), Some(cursor)) = (shape, cursor) else {
        return Ok(overlay);
    };
    if !visible
        || shape.contains(&0)
        || cursor
            .iter()
            .zip(shape)
            .any(|(value, limit)| *value >= limit)
    {
        return Ok(overlay);
    }
    for (view, viewport) in views.iter().zip(viewports) {
        let Some((row, column)) = map_voxel_to_view_row_col(view.axis, cursor) else {
            continue;
        };
        if column >= view.source_size[0] || row >= view.source_size[1] {
            continue;
        }
        let output_size = view.transform.output_size(view.source_size);
        if output_size.contains(&0) {
            continue;
        }
        let source_x = f64::from(u32::try_from(column)?) + 0.5;
        let source_y = f64::from(u32::try_from(row)?) + 0.5;
        let [output_x, output_y] = view
            .transform
            .source_to_output_coordinates([source_x, source_y], view.source_size);
        let output_width = f64::from(u32::try_from(output_size[0])?);
        let output_height = f64::from(u32::try_from(output_size[1])?);
        let point_x = viewport.image.x + output_x / output_width * viewport.image.width;
        let point_y = viewport.image.y + output_y / output_height * viewport.image.height;
        let left = viewport.image.x.max(viewport.panel.x);
        let right =
            (viewport.image.x + viewport.image.width).min(viewport.panel.x + viewport.panel.width);
        let top = viewport.image.y.max(viewport.panel.y);
        let bottom = (viewport.image.y + viewport.image.height)
            .min(viewport.panel.y + viewport.panel.height);
        if !(left < right && top < bottom && point_x.is_finite() && point_y.is_finite()) {
            continue;
        }
        let row_y = screen_coordinate(point_y, "crosshair row")?;
        let column_x = screen_coordinate(point_x, "crosshair column")?;
        overlay.append_line(
            (screen_coordinate(left, "crosshair left")?, row_y),
            (screen_coordinate(right, "crosshair right")?, row_y),
            CROSSHAIR_COLOR,
        )?;
        overlay.append_line(
            (column_x, screen_coordinate(top, "crosshair top")?),
            (column_x, screen_coordinate(bottom, "crosshair bottom")?),
            CROSSHAIR_COLOR,
        )?;
    }
    Ok(overlay)
}

fn screen_coordinate(value: f64, label: &str) -> Result<i32> {
    if !value.is_finite() || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
        bail!("{label} coordinate is outside the native display range")
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "finite display coordinates are checked against the i32 host contract"
    )]
    Ok(value.round() as i32)
}
