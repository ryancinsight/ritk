//! Clinical corners and scale ruler for the native overlay.
//!
//! The demographic corners, cursor readout and spacing ruler build on the
//! panel chrome in the parent module; this module owns their layout.

use anyhow::Result;
use metis_platform::rasterizer::CornerRadius;
use metis_platform::Rect;
use metis_ui_lang::DisplayCommand;

use super::super::frame::RenderedView;
use super::crosshair::screen_coordinate;
use super::geometry::NativeViewport;
use super::overlay::{
    ClinicalOverlay, OverlayCommands, OVERLAY_BAR_HEIGHT, OVERLAY_MARGIN, OVERLAY_TEXT,
};
use super::text::{
    fit_text, text_command, text_style, TextStyle, PANEL_DETAIL_SIZE, PANEL_TITLE_SIZE,
};

const CORNER_GAP: i32 = 8;

pub(crate) fn scale_bar_for_scale(px_per_mm: f64, max_px: f64) -> Option<(f64, f64)> {
    if !px_per_mm.is_finite() || !max_px.is_finite() || px_per_mm <= 0.0 || max_px <= 0.0 {
        return None;
    }
    let cap = max_px.min(MAX_SCALE_BAR_PX);
    let mut best: Option<(f64, f64)> = None;
    let mut magnitude = 0.001;
    while magnitude <= 100_000.0 {
        for mantissa in [1.0, 2.0, 5.0] {
            let length_mm = magnitude * mantissa;
            let width_px = length_mm * px_per_mm;
            if width_px <= cap {
                best = Some((length_mm, width_px));
            }
        }
        magnitude *= 10.0;
    }
    let (length_mm, width_px) = best?;
    if width_px < MIN_SCALE_BAR_PX {
        return None;
    }
    Some((length_mm, width_px))
}

/// Longest scale bar the overlay draws, in screen pixels.
const MAX_SCALE_BAR_PX: f64 = 140.0;
/// Shortest scale bar worth drawing, in screen pixels.
const MIN_SCALE_BAR_PX: f64 = 24.0;
/// Thickness of the scale bar, in screen pixels.
const SCALE_BAR_HEIGHT: i32 = 3;

/// Scale bar for one composed view, drawn bottom-right inside the image.
///
/// The bar length follows the displayed spacing so one bar always reads an
/// exact physical length; cramped panes keep their bars without a ruler.
pub(crate) fn append_scale_bar(
    overlay: &mut OverlayCommands,
    view: &RenderedView,
    viewport: &NativeViewport,
    panel_y: i32,
    panel_height: i32,
) -> Result<()> {
    if view.frame.width() == 0 || view.frame.height() == 0 {
        return Ok(());
    }
    let spacing = view.frame.display_spacing().values();
    let px_per_unit_x = viewport.image.width / f64::from(view.frame.width());
    let mm_per_px = spacing[1] / px_per_unit_x;
    if !mm_per_px.is_finite() || mm_per_px <= 0.0 {
        return Ok(());
    }
    let max_px = viewport.image.width - f64::from(2 * OVERLAY_MARGIN);
    let Some((length_mm, bar_px)) = scale_bar_for_scale(1.0 / mm_per_px, max_px) else {
        return Ok(());
    };
    let style = text_style(OVERLAY_TEXT, PANEL_DETAIL_SIZE)?;
    let line = screen_coordinate(style.line_height(), "scale bar label height")?;
    let bar_width = screen_coordinate(bar_px, "scale bar width")?;
    let label = format!("{} mm", format_scale_length(length_mm));
    let label_width = screen_coordinate(style.advance(&label).round(), "scale bar label width")?;
    let right = screen_coordinate(
        (viewport.image.x + viewport.image.width).round(),
        "scale bar right edge",
    )? - OVERLAY_MARGIN;
    let image_bottom = screen_coordinate(
        (viewport.image.y + viewport.image.height).round(),
        "scale bar bottom edge",
    )? - OVERLAY_MARGIN;
    let bar_top = image_bottom - SCALE_BAR_HEIGHT;
    let label_top = bar_top - 2 - line;
    let chrome_top = panel_y + OVERLAY_BAR_HEIGHT + OVERLAY_MARGIN;
    let chrome_bottom = panel_y + panel_height - OVERLAY_BAR_HEIGHT - OVERLAY_MARGIN;
    if label_top < chrome_top || image_bottom > chrome_bottom {
        return Ok(());
    }
    let bar = Rect::new(right - bar_width, bar_top, bar_width, SCALE_BAR_HEIGHT);
    let label_x = right - label_width;
    if overlay.overlaps_text(bar)
        || style
            .extent(label_x, label_top, &label)
            .is_some_and(|bounds| overlay.overlaps_text(bounds))
    {
        return Ok(());
    }
    overlay.push(DisplayCommand::FillRect {
        rect: bar,
        radius: CornerRadius::SQUARE,
        color: OVERLAY_TEXT,
    });
    overlay.push(text_command(label, label_x, label_top, style));
    Ok(())
}

/// Demographic corners for one composed view.
///
/// Mirrors the eframe four-corner layout without duplicating the title and
/// footer bars: patient data top-left, series data top-right, spacing and
/// zoom bottom-left, cursor readout bottom-right. Anonymized panes blank
/// patient corners while keeping modality and measurements.
pub(crate) fn append_clinical_corners(
    overlay: &mut OverlayCommands,
    panel_x: i32,
    panel_y: i32,
    panel_width: i32,
    panel_height: i32,
    clinical: &ClinicalOverlay<'_>,
) -> Result<()> {
    let style = text_style(OVERLAY_TEXT, PANEL_DETAIL_SIZE)?;
    let orientation_style = text_style(OVERLAY_TEXT, PANEL_TITLE_SIZE)?;
    // The center row belongs to the lateral orientation markers. The
    // top and bottom orientation markers occupy their own rows.
    let orientation_height = orientation_style
        .extent(0, 0, "R")
        .map_or(0, |bounds| bounds.height);
    let chrome = orientation_height.max(OVERLAY_BAR_HEIGHT);
    let top = panel_y + chrome + orientation_height + OVERLAY_MARGIN;
    let bottom = panel_y + panel_height - chrome - orientation_height - OVERLAY_MARGIN;
    let middle = panel_y + panel_height / 2;
    let upper_height = middle - orientation_height / 2 - OVERLAY_MARGIN - top;
    let lower_height = bottom - middle - orientation_height / 2 - OVERLAY_MARGIN;
    let left_edge = panel_x + OVERLAY_MARGIN;
    let right_edge =
        panel_x + panel_width - OVERLAY_MARGIN - super::super::slice_scrollbar::RIGHT_GUTTER;
    let top_left: Vec<String> = if clinical.anonymize {
        vec!["Anonymous".to_owned()]
    } else {
        let name = clinical
            .patient_name
            .filter(|name| !name.is_empty())
            .unwrap_or("(no name)");
        let mut lines = vec![name.to_owned()];
        if let Some(id) = clinical.patient_id.filter(|id| !id.is_empty()) {
            lines.push(format!("ID: {id}"));
        }
        lines
    };
    let top_right: Vec<String> = if clinical.anonymize {
        clinical
            .modality
            .iter()
            .map(|text| text.to_string())
            .collect()
    } else {
        [
            clinical.series_description,
            clinical.modality,
            clinical.study_date,
        ]
        .into_iter()
        .flatten()
        .filter(|text| !text.is_empty())
        .map(|text| {
            if Some(text) == clinical.study_date {
                format!("Date: {text}")
            } else {
                text.to_owned()
            }
        })
        .collect()
    };
    let spacing = format!(
        "{:.2} × {:.2} × {:.2} mm",
        clinical.spacing[2], clinical.spacing[1], clinical.spacing[0]
    );
    let bottom_left = vec![
        spacing,
        format!("Zoom: {:.0}%", f64::from(clinical.zoom) * 100.0),
    ];
    let bottom_right: Vec<String> = clinical
        .cursor_value
        .map(|value| format!("Cursor value: {value:.0}"))
        .into_iter()
        .collect();

    let side_width = (right_edge - left_edge - CORNER_GAP) / 2;
    let left = HorizontalAnchor::Left(left_edge);
    let right = HorizontalAnchor::Right(right_edge);
    let upper = VerticalAnchor::Top(top);
    let lower = VerticalAnchor::Bottom(bottom);
    for (lines, horizontal, vertical, height) in [
        (top_left, left, upper, upper_height),
        (top_right, right, upper, upper_height),
        (bottom_left, left, lower, lower_height),
        (bottom_right, right, lower, lower_height),
    ] {
        append_lines(
            overlay,
            &lines,
            style,
            [side_width, height],
            horizontal,
            vertical,
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum HorizontalAnchor {
    Left(i32),
    Right(i32),
}

#[derive(Clone, Copy)]
enum VerticalAnchor {
    Top(i32),
    Bottom(i32),
}

fn append_lines(
    overlay: &mut OverlayCommands,
    lines: &[String],
    style: TextStyle,
    [width, height]: [i32; 2],
    horizontal: HorizontalAnchor,
    vertical: VerticalAnchor,
) {
    let mut used = 0;
    for line in lines {
        let line = fit_text(line, width, style);
        let Some(bounds) = style.extent(0, 0, &line) else {
            continue;
        };
        if bounds.height > height - used {
            break;
        }
        let x = match horizontal {
            HorizontalAnchor::Left(edge) => edge,
            HorizontalAnchor::Right(edge) => edge - bounds.width,
        };
        let y = match vertical {
            VerticalAnchor::Top(edge) => edge + used,
            VerticalAnchor::Bottom(edge) => edge - used - bounds.height,
        };
        overlay.push(text_command(line, x - bounds.x, y - bounds.y, style));
        used += bounds.height;
    }
}

/// Whole millimetres print without decimals; fractional steps keep one.
fn format_scale_length(length_mm: f64) -> String {
    if length_mm.fract() == 0.0 {
        format!("{:.0}", length_mm)
    } else {
        format!("{length_mm}")
    }
}
