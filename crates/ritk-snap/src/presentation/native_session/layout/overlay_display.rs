//! Native histogram window panel and per-pane colorbar overlays.
//!
//! The histogram strip reproduces the egui window panel
//! ([`crate::ui::histogram`]) with log-scaled bars, the W/L band and the
//! centre line; the colorbar reproduces [`crate::ui::colorbar`] as a sampled
//! ramp with the window range labelled. Both project through the same
//! viewport geometry as the measurement overlays.

use anyhow::{anyhow, Result};
use eunomia::FloatElement;
use iris::color::{ColorMap, Normalized};
use metis_platform::rasterizer::CornerRadius;
use metis_platform::{Color, Rect};
use metis_ui_lang::DisplayCommand;

use super::crosshair::screen_coordinate;
use super::geometry::NativeViewport;
use super::overlay::{OVERLAY_BAR_HEIGHT, OVERLAY_MARGIN};
use super::text::{text_command, text_style, PANEL_DETAIL_SIZE};
use super::OverlayCommands;
use crate::render::histogram::{bar_height_log, histogram_peak_count, wl_to_x, Histogram};
use crate::render::NamedColorMap;

/// Height of the histogram strip docked at the active pane bottom.
pub(crate) const HISTOGRAM_STRIP_HEIGHT: i32 = 96;
/// Height available to histogram bars inside the strip.
const HISTOGRAM_BAR_HEIGHT: f32 = 60.0;
/// Room reserved below the bars for the min/max labels.
const HISTOGRAM_LABEL_ROOM: i32 = 20;
/// Horizontal inset of the histogram canvas inside the strip.
const HISTOGRAM_PAD: i32 = 8;
/// Width of the per-pane colorbar strip.
pub(crate) const COLORBAR_WIDTH: i32 = 14;
/// Colormap ramp samples; `u8` indices convert to `f32` exactly.
const COLORBAR_SEGMENTS: u8 = 64;

/// Panel pixel coordinate as an `f32` canvas bound.
///
/// `i64::from` is lossless and `from_integer` is exact below 2²⁴; native
/// panel extents stay far below that bound.
pub(crate) fn canvas_coord(value: i32) -> f32 {
    f32::from_integer(i64::from(value))
}

const HISTOGRAM_BACKGROUND: Color = Color::rgba(20, 20, 20, 255);
const HISTOGRAM_BAR: Color = Color::rgba(160, 160, 160, 255);
const HISTOGRAM_BAND: Color = Color::rgba(40, 100, 200, 50);
const HISTOGRAM_CENTER: Color = Color::rgba(255, 190, 60, 255);
const LABEL_COLOR: Color = Color::rgba(255, 255, 255, 255);

/// Format one window intensity for overlay labels.
///
/// Mirrors the egui colorbar: one decimal, integers past ±1000.
pub(crate) fn intensity_label(value: f32) -> String {
    if value.abs() >= 1000.0 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

/// X-range of the W/L band inside a histogram canvas.
pub(crate) fn window_band_x(
    center: f32,
    width: f32,
    hist_min: f32,
    hist_max: f32,
    x_left: f32,
    x_right: f32,
) -> (f32, f32) {
    let half = width / 2.0;
    (
        wl_to_x(center - half, hist_min, hist_max, x_left, x_right),
        wl_to_x(center + half, hist_min, hist_max, x_left, x_right),
    )
}

/// Append one pane's colorbar: a 64-sample ramp of `map` with the window
/// `[center - width / 2, center + width / 2]` labelled beside it.
///
/// The top segment samples the colormap maximum, the bottom its minimum.
pub(crate) fn append_colorbar_overlay(
    overlay: &mut OverlayCommands,
    viewport: &NativeViewport,
    map: NamedColorMap,
    center: f32,
    width: f32,
) -> Result<()> {
    let panel = viewport.panel_rect()?;
    let image = &viewport.image;
    // The strip sits left of the slice-scrollbar gutter so the scrollbar
    // rendered later never covers the ramp.
    let strip_x = screen_coordinate(
        f64::from(panel.x + panel.width)
            - f64::from(super::super::slice_scrollbar::RIGHT_GUTTER + COLORBAR_WIDTH),
        "colorbar x",
    )?;
    let strip_y = screen_coordinate(image.y, "colorbar top")?;
    let strip_height = screen_coordinate(image.height, "colorbar height")?;
    if strip_height <= 0 {
        return Ok(());
    }
    let segments = usize::from(COLORBAR_SEGMENTS);
    for index in 0..segments {
        let sample = u8::try_from(index)
            .map_err(|_| anyhow!("colorbar segment {index} exceeds the ramp table"))?;
        let top = strip_y
            + (i32::try_from(index)
                .map_err(|_| anyhow!("colorbar segment {index} exceeds the pane extent"))?
                * strip_height)
                / i32::from(COLORBAR_SEGMENTS);
        let bottom = strip_y
            + (i32::try_from(index + 1)
                .map_err(|_| anyhow!("colorbar segment {index} exceeds the pane extent"))?
                * strip_height)
                / i32::from(COLORBAR_SEGMENTS);
        let t = 1.0 - f32::from(sample) / f32::from(COLORBAR_SEGMENTS - 1);
        let value = Normalized::new(t)
            .map_err(|_| anyhow!("colorbar sample {t} leaves the unit interval"))?;
        let [red, green, blue, _] = map.sample(value).to_rgba8();
        overlay.push(DisplayCommand::FillRect {
            rect: Rect::new(strip_x, top, COLORBAR_WIDTH, (bottom - top).max(1)),
            radius: CornerRadius::SQUARE,
            color: Color::rgba(red, green, blue, 255),
        });
    }
    overlay.push(DisplayCommand::DrawBorder {
        rect: Rect::new(
            strip_x - 1,
            strip_y - 1,
            COLORBAR_WIDTH + 2,
            strip_height + 2,
        ),
        width: 1,
        radius: CornerRadius::SQUARE,
        color: Color::rgba(128, 128, 128, 255),
    });
    let half = width / 2.0;
    let style = text_style(LABEL_COLOR, PANEL_DETAIL_SIZE)?;
    let orientation_style = text_style(LABEL_COLOR, super::text::PANEL_TITLE_SIZE)?;
    let orientation_height = orientation_style
        .extent(0, 0, "R")
        .map_or(0, |bounds| bounds.height);
    let line = screen_coordinate(style.line_height(), "colorbar label height")?;
    let center_intensity = center;
    let center = panel.y + panel.height / 2;
    let orientation_top = center - orientation_height / 2;
    let orientation_bottom = orientation_top + orientation_height;
    // Clinical corners show at most three upper and two lower detail lines.
    let chrome = OVERLAY_BAR_HEIGHT.max(orientation_height) + orientation_height + OVERLAY_MARGIN;
    let upper = panel.y + chrome + 3 * line;
    let lower = panel.y + panel.height - chrome - 2 * line;
    let right_rail = panel.x + panel.width / 2;
    for (value, y) in [
        (
            center_intensity + half,
            (strip_y + OVERLAY_MARGIN).max(upper),
        ),
        (
            center_intensity - half,
            (strip_y + strip_height - line - OVERLAY_MARGIN).min(lower - line),
        ),
        (center_intensity, orientation_bottom + OVERLAY_MARGIN),
    ] {
        let label = intensity_label(value);
        let Some(bounds) = style.extent(0, y, &label) else {
            continue;
        };
        let x = strip_x - OVERLAY_MARGIN - bounds.width - bounds.x;
        let Some(bounds) = style.extent(x, y, &label) else {
            continue;
        };
        if bounds.x >= right_rail
            && bounds.y >= upper
            && bounds.y + bounds.height <= lower
            && (bounds.y + bounds.height <= orientation_top || bounds.y >= orientation_bottom)
            && !overlay.overlaps_text(bounds)
        {
            overlay.push(text_command(label, x, y, style));
        }
    }
    Ok(())
}

/// Append the histogram window panel for the active pane: log-scaled bars,
/// the W/L band with its centre line, and the histogram range labels.
pub(crate) fn append_histogram_overlay(
    overlay: &mut OverlayCommands,
    viewport: &NativeViewport,
    histogram: &Histogram,
    center: f32,
    width: f32,
) -> Result<()> {
    if histogram.bins == 0 || histogram.counts.is_empty() {
        return Ok(());
    }
    let panel = viewport.panel_rect()?;
    let strip_y = screen_coordinate(
        f64::from(panel.y + panel.height) - f64::from(HISTOGRAM_STRIP_HEIGHT),
        "histogram strip top",
    )?;
    overlay.push(DisplayCommand::FillRect {
        rect: Rect::new(panel.x, strip_y, panel.width, HISTOGRAM_STRIP_HEIGHT),
        radius: CornerRadius::SQUARE,
        color: HISTOGRAM_BACKGROUND,
    });
    let canvas_left = screen_coordinate(f64::from(panel.x + HISTOGRAM_PAD), "histogram left")?;
    let canvas_right = screen_coordinate(
        f64::from(panel.x + panel.width - HISTOGRAM_PAD),
        "histogram right",
    )?;
    let bars_bottom = strip_y + HISTOGRAM_STRIP_HEIGHT - HISTOGRAM_LABEL_ROOM;
    let peak = histogram_peak_count(histogram);
    for (bin, &count) in histogram.counts.iter().enumerate() {
        let height = bar_height_log(count, peak, HISTOGRAM_BAR_HEIGHT);
        if height <= 0.0 {
            continue;
        }
        let bin_left = f64::from(canvas_left)
            + f64::from(u32::try_from(bin).map_err(|_| anyhow!("histogram bin {bin} overflows"))?)
                * f64::from(canvas_right - canvas_left)
                / f64::from(
                    u32::try_from(histogram.bins)
                        .map_err(|_| anyhow!("histogram bin count overflows"))?,
                );
        let x0 = screen_coordinate(bin_left, "histogram bar left")?;
        let x1 = screen_coordinate(
            bin_left
                + f64::from(canvas_right - canvas_left)
                    / f64::from(
                        u32::try_from(histogram.bins)
                            .map_err(|_| anyhow!("histogram bin count overflows"))?,
                    ),
            "histogram bar right",
        )?;
        let y0 = screen_coordinate(
            f64::from(bars_bottom) - f64::from(height),
            "histogram bar top",
        )?;
        overlay.push(DisplayCommand::FillRect {
            rect: Rect::new(x0, y0, (x1 - x0).max(1), bars_bottom - y0),
            radius: CornerRadius::SQUARE,
            color: HISTOGRAM_BAR,
        });
    }
    let (band_left, band_right) = window_band_x(
        center,
        width,
        histogram.min(),
        histogram.max(),
        canvas_coord(canvas_left),
        canvas_coord(canvas_right),
    );
    let band_x0 = screen_coordinate(f64::from(band_left), "window band left")?;
    let band_x1 = screen_coordinate(f64::from(band_right), "window band right")?;
    overlay.push(DisplayCommand::FillRect {
        rect: Rect::new(
            band_x0,
            strip_y,
            (band_x1 - band_x0).max(1),
            HISTOGRAM_STRIP_HEIGHT,
        ),
        radius: CornerRadius::SQUARE,
        color: HISTOGRAM_BAND,
    });
    let center_x = screen_coordinate(
        f64::from(wl_to_x(
            center,
            histogram.min(),
            histogram.max(),
            canvas_coord(canvas_left),
            canvas_coord(canvas_right),
        )),
        "window centre",
    )?;
    overlay.push(DisplayCommand::DrawLine {
        start: (center_x, strip_y),
        end: (center_x, strip_y + HISTOGRAM_STRIP_HEIGHT),
        color: HISTOGRAM_CENTER,
    });
    let style = text_style(LABEL_COLOR, PANEL_DETAIL_SIZE)?;
    overlay.push(text_command(
        intensity_label(histogram.min()),
        canvas_left,
        strip_y + HISTOGRAM_STRIP_HEIGHT - 16,
        style,
    ));
    overlay.push(text_command(
        intensity_label(histogram.max()),
        canvas_right - 60,
        strip_y + HISTOGRAM_STRIP_HEIGHT - 16,
        style,
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intensity_label_matches_eframe_colorbar_rule() {
        assert_eq!(intensity_label(12.345), "12.3");
        assert_eq!(intensity_label(-0.04), "-0.0");
        assert_eq!(intensity_label(1234.56), "1235");
        assert_eq!(intensity_label(-1500.0), "-1500");
    }

    #[test]
    fn window_band_x_marks_centered_window() {
        let (left, right) = window_band_x(100.0, 40.0, 0.0, 200.0, 10.0, 210.0);
        assert!((left - 90.0).abs() < 1e-4, "band left {left}");
        assert!((right - 130.0).abs() < 1e-4, "band right {right}");
    }

    #[test]
    fn window_band_x_clamps_outside_histogram() {
        let (left, right) = window_band_x(100.0, 400.0, 0.0, 200.0, 10.0, 210.0);
        assert_eq!((left, right), (10.0, 210.0));
    }
}
