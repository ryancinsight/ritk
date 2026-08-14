//! Generate the temporal-synchronization figure used by the RITK mdBook.
//!
//! The example synthesizes a delayed finite signal, estimates its lag through
//! RITK's public API, verifies the reported diagnostics against independently
//! interpolated residuals, and renders before/profile/after/residual panels.
#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: example/test diagnostic output"
)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use leto::Array1;
use ritk_registration::{TemporalSync, TemporalSyncConfig, TemporalSyncResult, TemporalSyncStatus};

const SAMPLE_COUNT: usize = 240;
const FRAME_SPACING_SECONDS: f64 = 0.04;
const TRUE_DELAY_FRAMES: f64 = 7.25;
const SEARCH_RANGE_FRAMES: usize = 20;
const MINIMUM_CORRELATION: f64 = 0.85;
const FIGURE_WIDTH: u32 = 1_180;
const FIGURE_HEIGHT: u32 = 760;
const PANEL_WIDTH: f64 = 555.0;
const PANEL_HEIGHT: f64 = 235.0;

struct ExampleData {
    reference: Vec<f64>,
    moving: Vec<f64>,
    aligned: Vec<Option<f64>>,
    residuals: Vec<Option<f64>>,
    profile: Vec<(f64, f64)>,
    result: TemporalSyncResult,
    unaligned_rms: f64,
}

fn waveform(sample: f64) -> f64 {
    (sample * 0.105).sin()
        + 0.42 * (sample * 0.041 + 0.6).cos()
        + 0.18 * (sample * 0.227 - 0.4).sin()
}

fn interpolate(signal: &[f64], coordinate: f64) -> Option<f64> {
    if coordinate < 0.0 || coordinate > signal.len().saturating_sub(1) as f64 {
        return None;
    }
    let lower = coordinate.floor() as usize;
    let fraction = coordinate - lower as f64;
    let lower_value = *signal.get(lower)?;
    if fraction == 0.0 {
        return Some(lower_value);
    }
    let upper_value = *signal.get(lower.checked_add(1)?)?;
    Some((upper_value - lower_value).mul_add(fraction, lower_value))
}

fn example_data() -> Result<ExampleData> {
    let reference = (0..SAMPLE_COUNT)
        .map(|index| waveform(index as f64))
        .collect::<Vec<_>>();
    let moving = (0..SAMPLE_COUNT)
        .map(|index| {
            let sample = index as f64;
            waveform(sample - TRUE_DELAY_FRAMES) + 0.012 * (sample * 0.71).sin()
        })
        .collect::<Vec<_>>();
    let reference_array =
        Array1::from_vec([SAMPLE_COUNT], reference.clone()).context("reference signal shape")?;
    let moving_array =
        Array1::from_vec([SAMPLE_COUNT], moving.clone()).context("moving signal shape")?;
    let config = TemporalSyncConfig::try_new(
        FRAME_SPACING_SECONDS,
        SEARCH_RANGE_FRAMES,
        MINIMUM_CORRELATION,
    )?;
    let synchronizer = TemporalSync::with_config(config);
    let result = synchronizer.synchronize(&reference_array, &moving_array)?;
    let profile = synchronizer
        .correlation_profile(&reference_array, &moving_array)?
        .iter()
        .filter_map(|sample| {
            sample
                .correlation()
                .map(|correlation| (sample.lag_frames() as f64, correlation))
        })
        .collect::<Vec<_>>();

    let aligned = (0..SAMPLE_COUNT)
        .map(|index| interpolate(&moving, index as f64 + result.shift_frames()))
        .collect::<Vec<_>>();
    let residuals = reference
        .iter()
        .zip(&aligned)
        .map(|(&reference_value, aligned_value)| {
            aligned_value.map(|moving_value| reference_value - moving_value)
        })
        .collect::<Vec<_>>();
    let aligned_values = residuals.iter().flatten().copied().collect::<Vec<_>>();
    let residual_rms = (aligned_values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        / aligned_values.len() as f64)
        .sqrt();
    let residual_max = aligned_values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let unaligned_rms = (reference
        .iter()
        .zip(&moving)
        .map(|(&reference_value, &moving_value)| {
            let residual = reference_value - moving_value;
            residual * residual
        })
        .sum::<f64>()
        / SAMPLE_COUNT as f64)
        .sqrt();

    let numerical_bound = 256.0 * f64::EPSILON * SAMPLE_COUNT as f64;
    if (result.shift_frames() - TRUE_DELAY_FRAMES).abs() >= 0.25 {
        bail!(
            "estimated shift {:.4} does not improve on integer quantization of the {:.2}-frame delay",
            result.shift_frames(),
            TRUE_DELAY_FRAMES
        );
    }
    if result.peak_correlation() < 0.99 || result.status() != TemporalSyncStatus::Accepted {
        bail!(
            "known delayed pair must be accepted with correlation >= 0.99, got {:.6}",
            result.peak_correlation()
        );
    }
    if (residual_rms - result.residual_rms()).abs() > numerical_bound
        || (residual_max - result.residual_max_abs()).abs() > numerical_bound
        || aligned_values.len() != result.overlap_samples()
    {
        bail!("independent aligned residuals disagree with RITK diagnostics");
    }
    if result.residual_rms() >= 0.15 * unaligned_rms {
        bail!(
            "alignment must reduce RMS residual by at least 85%, got before={unaligned_rms:.6}, after={:.6}",
            result.residual_rms()
        );
    }

    Ok(ExampleData {
        reference,
        moving,
        aligned,
        residuals,
        profile,
        result,
        unaligned_rms,
    })
}

fn map_x(index: f64, left: f64, width: f64, maximum: f64) -> f64 {
    left + index / maximum * width
}

fn map_y(value: f64, top: f64, height: f64, minimum: f64, maximum: f64) -> f64 {
    top + (maximum - value) / (maximum - minimum) * height
}

fn points(
    values: impl Iterator<Item = (f64, f64)>,
    bounds: (f64, f64, f64, f64),
    ranges: (f64, f64, f64, f64),
) -> String {
    let (left, top, width, height) = bounds;
    let (x_min, x_max, y_min, y_max) = ranges;
    values
        .map(|(x, y)| {
            format!(
                "{:.2},{:.2}",
                map_x(x - x_min, left, width, x_max - x_min),
                map_y(y, top, height, y_min, y_max)
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn panel(svg: &mut String, x: f64, y: f64, title: &str, subtitle: &str) -> Result<()> {
    writeln!(
        svg,
        "<rect x=\"{x}\" y=\"{y}\" width=\"{PANEL_WIDTH}\" height=\"{PANEL_HEIGHT}\" class=\"panel\"/><text x=\"{}\" y=\"{}\" class=\"panel-title\">{title}</text><text x=\"{}\" y=\"{}\" class=\"subtitle\">{subtitle}</text>",
        x + 18.0,
        y + 28.0,
        x + 18.0,
        y + 48.0
    )?;
    Ok(())
}

fn axes(svg: &mut String, left: f64, top: f64, width: f64, height: f64) -> Result<()> {
    writeln!(
        svg,
        "<line x1=\"{left}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"axis\"/><line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{}\" class=\"axis\"/>",
        top + height,
        left + width,
        top + height,
        top + height
    )?;
    Ok(())
}

fn draw_before(svg: &mut String, data: &ExampleData) -> Result<()> {
    let panel_x = 20.0;
    let panel_y = 65.0;
    panel(
        svg,
        panel_x,
        panel_y,
        "1 · BEFORE — same event, different clocks",
        "The orange acquisition is delayed; peaks and troughs do not coincide.",
    )?;
    let bounds = (panel_x + 45.0, panel_y + 65.0, 485.0, 135.0);
    axes(svg, bounds.0, bounds.1, bounds.2, bounds.3)?;
    let ranges = (0.0, (SAMPLE_COUNT - 1) as f64, -1.7, 1.7);
    let reference_points = points(
        data.reference
            .iter()
            .enumerate()
            .map(|(index, &value)| (index as f64, value)),
        bounds,
        ranges,
    );
    let moving_points = points(
        data.moving
            .iter()
            .enumerate()
            .map(|(index, &value)| (index as f64, value)),
        bounds,
        ranges,
    );
    writeln!(
        svg,
        "<text x=\"{}\" y=\"{}\" class=\"axis-label\">+1.7</text><text x=\"{}\" y=\"{}\" class=\"axis-label\">−1.7</text><text x=\"{}\" y=\"{}\" text-anchor=\"end\" class=\"axis-label\">sample index →</text><polyline points=\"{reference_points}\" class=\"reference\"/><polyline points=\"{moving_points}\" class=\"moving\"/><text x=\"{}\" y=\"{}\" class=\"legend reference-text\">reference</text><text x=\"{}\" y=\"{}\" class=\"legend moving-text\">moving (delayed {:.2} frames)</text>",
        bounds.0 - 7.0,
        bounds.1 + 5.0,
        bounds.0 - 7.0,
        bounds.1 + bounds.3,
        bounds.0 + bounds.2,
        bounds.1 + bounds.3 - 5.0,
        panel_x + 60.0,
        panel_y + 222.0,
        panel_x + 150.0,
        panel_y + 222.0,
        TRUE_DELAY_FRAMES
    )?;
    Ok(())
}

fn draw_profile(svg: &mut String, data: &ExampleData) -> Result<()> {
    let panel_x = 605.0;
    let panel_y = 65.0;
    panel(
        svg,
        panel_x,
        panel_y,
        "2 · SEARCH — normalized correlation by lag",
        "The maximum identifies the delay; the dashed line is the acceptance threshold.",
    )?;
    let bounds = (panel_x + 50.0, panel_y + 65.0, 475.0, 135.0);
    axes(svg, bounds.0, bounds.1, bounds.2, bounds.3)?;
    let ranges = (
        -(SEARCH_RANGE_FRAMES as f64),
        SEARCH_RANGE_FRAMES as f64,
        -0.55,
        1.05,
    );
    let profile_points = points(data.profile.iter().copied(), bounds, ranges);
    let threshold_y = map_y(MINIMUM_CORRELATION, bounds.1, bounds.3, ranges.2, ranges.3);
    let peak_x = map_x(
        data.result.shift_frames() - ranges.0,
        bounds.0,
        bounds.2,
        ranges.1 - ranges.0,
    );
    for &(lag, correlation) in &data.profile {
        let x = map_x(lag - ranges.0, bounds.0, bounds.2, ranges.1 - ranges.0);
        let y = map_y(correlation, bounds.1, bounds.3, ranges.2, ranges.3);
        writeln!(
            svg,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"2.4\" class=\"correlation-sample\"/>"
        )?;
    }
    writeln!(
        svg,
        "<text x=\"{}\" y=\"{}\" class=\"axis-label\">1.0</text><text x=\"{}\" y=\"{}\" class=\"axis-label\">0</text><text x=\"{}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\">−20</text><text x=\"{}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\">0</text><text x=\"{}\" y=\"{}\" text-anchor=\"middle\" class=\"axis-label\">+20 frames</text><line x1=\"{}\" y1=\"{threshold_y:.2}\" x2=\"{}\" y2=\"{threshold_y:.2}\" class=\"threshold\"/><text x=\"{}\" y=\"{}\" text-anchor=\"end\" class=\"threshold-label\">minimum r = {MINIMUM_CORRELATION:.2}</text><polyline points=\"{profile_points}\" class=\"correlation\"/><line x1=\"{peak_x:.2}\" y1=\"{}\" x2=\"{peak_x:.2}\" y2=\"{}\" class=\"peak\"/><text x=\"{}\" y=\"{}\" class=\"legend\">estimated {:.3} frames · r = {:.4}</text>",
        bounds.0 - 8.0,
        map_y(1.0, bounds.1, bounds.3, ranges.2, ranges.3) + 4.0,
        bounds.0 - 8.0,
        map_y(0.0, bounds.1, bounds.3, ranges.2, ranges.3) + 4.0,
        bounds.0,
        bounds.1 + bounds.3 + 14.0,
        map_x(-ranges.0, bounds.0, bounds.2, ranges.1 - ranges.0),
        bounds.1 + bounds.3 + 14.0,
        bounds.0 + bounds.2,
        bounds.1 + bounds.3 + 14.0,
        bounds.0,
        bounds.0 + bounds.2,
        bounds.0 + bounds.2 - 5.0,
        threshold_y - 5.0,
        bounds.1,
        bounds.1 + bounds.3,
        panel_x + 65.0,
        panel_y + 82.0,
        data.result.shift_frames(),
        data.result.peak_correlation()
    )?;
    Ok(())
}

fn draw_after(svg: &mut String, data: &ExampleData) -> Result<()> {
    let panel_x = 20.0;
    let panel_y = 325.0;
    panel(
        svg,
        panel_x,
        panel_y,
        "3 · AFTER — moving sampled at index + shift",
        "The estimated positive lag advances the delayed signal onto the reference clock.",
    )?;
    let bounds = (panel_x + 45.0, panel_y + 65.0, 485.0, 135.0);
    axes(svg, bounds.0, bounds.1, bounds.2, bounds.3)?;
    let ranges = (0.0, (SAMPLE_COUNT - 1) as f64, -1.7, 1.7);
    let reference_points = points(
        data.reference
            .iter()
            .enumerate()
            .map(|(index, &value)| (index as f64, value)),
        bounds,
        ranges,
    );
    let aligned_points = points(
        data.aligned
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.map(|value| (index as f64, value))),
        bounds,
        ranges,
    );
    writeln!(
        svg,
        "<text x=\"{}\" y=\"{}\" class=\"axis-label\">+1.7</text><text x=\"{}\" y=\"{}\" class=\"axis-label\">−1.7</text><text x=\"{}\" y=\"{}\" text-anchor=\"end\" class=\"axis-label\">sample index →</text><polyline points=\"{reference_points}\" class=\"reference\"/><polyline points=\"{aligned_points}\" class=\"aligned\"/><text x=\"{}\" y=\"{}\" class=\"legend reference-text\">reference</text><text x=\"{}\" y=\"{}\" class=\"legend aligned-text\">aligned moving</text>",
        bounds.0 - 7.0,
        bounds.1 + 5.0,
        bounds.0 - 7.0,
        bounds.1 + bounds.3,
        bounds.0 + bounds.2,
        bounds.1 + bounds.3 - 5.0,
        panel_x + 60.0,
        panel_y + 222.0,
        panel_x + 150.0,
        panel_y + 222.0
    )?;
    Ok(())
}

fn draw_residuals(svg: &mut String, data: &ExampleData) -> Result<()> {
    let panel_x = 605.0;
    let panel_y = 325.0;
    panel(
        svg,
        panel_x,
        panel_y,
        "4 · VERIFY — aligned residual in signal units",
        "Only the valid interpolated overlap contributes to RMS and maximum error.",
    )?;
    let bounds = (panel_x + 50.0, panel_y + 65.0, 475.0, 135.0);
    axes(svg, bounds.0, bounds.1, bounds.2, bounds.3)?;
    let residual_limit = data.result.residual_max_abs().max(0.02) * 1.2;
    let ranges = (
        0.0,
        (SAMPLE_COUNT - 1) as f64,
        -residual_limit,
        residual_limit,
    );
    let zero_y = map_y(0.0, bounds.1, bounds.3, ranges.2, ranges.3);
    let residual_points = points(
        data.residuals
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.map(|value| (index as f64, value))),
        bounds,
        ranges,
    );
    writeln!(
        svg,
        "<text x=\"{}\" y=\"{}\" class=\"axis-label\">+{residual_limit:.3}</text><text x=\"{}\" y=\"{}\" class=\"axis-label\">−{residual_limit:.3}</text><text x=\"{}\" y=\"{}\" text-anchor=\"end\" class=\"axis-label\">sample index →</text><line x1=\"{}\" y1=\"{zero_y:.2}\" x2=\"{}\" y2=\"{zero_y:.2}\" class=\"zero\"/><polyline points=\"{residual_points}\" class=\"residual\"/><text x=\"{}\" y=\"{}\" class=\"legend\">RMS {:.4} · max |e| {:.4} · overlap {} / {SAMPLE_COUNT}</text>",
        bounds.0 - 7.0,
        bounds.1 + 5.0,
        bounds.0 - 7.0,
        bounds.1 + bounds.3,
        bounds.0 + bounds.2,
        bounds.1 + bounds.3 - 5.0,
        bounds.0,
        bounds.0 + bounds.2,
        panel_x + 65.0,
        panel_y + 222.0,
        data.result.residual_rms(),
        data.result.residual_max_abs(),
        data.result.overlap_samples()
    )?;
    Ok(())
}

fn write_figure(path: &Path, data: &ExampleData) -> Result<()> {
    let reduction = 100.0 * (1.0 - data.result.residual_rms() / data.unaligned_rms);
    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {FIGURE_WIDTH} {FIGURE_HEIGHT}\"><rect width=\"{FIGURE_WIDTH}\" height=\"{FIGURE_HEIGHT}\" fill=\"#f8fafc\"/><style>.panel{{fill:#fff;stroke:#cbd5e1;stroke-width:1}}.panel-title{{font:600 17px sans-serif;fill:#172033}}.subtitle,.legend,.axis-label{{font:12px sans-serif;fill:#475569}}.axis{{stroke:#64748b;stroke-width:1}}.reference,.moving,.aligned,.correlation,.residual{{fill:none;stroke-width:2.2;stroke-linejoin:round}}.reference{{stroke:#2563eb}}.moving{{stroke:#f97316}}.aligned{{stroke:#16a34a;stroke-dasharray:6 3}}.correlation{{stroke:#7c3aed}}.correlation-sample{{fill:#7c3aed}}.residual{{stroke:#dc2626}}.threshold{{stroke:#f97316;stroke-dasharray:5 4}}.threshold-label{{font:11px sans-serif;fill:#c2410c}}.peak{{stroke:#16a34a;stroke-width:2}}.zero{{stroke:#94a3b8;stroke-dasharray:3 3}}.reference-text{{fill:#1d4ed8}}.moving-text{{fill:#c2410c}}.aligned-text{{fill:#15803d}}.metric{{font:600 16px sans-serif;fill:#172033}}.success{{font:700 18px sans-serif;fill:#15803d}}</style><text x=\"20\" y=\"34\" class=\"metric\">Temporal synchronization: measure → shift → verify</text>"
    )?;
    draw_before(&mut svg, data)?;
    draw_profile(&mut svg, data)?;
    draw_after(&mut svg, data)?;
    draw_residuals(&mut svg, data)?;
    writeln!(
        svg,
        "<rect x=\"20\" y=\"585\" width=\"1140\" height=\"145\" rx=\"8\" fill=\"#ecfdf5\" stroke=\"#86efac\"/><text x=\"45\" y=\"620\" class=\"success\">Alignment is visible and measured: RMS residual falls {reduction:.1}%</text><text x=\"45\" y=\"650\" class=\"metric\">Known delay</text><text x=\"180\" y=\"650\" class=\"legend\">{TRUE_DELAY_FRAMES:.3} frames</text><text x=\"340\" y=\"650\" class=\"metric\">Estimated</text><text x=\"440\" y=\"650\" class=\"legend\">{:.3} frames / {:.4} s</text><text x=\"690\" y=\"650\" class=\"metric\">Correlation</text><text x=\"800\" y=\"650\" class=\"legend\">{:.4} — accepted</text><text x=\"45\" y=\"686\" class=\"metric\">Before RMS</text><text x=\"180\" y=\"686\" class=\"legend\">{:.4}</text><text x=\"340\" y=\"686\" class=\"metric\">After RMS</text><text x=\"440\" y=\"686\" class=\"legend\">{:.4}</text><text x=\"690\" y=\"686\" class=\"metric\">Residual units</text><text x=\"820\" y=\"686\" class=\"legend\">signal amplitude, not seconds</text></svg>",
        data.result.shift_frames(),
        data.result.shift_seconds(),
        data.result.peak_correlation(),
        data.unaligned_rms,
        data.result.residual_rms()
    )?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg).with_context(|| format!("write {}", path.display()))?;
    println!(
        "wrote {} (shift {:.4} frames, correlation {:.5}, RMS {:.5} -> {:.5})",
        path.display(),
        data.result.shift_frames(),
        data.result.peak_correlation(),
        data.unaligned_rms,
        data.result.residual_rms()
    );
    Ok(())
}

fn output_path() -> PathBuf {
    std::env::args_os().nth(1).map_or_else(
        || PathBuf::from("docs/book/figures/temporal_synchronization.svg"),
        PathBuf::from,
    )
}

fn main() -> Result<()> {
    let data = example_data()?;
    write_figure(&output_path(), &data)
}
