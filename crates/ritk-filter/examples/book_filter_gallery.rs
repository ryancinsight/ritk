//! Generate the filter-gallery figure used by the RITK mdBook.
//!
//! The input is a deterministic 3-D phantom with one axial slice. The
//! Gaussian and Canny stages execute through their public native image APIs;
//! the SVG writer only renders the resulting values for documentation.

use anyhow::{bail, Context, Result};
use coeus_core::SequentialBackend;
use ritk_filter::{CannyEdgeDetector, GaussianFilter, GaussianSigma};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const DIMS: [usize; 3] = [1, 96, 96];
const CANNY_LOW_THRESHOLD: f64 = 0.01;
const CANNY_HIGH_THRESHOLD: f64 = 0.03;
const PANEL_WIDTH: u32 = 256;
const PANEL_HEIGHT: u32 = 288;

fn phantom() -> Result<Vec<f32>> {
    let [_, height, width] = DIMS;
    let mut values = Vec::with_capacity(
        height
            .checked_mul(width)
            .context("phantom size overflows")?,
    );
    for y in 0..height {
        for x in 0..width {
            let x_index = x;
            let y_index = y;
            let x = f32::from(u16::try_from(x_index).context("x coordinate exceeds u16")?);
            let y = f32::from(u16::try_from(y_index).context("y coordinate exceeds u16")?);
            let first = ((x - 32.0).powi(2) + (y - 38.0).powi(2)) / (2.0 * 11.0_f32.powi(2));
            let second = ((x - 68.0).powi(2) + (y - 62.0).powi(2)) / (2.0 * 8.0_f32.powi(2));
            let disk = if (x - 66.0).abs() < 16.0 && (y - 30.0).abs() < 10.0 {
                0.18
            } else {
                0.0
            };
            let residue = i16::try_from((x_index * 17 + y_index * 29) % 23)
                .context("phantom residue exceeds i16")?;
            let noise = f32::from(residue - 11) / 220.0;
            values.push((0.78 * (-first).exp() + 0.58 * (-second).exp() + disk + noise).max(0.0));
        }
    }
    Ok(values)
}

fn build_image(
    values: Vec<f32>,
    backend: &SequentialBackend,
) -> Result<Image<f32, SequentialBackend, 3>> {
    Image::from_flat_on(
        values,
        DIMS,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        backend,
    )
}

fn normalized(values: &[f32], fixed_range: Option<(f32, f32)>) -> Result<Vec<f32>> {
    let (lower, upper) = fixed_range.unwrap_or_else(|| {
        values.iter().copied().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(lower, upper), value| (lower.min(value), upper.max(value)),
        )
    });
    let range = (upper - lower).max(f32::EPSILON);
    if !lower.is_finite() || !upper.is_finite() {
        anyhow::bail!("figure input contains no finite intensity range")
    }
    Ok(values
        .iter()
        .map(|value| ((*value - lower) / range).clamp(0.0, 1.0))
        .collect())
}

fn svg_panel(svg: &mut String, values: &[f32], title: &str, offset_x: u32) -> Result<()> {
    let [_, height, width] = DIMS;
    let width_u32 = u32::try_from(width).context("panel width exceeds u32")?;
    let height_u32 = u32::try_from(height).context("panel height exceeds u32")?;
    let cell_x = f64::from(PANEL_WIDTH - 32) / f64::from(width_u32);
    let cell_y = f64::from(PANEL_HEIGHT - 48) / f64::from(height_u32);
    writeln!(svg, "<g transform=\"translate({offset_x},0)\">")?;
    writeln!(
        svg,
        "<text x=\"{half}\" y=\"22\" text-anchor=\"middle\" class=\"title\">{title}</text>",
        half = PANEL_WIDTH / 2
    )?;
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let intensity =
                f64::from(values.get(index).copied().context("panel shape mismatch")?) * 255.0;
            let x0 = 16.0 + f64::from(u32::try_from(x).context("x index exceeds u32")?) * cell_x;
            let y0 = 32.0 + f64::from(u32::try_from(y).context("y index exceeds u32")?) * cell_y;
            writeln!(svg, "<rect x=\"{x0:.3}\" y=\"{y0:.3}\" width=\"{cell_x:.3}\" height=\"{cell_y:.3}\" fill=\"rgb({intensity:.0},{intensity:.0},{intensity:.0})\"/>")?;
        }
    }
    svg.push_str("</g>\n");
    Ok(())
}

fn write_figure(path: &Path, input: &[f32], smoothed: &[f32], edges: &[f32]) -> Result<()> {
    let input = normalized(input, None)?;
    let smoothed = normalized(smoothed, None)?;
    let edges = normalized(edges, Some((0.0, 1.0)))?;
    let figure_width = PANEL_WIDTH
        .checked_mul(3)
        .context("figure width overflows")?;
    let mut svg = String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 ");
    writeln!(svg, "{figure_width} {PANEL_HEIGHT}\">\n<style>.title{{font:600 14px sans-serif;fill:#172033}}</style>")?;
    svg_panel(&mut svg, &input, "Input phantom", 0)?;
    svg_panel(&mut svg, &smoothed, "Gaussian σ = 2", PANEL_WIDTH)?;
    svg_panel(&mut svg, &edges, "Canny edges", PANEL_WIDTH * 2)?;
    svg.push_str("</svg>\n");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg).with_context(|| format!("write figure {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/filter_gallery.svg"));
    let backend = SequentialBackend;
    let input_values = phantom()?;
    let input = build_image(input_values.clone(), &backend)?;
    let sigma = GaussianSigma::new(2.0).context("Gaussian sigma must be positive")?;
    let gaussian: GaussianFilter<SequentialBackend> = GaussianFilter::new(vec![sigma]);
    let smoothed = gaussian.apply_native(&input, &backend)?;
    let canny = CannyEdgeDetector::new(sigma, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD);
    let edges = canny.apply_native(&input, &backend)?;
    let edge_values = edges.data_slice()?;
    if !edge_values.iter().any(|&value| value > 0.0) {
        bail!("Canny example produced an empty edge map")
    }
    write_figure(&output, &input_values, smoothed.data_slice()?, edge_values)?;
    println!("wrote {}", output.display());
    Ok(())
}
