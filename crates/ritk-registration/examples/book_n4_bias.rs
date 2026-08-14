//! Generate the N4 bias-correction figure used by the RITK mdBook.
//!
//! The example loads the RIRE Patient 001 MR volume, applies the native N4
//! bias-field correction filter, and renders an axial slice together with the
//! multiplicative field estimated by the correction. Every displayed value is
//! derived from the dataset or from the filter output.
#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: example/test diagnostic output"
)]

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use coeus_core::SequentialBackend;
use eunomia::CastFrom;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder, Rgb, RgbImage};
use ritk_filter::{bias::N4Config, N4BiasFieldCorrectionFilter};
use ritk_io::{format::metaimage::native::MetaImageReader, ImageReader};
use ritk_spatial::VolumeDims;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

type Backend = SequentialBackend;

const MR_PATH: &str = "test_data/registration/rire/training_001_mr_T1.mha";
const DISPLAY_WIDTH: u32 = 256;
const DISPLAY_HEIGHT: u32 = 256;
const PANEL_WIDTH: u32 = 300;
const PANEL_HEIGHT: u32 = 320;
const PANEL_GAP: u32 = 12;

fn read_input() -> Result<ritk_image::Image<f32, Backend, 3>> {
    MetaImageReader::new(Backend::default())
        .read(MR_PATH)
        .with_context(|| format!("read RIRE MR volume {MR_PATH}"))
}

fn percentile(values: &[f32], hundredths: usize) -> Result<f32> {
    let mut finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        bail!("cannot compute a percentile from an empty finite sample");
    }
    finite.sort_by(f32::total_cmp);
    let index = finite.len().saturating_mul(hundredths) / 100;
    finite
        .get(index.min(finite.len() - 1))
        .copied()
        .context("percentile index is outside the sorted sample")
}

fn slice(values: &[f32], shape: [usize; 3], z: usize) -> Result<&[f32]> {
    let [depth, height, width] = shape;
    if z >= depth {
        bail!("axial slice {z} is outside volume depth {depth}");
    }
    let plane = height
        .checked_mul(width)
        .context("axial plane size overflows usize")?;
    let start = z
        .checked_mul(plane)
        .context("axial slice offset overflows usize")?;
    values
        .get(start..start + plane)
        .context("image data length does not match its declared shape")
}

fn grayscale(values: &[f32], lower: f32, upper: f32) -> Result<Vec<Rgb<u8>>> {
    if !lower.is_finite()
        || !upper.is_finite()
        || !matches!(lower.partial_cmp(&upper), Some(std::cmp::Ordering::Less))
    {
        bail!("figure intensity range must be finite and strictly increasing");
    }
    Ok(values
        .iter()
        .map(|value| {
            let intensity = ((*value - lower) / (upper - lower)).clamp(0.0, 1.0) * 255.0;
            let intensity = u8::cast_from(intensity.round());
            Rgb([intensity, intensity, intensity])
        })
        .collect())
}

fn absolute_percentile(values: &[f32], hundredths: usize) -> Result<f32> {
    let magnitudes: Vec<f32> = values
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value.abs())
        .collect();
    percentile(&magnitudes, hundredths)
}

fn diverging(values: &[f32], extent: f32) -> Result<Vec<Rgb<u8>>> {
    if !extent.is_finite() || extent <= 0.0 {
        bail!("diverging display extent must be finite and positive");
    }
    Ok(values
        .iter()
        .map(|value| {
            if !value.is_finite() {
                return Rgb([16, 16, 16]);
            }
            let position = ((*value / extent).clamp(-1.0, 1.0) + 1.0) * 0.5;
            let channel = if position < 0.5 {
                u8::cast_from((position * 2.0 * 255.0).round())
            } else {
                u8::cast_from(((1.0 - position) * 2.0 * 255.0).round())
            };
            if position < 0.5 {
                Rgb([channel, channel, 255])
            } else {
                Rgb([255, channel, channel])
            }
        })
        .collect())
}

fn render_panel(values: &[Rgb<u8>], shape: [usize; 3]) -> Result<RgbImage> {
    let [_, height, width] = shape;
    let panel_width = usize::try_from(DISPLAY_WIDTH).context("display width exceeds usize")?;
    let panel_height = usize::try_from(DISPLAY_HEIGHT).context("display height exceeds usize")?;
    let mut panel = RgbImage::from_pixel(DISPLAY_WIDTH, DISPLAY_HEIGHT, Rgb([16, 16, 16]));
    for output_y in 0..panel_height {
        let source_y = output_y.saturating_mul(height) / panel_height;
        for output_x in 0..panel_width {
            let source_x = output_x.saturating_mul(width) / panel_width;
            let value = *values
                .get(source_y * width + source_x)
                .context("normalized panel shape mismatch")?;
            panel.put_pixel(
                u32::try_from(output_x).context("output x exceeds u32")?,
                u32::try_from(output_y).context("output y exceeds u32")?,
                value,
            );
        }
    }
    Ok(panel)
}

fn panel_data_uri(panel: &RgbImage) -> Result<String> {
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(
            panel.as_raw(),
            panel.width(),
            panel.height(),
            ColorType::Rgb8,
        )
        .context("encode N4 figure panel as PNG")?;
    Ok(STANDARD.encode(png))
}

fn draw_panel(
    svg: &mut String,
    index: usize,
    title: &str,
    subtitle: &str,
    panel: &RgbImage,
) -> Result<()> {
    let index = u32::try_from(index).context("N4 panel index exceeds u32")?;
    let offset_x = (index % 2)
        .checked_mul(PANEL_WIDTH + PANEL_GAP)
        .context("N4 panel x offset overflows u32")?;
    let offset_y = (index / 2)
        .checked_mul(PANEL_HEIGHT + PANEL_GAP)
        .context("N4 panel y offset overflows u32")?;
    let encoded = panel_data_uri(panel)?;
    writeln!(svg, "<g transform=\"translate({offset_x},{offset_y})\">")?;
    writeln!(
        svg,
        "<rect width=\"{PANEL_WIDTH}\" height=\"{PANEL_HEIGHT}\" rx=\"8\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"23\" text-anchor=\"middle\" class=\"title\">{title}</text>",
        PANEL_WIDTH / 2
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"42\" text-anchor=\"middle\" class=\"subtitle\">{subtitle}</text>",
        PANEL_WIDTH / 2
    )?;
    writeln!(
        svg,
        "<image x=\"22\" y=\"54\" width=\"{DISPLAY_WIDTH}\" height=\"{DISPLAY_HEIGHT}\" href=\"data:image/png;base64,{encoded}\" image-rendering=\"pixelated\"/>"
    )?;
    svg.push_str("</g>\n");
    Ok(())
}

fn write_figure(
    path: &Path,
    input: &[f32],
    corrected: &[f32],
    shape: [usize; 3],
) -> Result<(f32, f32, f32)> {
    let axial_slice = shape[0] / 2;
    let input_slice = slice(input, shape, axial_slice)?;
    let corrected_slice = slice(corrected, shape, axial_slice)?;
    let input_lower = percentile(input_slice, 2)?;
    let input_upper = percentile(input_slice, 98)?;
    let corrected_lower = percentile(corrected_slice, 2)?;
    let corrected_upper = percentile(corrected_slice, 98)?;
    let display_lower = input_lower.min(corrected_lower);
    let display_upper = input_upper.max(corrected_upper);
    let field: Vec<f32> = input_slice
        .iter()
        .zip(corrected_slice.iter())
        .map(|(&source, &estimate)| {
            if source > input_lower
                && source > 0.0
                && source.is_finite()
                && estimate > 0.0
                && estimate.is_finite()
            {
                source / estimate
            } else {
                f32::NAN
            }
        })
        .collect();
    let field_median = percentile(&field, 50)?;
    let field_deviation: Vec<f32> = field
        .iter()
        .map(|value| {
            if value.is_finite() {
                ((*value / field_median) - 1.0) * 100.0
            } else {
                f32::NAN
            }
        })
        .collect();
    let relative_extent = absolute_percentile(&field_deviation, 98)?.max(1.0);
    let field_lower = percentile(&field, 2)?;
    let field_upper = percentile(&field, 98)?;
    let panel_shape = [1, shape[1], shape[2]];
    let input_panel = render_panel(
        &grayscale(input_slice, display_lower, display_upper)?,
        panel_shape,
    )?;
    let corrected_panel = render_panel(
        &grayscale(corrected_slice, display_lower, display_upper)?,
        panel_shape,
    )?;
    let relative_panel = render_panel(&diverging(&field_deviation, relative_extent)?, panel_shape)?;
    let field_panel = render_panel(&grayscale(&field, field_lower, field_upper)?, panel_shape)?;

    let figure_width = PANEL_WIDTH
        .checked_mul(2)
        .and_then(|width| width.checked_add(PANEL_GAP))
        .context("N4 figure width overflows u32")?;
    let figure_height = PANEL_HEIGHT
        .checked_mul(2)
        .and_then(|height| height.checked_add(PANEL_GAP))
        .context("N4 figure height overflows u32")?;
    let shared_window = format!("shared p2/p98 window [{display_lower:.1}, {display_upper:.1}]");
    let deviation_scale = format!("blue/red = ±{relative_extent:.1}% about median field");
    let field_window = format!("field p2/p98 [{field_lower:.3}, {field_upper:.3}]");
    let mut figure = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {figure_width} {figure_height}\">\n<style>.title{{font:600 16px sans-serif;fill:#172033}}.subtitle{{font:12px sans-serif;fill:#475569}}</style>\n"
    );
    draw_panel(&mut figure, 0, "Input MR", &shared_window, &input_panel)?;
    draw_panel(
        &mut figure,
        1,
        "N4-corrected MR",
        &shared_window,
        &corrected_panel,
    )?;
    draw_panel(
        &mut figure,
        2,
        "Amplified bias deviation",
        &deviation_scale,
        &relative_panel,
    )?;
    draw_panel(
        &mut figure,
        3,
        "Estimated multiplicative field",
        &field_window,
        &field_panel,
    )?;
    figure.push_str("</svg>\n");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, figure).with_context(|| format!("write N4 figure {}", path.display()))?;
    Ok((display_lower, display_upper, relative_extent))
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/n4_bias_correction.svg"));
    let input = read_input()?;
    let config = N4Config {
        num_fitting_levels: 2,
        num_iterations: 12,
        convergence_threshold: 0.001,
        num_histogram_bins: 200,
        bias_field_fwhm: 0.15,
        bspline_mesh: VolumeDims::new([1, 1, 1]),
        noise_estimate: 0.01,
        shrink_factor: 4,
    };
    let fitting_levels = config.num_fitting_levels;
    let iterations = config.num_iterations;
    let corrected = N4BiasFieldCorrectionFilter::new(config)
        .apply_native(&input, &Backend::default())
        .context("apply native N4 bias-field correction")?;
    let input_values = input.data_slice()?;
    let corrected_values = corrected.data_slice()?;
    let changed = input_values
        .iter()
        .zip(corrected_values.iter())
        .any(|(&source, &estimate)| {
            source.is_finite() && estimate.is_finite() && (source - estimate).abs() > f32::EPSILON
        });
    if input.shape() != corrected.shape()
        || input_values.len() != corrected_values.len()
        || !changed
        || !corrected_values
            .iter()
            .any(|value| value.is_finite() && *value > 0.0)
    {
        bail!("N4 correction did not preserve geometry or change the positive finite image volume");
    }
    let (display_lower, display_upper, bias_deviation_extent) =
        write_figure(&output, input_values, corrected_values, input.shape())?;
    println!(
        "wrote {} (RIRE MR axial slice {}; shared window [{display_lower:.3}, {display_upper:.3}]; amplified bias deviation ±{bias_deviation_extent:.3}%; N4 levels {}, iterations {})",
        output.display(),
        input.shape()[0] / 2,
        fitting_levels,
        iterations
    );
    Ok(())
}
