//! Generate the N4 bias-correction figure used by the RITK mdBook.
//!
//! The example loads the RIRE Patient 001 MR volume, applies the native N4
//! bias-field correction filter, and renders an axial slice together with the
//! multiplicative field estimated by the correction. Every displayed value is
//! derived from the dataset or from the filter output.

use anyhow::{bail, Context, Result};
use coeus_core::SequentialBackend;
use eunomia::CastFrom;
use image::{Rgb, RgbImage};
use ritk_filter::{bias::N4Config, N4BiasFieldCorrectionFilter};
use ritk_io::{format::metaimage::native::MetaImageReader, ImageReader};
use ritk_spatial::VolumeDims;
use std::path::{Path, PathBuf};

type Backend = SequentialBackend;

const MR_PATH: &str = "test_data/registration/rire/training_001_mr_T1.mha";
const DISPLAY_WIDTH: u32 = 256;
const DISPLAY_HEIGHT: u32 = 256;
const PANEL_GAP: u32 = 8;

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

fn normalized(values: &[f32], lower: f32, upper: f32) -> Result<Vec<u8>> {
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
            u8::cast_from(intensity.round())
        })
        .collect())
}

fn render_panel(
    figure: &mut RgbImage,
    panel_index: usize,
    values: &[u8],
    shape: [usize; 3],
) -> Result<()> {
    let [_, height, width] = shape;
    let panel_width = usize::try_from(DISPLAY_WIDTH).context("display width exceeds usize")?;
    let panel_height = usize::try_from(DISPLAY_HEIGHT).context("display height exceeds usize")?;
    let panel_offset = u32::try_from(panel_index)
        .context("panel index exceeds u32")?
        .checked_mul(DISPLAY_WIDTH + PANEL_GAP)
        .context("figure width overflows u32")?;
    for output_y in 0..panel_height {
        let source_y = output_y.saturating_mul(height) / panel_height;
        for output_x in 0..panel_width {
            let source_x = output_x.saturating_mul(width) / panel_width;
            let value = *values
                .get(source_y * width + source_x)
                .context("normalized panel shape mismatch")?;
            figure.put_pixel(
                panel_offset + u32::try_from(output_x).context("output x exceeds u32")?,
                u32::try_from(output_y).context("output y exceeds u32")?,
                Rgb([value, value, value]),
            );
        }
    }
    Ok(())
}

fn write_figure(path: &Path, input: &[f32], corrected: &[f32], shape: [usize; 3]) -> Result<()> {
    let axial_slice = shape[0] / 2;
    let input_slice = slice(input, shape, axial_slice)?;
    let corrected_slice = slice(corrected, shape, axial_slice)?;
    let input_lower = percentile(input_slice, 2)?;
    let input_upper = percentile(input_slice, 98)?;
    let corrected_lower = percentile(corrected_slice, 2)?;
    let corrected_upper = percentile(corrected_slice, 98)?;
    let field: Vec<f32> = input
        .iter()
        .zip(corrected.iter())
        .map(|(&source, &estimate)| {
            if estimate > 0.0 && estimate.is_finite() {
                source / estimate
            } else {
                f32::NAN
            }
        })
        .collect();
    let field_slice = slice(&field, shape, axial_slice)?;
    let field_lower = percentile(field_slice, 2)?;
    let field_upper = percentile(field_slice, 98)?;
    let input_panel = normalized(input_slice, input_lower, input_upper)?;
    let corrected_panel = normalized(corrected_slice, corrected_lower, corrected_upper)?;
    let field_panel = normalized(field_slice, field_lower, field_upper)?;

    let figure_width = DISPLAY_WIDTH
        .checked_mul(3)
        .and_then(|width| width.checked_add(PANEL_GAP * 2))
        .context("N4 figure width overflows u32")?;
    let mut figure = RgbImage::from_pixel(figure_width, DISPLAY_HEIGHT, Rgb([16, 16, 16]));
    let panel_shape = [1, shape[1], shape[2]];
    render_panel(&mut figure, 0, &input_panel, panel_shape)?;
    render_panel(&mut figure, 1, &corrected_panel, panel_shape)?;
    render_panel(&mut figure, 2, &field_panel, panel_shape)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    figure
        .save(path)
        .with_context(|| format!("write N4 figure {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/n4_bias_correction.png"));
    let input = read_input()?;
    let input_values = input.data_slice()?.to_vec();
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
    let corrected = N4BiasFieldCorrectionFilter::new(config)
        .apply_native(&input, &Backend::default())
        .context("apply native N4 bias-field correction")?;
    let corrected_values = corrected.data_slice()?.to_vec();
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
    write_figure(&output, &input_values, &corrected_values, input.shape())?;
    println!(
        "wrote {} (RIRE MR axial slice {}; N4 levels {}, iterations {})",
        output.display(),
        input.shape()[0] / 2,
        2,
        12
    );
    Ok(())
}
