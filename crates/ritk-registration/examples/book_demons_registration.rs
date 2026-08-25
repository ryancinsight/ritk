//! Generate the deformable-registration figure used by the RITK mdBook.
//!
//! The example loads an in-tree RIRE MR volume, extracts a bounded
//! same-modality crop, applies a known translation, and registers the shifted
//! volume with classic Thirion Demons. The figure is generated from the actual
//! fixed, moving, warped, and displacement-field values.
#![expect(
    clippy::print_stdout,
    reason = "RITK-LINT-1: example/test diagnostic output"
)]

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use coeus_core::SequentialBackend;
use eunomia::CastFrom;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder, Rgb, RgbImage};
use ritk_io::{format::metaimage::native::MetaImageReader, ImageReader};
use ritk_registration::{DemonsConfig, ThirionDemonsRegistration};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

type Backend = SequentialBackend;

const MR_PATH: &str = "test_data/registration/rire/training_001_mr_T1.mha";
// Six 48×48 slices preserve the anatomical registration signal while keeping
// the dev-profile example below the repository's finite runtime budget.
const CROP_SHAPE: [usize; 3] = [6, 48, 48];
const TRANSLATION_X: usize = 3;
const REGISTRATION_ITERATIONS: usize = 35;
const DISPLAY_SIDE: u32 = 224;
const PANEL_WIDTH: u32 = 260;
const PANEL_HEIGHT: u32 = 300;
const PANEL_GAP: u32 = 12;

fn percentile(values: &[f32], hundredths: usize) -> Result<f32> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
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

fn extract_crop(values: &[f32], source_shape: [usize; 3]) -> Result<Vec<f32>> {
    let [source_depth, source_height, source_width] = source_shape;
    let [crop_depth, crop_height, crop_width] = CROP_SHAPE;
    if source_depth < crop_depth || source_height < crop_height || source_width < crop_width {
        bail!(
            "RIRE MR volume {:?} is smaller than required crop {:?}",
            source_shape,
            CROP_SHAPE
        );
    }

    let stride_y = (source_height / crop_height).max(1);
    let stride_x = (source_width / crop_width).max(1);
    let sampled_height = crop_height
        .checked_mul(stride_y)
        .context("sampled crop height overflows usize")?;
    let sampled_width = crop_width
        .checked_mul(stride_x)
        .context("sampled crop width overflows usize")?;
    if sampled_height > source_height || sampled_width > source_width {
        bail!("computed MR crop exceeds the source geometry");
    }

    let start_z = (source_depth - crop_depth) / 2;
    let start_y = (source_height - sampled_height) / 2;
    let start_x = (source_width - sampled_width) / 2;
    let source_plane = source_height
        .checked_mul(source_width)
        .context("source MR plane size overflows usize")?;
    let crop_len = crop_depth
        .checked_mul(crop_height)
        .and_then(|count| count.checked_mul(crop_width))
        .context("MR crop size overflows usize")?;
    let mut crop = Vec::with_capacity(crop_len);

    for z in 0..crop_depth {
        for y in 0..crop_height {
            for x in 0..crop_width {
                let source_z = start_z + z;
                let source_y = start_y + y * stride_y;
                let source_x = start_x + x * stride_x;
                let index = source_z
                    .checked_mul(source_plane)
                    .and_then(|offset| {
                        source_y
                            .checked_mul(source_width)
                            .and_then(|row| offset.checked_add(row))
                    })
                    .and_then(|offset| offset.checked_add(source_x))
                    .context("MR crop source index overflows usize")?;
                crop.push(
                    *values
                        .get(index)
                        .context("MR payload does not match its declared shape")?,
                );
            }
        }
    }

    let lower = percentile(&crop, 2)?;
    let upper = percentile(&crop, 98)?;
    if !matches!(lower.partial_cmp(&upper), Some(std::cmp::Ordering::Less)) {
        bail!("RIRE MR crop has no usable finite intensity range");
    }
    for value in &mut crop {
        *value = ((*value - lower) / (upper - lower)).clamp(0.0, 1.0);
    }
    Ok(crop)
}

fn translate_x(values: &[f32], shift: usize) -> Result<Vec<f32>> {
    let [depth, height, width] = CROP_SHAPE;
    if shift >= width {
        bail!("translation {shift} must be smaller than crop width {width}");
    }
    let expected_len = depth
        .checked_mul(height)
        .and_then(|count| count.checked_mul(width))
        .context("translated volume size overflows usize")?;
    if values.len() != expected_len {
        bail!(
            "translation input length {} does not match crop size {expected_len}",
            values.len()
        );
    }
    let mut translated = vec![0.0; expected_len];
    for z in 0..depth {
        for y in 0..height {
            let row = z * height * width + y * width;
            let source = values
                .get(row..row + width - shift)
                .context("source translation row is outside the volume")?;
            let destination = translated
                .get_mut(row + shift..row + width)
                .context("destination translation row is outside the volume")?;
            destination.copy_from_slice(source);
        }
    }
    Ok(translated)
}

fn mean_squared_error(left: &[f32], right: &[f32]) -> Result<f64> {
    if left.len() != right.len() || left.is_empty() {
        bail!(
            "MSE requires equal non-empty inputs, got {} and {}",
            left.len(),
            right.len()
        );
    }
    let sum = left
        .iter()
        .zip(right)
        .map(|(&left, &right)| {
            let difference = f64::from(left) - f64::from(right);
            difference * difference
        })
        .sum::<f64>();
    let count = u32::try_from(left.len()).context("MSE sample count exceeds u32")?;
    Ok(sum / f64::from(count))
}

fn axial_slice(values: &[f32]) -> Result<&[f32]> {
    let [depth, height, width] = CROP_SHAPE;
    let plane = height
        .checked_mul(width)
        .context("display plane size overflows usize")?;
    let start = (depth / 2)
        .checked_mul(plane)
        .context("display slice offset overflows usize")?;
    values
        .get(start..start + plane)
        .context("display volume does not match the crop shape")
}

enum Panel<'a> {
    Grayscale(&'a [f32]),
    Overlay { fixed: &'a [f32], moving: &'a [f32] },
    Diverging { values: &'a [f32], extent: f32 },
}

fn channel(value: f32) -> u8 {
    u8::cast_from((value.clamp(0.0, 1.0) * 255.0).round())
}

fn render_panel(panel: Panel<'_>) -> Result<RgbImage> {
    let [_, height, width] = CROP_SHAPE;
    let display_side = usize::try_from(DISPLAY_SIDE).context("display side exceeds usize")?;
    let mut image = RgbImage::from_pixel(DISPLAY_SIDE, DISPLAY_SIDE, Rgb([16, 16, 16]));
    for output_y in 0..display_side {
        let source_y = output_y * height / display_side;
        for output_x in 0..display_side {
            let source_x = output_x * width / display_side;
            let index = source_y * width + source_x;
            let pixel = match &panel {
                Panel::Grayscale(values) => {
                    let value = *values
                        .get(index)
                        .context("grayscale panel length does not match its shape")?;
                    let value = channel(value);
                    Rgb([value, value, value])
                }
                Panel::Overlay { fixed, moving } => {
                    let fixed = channel(
                        *fixed
                            .get(index)
                            .context("fixed overlay length does not match its shape")?,
                    );
                    let moving = channel(
                        *moving
                            .get(index)
                            .context("moving overlay length does not match its shape")?,
                    );
                    Rgb([fixed, moving, 0])
                }
                Panel::Diverging { values, extent } => {
                    let value = *values
                        .get(index)
                        .context("displacement panel length does not match its shape")?;
                    let position = (value / *extent).clamp(-1.0, 1.0);
                    if position < 0.0 {
                        let fade = channel(1.0 + position);
                        Rgb([fade, fade, 255])
                    } else {
                        let fade = channel(1.0 - position);
                        Rgb([255, fade, fade])
                    }
                }
            };
            image.put_pixel(
                u32::try_from(output_x).context("display x exceeds u32")?,
                u32::try_from(output_y).context("display y exceeds u32")?,
                pixel,
            );
        }
    }
    Ok(image)
}

fn png_data_uri(image: &RgbImage) -> Result<String> {
    let mut encoded = Vec::new();
    PngEncoder::new(&mut encoded)
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ColorType::Rgb8,
        )
        .context("encode deformable-registration panel as PNG")?;
    Ok(format!(
        "data:image/png;base64,{}",
        STANDARD.encode(encoded)
    ))
}

struct FigurePanel<'a> {
    title: &'a str,
    subtitle: &'a str,
    raster: RgbImage,
}

fn write_figure(
    path: &Path,
    fixed: &[f32],
    moving: &[f32],
    warped: &[f32],
    displacement_x: &[f32],
    initial_mse: f64,
    final_mse: f64,
) -> Result<f32> {
    let fixed_slice = axial_slice(fixed)?;
    let moving_slice = axial_slice(moving)?;
    let warped_slice = axial_slice(warped)?;
    let displacement_slice = axial_slice(displacement_x)?;
    let displacement_extent = displacement_slice
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(f32::abs)
        .max_by(f32::total_cmp)
        .context("displacement slice contains no finite values")?
        .max(f32::EPSILON);
    let panels = [
        FigurePanel {
            title: "Fixed RIRE MR",
            subtitle: "normalized same-modality target",
            raster: render_panel(Panel::Grayscale(fixed_slice))?,
        },
        FigurePanel {
            title: "Before registration",
            subtitle: "red=fixed, green=translated moving",
            raster: render_panel(Panel::Overlay {
                fixed: fixed_slice,
                moving: moving_slice,
            })?,
        },
        FigurePanel {
            title: "After Thirion Demons",
            subtitle: "red=fixed, green=warped moving",
            raster: render_panel(Panel::Overlay {
                fixed: fixed_slice,
                moving: warped_slice,
            })?,
        },
        FigurePanel {
            title: "Recovered x displacement",
            subtitle: "blue=negative, white=zero, red=positive",
            raster: render_panel(Panel::Diverging {
                values: displacement_slice,
                extent: displacement_extent,
            })?,
        },
    ];

    let width = PANEL_WIDTH
        .checked_mul(u32::try_from(panels.len()).context("panel count exceeds u32")?)
        .and_then(|value| value.checked_add(PANEL_GAP * (u32::try_from(panels.len()).ok()? - 1)))
        .context("figure width overflows u32")?;
    let height = PANEL_HEIGHT + 72;
    let mut svg = String::new();
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"#
    )?;
    writeln!(
        svg,
        r##"<rect width="100%" height="100%" fill="#111827"/>"##
    )?;
    writeln!(
        svg,
        r##"<text x="20" y="28" fill="#f9fafb" font-family="sans-serif" font-size="18" font-weight="700">Same-modality deformable registration with a known {TRANSLATION_X}-voxel translation</text>"##
    )?;
    writeln!(
        svg,
        r##"<text x="20" y="52" fill="#d1d5db" font-family="sans-serif" font-size="14">MSE {initial_mse:.6} → {final_mse:.6}; displacement display ±{displacement_extent:.3} voxels</text>"##
    )?;

    for (index, panel) in panels.iter().enumerate() {
        let index = u32::try_from(index).context("panel index exceeds u32")?;
        let x = index * (PANEL_WIDTH + PANEL_GAP);
        let data_uri = png_data_uri(&panel.raster)?;
        writeln!(
            svg,
            r##"<rect x="{x}" y="70" width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" rx="8" fill="#1f2937"/>"##
        )?;
        writeln!(
            svg,
            r##"<text x="{}" y="96" fill="#f9fafb" font-family="sans-serif" font-size="15" font-weight="700">{}</text>"##,
            x + 14,
            panel.title
        )?;
        writeln!(
            svg,
            r##"<text x="{}" y="117" fill="#9ca3af" font-family="sans-serif" font-size="11">{}</text>"##,
            x + 14,
            panel.subtitle
        )?;
        writeln!(
            svg,
            r#"<image x="{}" y="132" width="{DISPLAY_SIDE}" height="{DISPLAY_SIDE}" href="{data_uri}"/>"#,
            x + (PANEL_WIDTH - DISPLAY_SIDE) / 2
        )?;
    }
    writeln!(svg, "</svg>")?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg)
        .with_context(|| format!("write deformable-registration figure {}", path.display()))?;
    Ok(displacement_extent)
}

fn output_path() -> PathBuf {
    std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("demons_registration.svg"), PathBuf::from)
}

fn main() -> Result<()> {
    let output = output_path();
    let image = MetaImageReader::new(Backend::default())
        .read(MR_PATH)
        .with_context(|| format!("read RIRE MR volume {MR_PATH}"))?;
    let fixed = extract_crop(image.data_slice()?, image.shape())?;
    let moving = translate_x(&fixed, TRANSLATION_X)?;
    let initial_mse = mean_squared_error(&fixed, &moving)?;
    let result = ThirionDemonsRegistration::new(DemonsConfig {
        max_iterations: REGISTRATION_ITERATIONS,
        ..DemonsConfig::default()
    })
    .register(&fixed, &moving, CROP_SHAPE, [1.0, 1.0, 1.0])
    .context("register translated RIRE MR crop with Thirion Demons")?;
    let measured_final_mse = mean_squared_error(&fixed, &result.warped)?;
    if !result.final_mse.is_finite()
        || !measured_final_mse.is_finite()
        || measured_final_mse >= initial_mse
        || result
            .disp_x
            .iter()
            .chain(&result.disp_y)
            .chain(&result.disp_z)
            .any(|value| !value.is_finite())
    {
        bail!(
            "Demons validation failed: initial MSE {initial_mse:.6}, reported final MSE {:.6}, measured final MSE {measured_final_mse:.6}",
            result.final_mse
        );
    }
    let displacement_extent = write_figure(
        &output,
        &fixed,
        &moving,
        &result.warped,
        &result.disp_x,
        initial_mse,
        measured_final_mse,
    )?;
    println!(
        "wrote {} (RIRE MR crop {:?}; translation {} voxels; MSE {:.6} -> {:.6}; {} iterations; display displacement ±{displacement_extent:.3} voxels)",
        output.display(),
        CROP_SHAPE,
        TRANSLATION_X,
        initial_mse,
        measured_final_mse,
        result.num_iterations
    );
    Ok(())
}
