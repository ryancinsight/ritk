//! Generate the GrowCut segmentation figure used by the RITK mdBook.
//!
//! The example constructs an analytical two-tissue phantom, supplies sparse
//! foreground/background seeds, executes the public Coeus-native GrowCut API,
//! and verifies the resulting labels against the known circular region.

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use coeus_core::SequentialBackend;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use ritk_image::Image;
use ritk_segmentation::GrowCutFilter;
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const DIMS: [usize; 3] = [1, 96, 96];
const CENTER: [usize; 2] = [48, 48];
const RADIUS: usize = 25;
const BACKGROUND_LABEL: f32 = 1.0;
const FOREGROUND_LABEL: f32 = 2.0;
const MAX_ITERATIONS: usize = 200;
const PANEL_WIDTH: u32 = 240;
const PANEL_HEIGHT: u32 = 276;
const IMAGE_SIDE: u32 = 208;
const IMAGE_OFFSET: u32 = 16;
const IMAGE_TOP: u32 = 48;

struct Phantom {
    intensities: Vec<f32>,
    seeds: Vec<f32>,
    truth: Vec<f32>,
}

fn disk_contains(y: usize, x: usize) -> bool {
    let dy = y.abs_diff(CENTER[0]);
    let dx = x.abs_diff(CENTER[1]);
    dy * dy + dx * dx <= RADIUS * RADIUS
}

fn phantom() -> Result<Phantom> {
    let [_, height, width] = DIMS;
    let voxel_count = height
        .checked_mul(width)
        .context("GrowCut phantom size overflows usize")?;
    let mut intensities = Vec::with_capacity(voxel_count);
    let mut seeds = vec![0.0; voxel_count];
    let mut truth = Vec::with_capacity(voxel_count);

    for y in 0..height {
        for x in 0..width {
            let foreground = disk_contains(y, x);
            intensities.push(if foreground { 0.85 } else { 0.15 });
            truth.push(if foreground {
                FOREGROUND_LABEL
            } else {
                BACKGROUND_LABEL
            });
        }
    }

    for y in 10..13 {
        for x in 10..13 {
            seeds[y * width + x] = BACKGROUND_LABEL;
        }
    }
    for y in CENTER[0] - 1..=CENTER[0] + 1 {
        for x in CENTER[1] - 1..=CENTER[1] + 1 {
            seeds[y * width + x] = FOREGROUND_LABEL;
        }
    }

    Ok(Phantom {
        intensities,
        seeds,
        truth,
    })
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

#[derive(Clone, Copy)]
enum PanelKind {
    Intensity,
    Seeds,
    Labels,
}

fn panel_png(values: &[f32], kind: PanelKind) -> Result<String> {
    let [_, height, width] = DIMS;
    let expected = height
        .checked_mul(width)
        .context("GrowCut panel size overflows usize")?;
    if values.len() != expected {
        bail!(
            "GrowCut panel length {} does not match {}x{}",
            values.len(),
            height,
            width
        );
    }
    let mut raster = Vec::with_capacity(
        expected
            .checked_mul(3)
            .context("GrowCut RGB panel size overflows usize")?,
    );
    for &value in values {
        let color = match kind {
            PanelKind::Intensity => {
                if value > 0.5 {
                    [224, 231, 255]
                } else {
                    [30, 41, 59]
                }
            }
            PanelKind::Seeds => {
                if value == BACKGROUND_LABEL {
                    [249, 115, 22]
                } else if value == FOREGROUND_LABEL {
                    [34, 211, 238]
                } else {
                    [30, 41, 59]
                }
            }
            PanelKind::Labels => {
                if value == BACKGROUND_LABEL {
                    [51, 65, 85]
                } else if value == FOREGROUND_LABEL {
                    [14, 165, 233]
                } else {
                    [239, 68, 68]
                }
            }
        };
        raster.extend_from_slice(&color);
    }
    let width = u32::try_from(width).context("GrowCut panel width exceeds u32")?;
    let height = u32::try_from(height).context("GrowCut panel height exceeds u32")?;
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&raster, width, height, ColorType::Rgb8)
        .context("encode GrowCut panel as PNG")?;
    Ok(STANDARD.encode(png))
}

struct FigurePanel<'a> {
    title: &'a str,
    subtitle: &'a str,
    values: &'a [f32],
    kind: PanelKind,
}

fn draw_panel(svg: &mut String, index: usize, panel: FigurePanel<'_>) -> Result<()> {
    let offset_x = u32::try_from(index)
        .context("GrowCut panel index exceeds u32")?
        .checked_mul(PANEL_WIDTH)
        .context("GrowCut panel offset overflows u32")?;
    let encoded = panel_png(panel.values, panel.kind)?;
    writeln!(svg, "<g transform=\"translate({offset_x},0)\">")?;
    writeln!(
        svg,
        "<rect width=\"{PANEL_WIDTH}\" height=\"{PANEL_HEIGHT}\" fill=\"#ffffff\" stroke=\"#cbd5e1\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"21\" text-anchor=\"middle\" class=\"title\">{}</text>",
        PANEL_WIDTH / 2,
        panel.title
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"39\" text-anchor=\"middle\" class=\"subtitle\">{}</text>",
        PANEL_WIDTH / 2,
        panel.subtitle
    )?;
    writeln!(
        svg,
        "<image x=\"{IMAGE_OFFSET}\" y=\"{IMAGE_TOP}\" width=\"{IMAGE_SIDE}\" height=\"{IMAGE_SIDE}\" href=\"data:image/png;base64,{encoded}\" image-rendering=\"pixelated\"/>"
    )?;
    svg.push_str("</g>\n");
    Ok(())
}

fn dice_and_errors(result: &[f32], truth: &[f32]) -> Result<(f64, usize)> {
    if result.len() != truth.len() {
        bail!(
            "GrowCut result length {} differs from truth length {}",
            result.len(),
            truth.len()
        );
    }
    let mut intersection = 0_usize;
    let mut result_count = 0_usize;
    let mut truth_count = 0_usize;
    let mut errors = 0_usize;
    for (&actual, &expected) in result.iter().zip(truth) {
        let actual_foreground = actual == FOREGROUND_LABEL;
        let expected_foreground = expected == FOREGROUND_LABEL;
        intersection += usize::from(actual_foreground && expected_foreground);
        result_count += usize::from(actual_foreground);
        truth_count += usize::from(expected_foreground);
        errors += usize::from(actual != expected);
    }
    let denominator = result_count + truth_count;
    if denominator == 0 {
        bail!("GrowCut analytical phantom has no foreground voxels");
    }
    let intersection =
        u32::try_from(intersection).context("GrowCut intersection count exceeds u32")?;
    let denominator = u32::try_from(denominator).context("GrowCut Dice denominator exceeds u32")?;
    Ok((
        2.0 * f64::from(intersection) / f64::from(denominator),
        errors,
    ))
}

fn write_figure(
    path: &Path,
    phantom: &Phantom,
    result: &[f32],
    dice: f64,
    errors: usize,
) -> Result<()> {
    let figure_width = PANEL_WIDTH
        .checked_mul(4)
        .context("GrowCut figure width overflows u32")?;
    let metric = format!("Dice = {dice:.3}; errors = {errors}");
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {figure_width} {PANEL_HEIGHT}\">\n<style>.title{{font:600 15px sans-serif;fill:#172033}}.subtitle{{font:11px sans-serif;fill:#475569}}</style>\n"
    );
    for (index, panel) in [
        FigurePanel {
            title: "Input image",
            subtitle: "two-tissue analytical phantom",
            values: &phantom.intensities,
            kind: PanelKind::Intensity,
        },
        FigurePanel {
            title: "Sparse seeds",
            subtitle: "orange background; cyan target",
            values: &phantom.seeds,
            kind: PanelKind::Seeds,
        },
        FigurePanel {
            title: "Known truth",
            subtitle: "circular target region",
            values: &phantom.truth,
            kind: PanelKind::Labels,
        },
        FigurePanel {
            title: "GrowCut result",
            subtitle: &metric,
            values: result,
            kind: PanelKind::Labels,
        },
    ]
    .into_iter()
    .enumerate()
    {
        draw_panel(&mut svg, index, panel)?;
    }
    svg.push_str("</svg>\n");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg)
        .with_context(|| format!("write GrowCut figure {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/growcut.svg"));
    let backend = SequentialBackend;
    let phantom = phantom()?;
    let image = build_image(phantom.intensities.clone(), &backend)?;
    let seeds = build_image(phantom.seeds.clone(), &backend)?;
    let result = GrowCutFilter::new(MAX_ITERATIONS)
        .apply_native(&image, &seeds, &backend)
        .context("apply GrowCut segmentation")?;
    if result.shape() != image.shape()
        || result.origin() != image.origin()
        || result.spacing() != image.spacing()
        || result.direction() != image.direction()
    {
        bail!("GrowCut changed the analytical phantom geometry");
    }
    let result_values = result.data_slice()?;
    let (dice, errors) = dice_and_errors(result_values, &phantom.truth)?;
    if errors != 0 || dice != 1.0 {
        bail!("GrowCut missed the analytical oracle: Dice={dice:.6}, label errors={errors}");
    }
    write_figure(&output, &phantom, result_values, dice, errors)?;
    println!(
        "wrote {} (Dice={dice:.3}; label errors={errors})",
        output.display()
    );
    Ok(())
}
