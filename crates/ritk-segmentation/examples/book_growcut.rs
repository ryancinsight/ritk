//! Generate the GrowCut segmentation figure used by the RITK mdBook.
//!
//! The example constructs an analytical two-tissue phantom, supplies sparse
//! foreground/background seeds, executes the public Coeus-native GrowCut API,
//! and verifies the resulting labels against the known circular region.
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

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
const BACKGROUND_INTENSITY: f32 = 0.15;
const FOREGROUND_INTENSITY: f32 = 0.85;
const BACKGROUND_LABEL: f32 = 1.0;
const FOREGROUND_LABEL: f32 = 2.0;
const EARLY_ITERATIONS: usize = 8;
const MIDDLE_ITERATIONS: usize = 40;
const MAX_ITERATIONS: usize = 200;
const PANEL_WIDTH: u32 = 240;
const PANEL_HEIGHT: u32 = 286;
const EXPLANATION_HEIGHT: u32 = 170;
const IMAGE_SIDE: u32 = 208;
const IMAGE_OFFSET: u32 = 16;
const IMAGE_TOP: u32 = 58;

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
            intensities.push(if foreground {
                FOREGROUND_INTENSITY
            } else {
                BACKGROUND_INTENSITY
            });
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

fn panel_png(intensities: &[f32], labels: &[f32]) -> Result<String> {
    let [_, height, width] = DIMS;
    let expected = height
        .checked_mul(width)
        .context("GrowCut panel size overflows usize")?;
    if intensities.len() != expected || labels.len() != expected {
        bail!(
            "GrowCut panel lengths ({}, {}) do not match {}x{}",
            intensities.len(),
            labels.len(),
            height,
            width
        );
    }
    let mut raster = Vec::with_capacity(
        expected
            .checked_mul(3)
            .context("GrowCut RGB panel size overflows usize")?,
    );
    for (&intensity, &label) in intensities.iter().zip(labels) {
        let color = if label == BACKGROUND_LABEL {
            [249, 115, 22]
        } else if label == FOREGROUND_LABEL {
            [34, 211, 238]
        } else if intensity > 0.5 {
            [224, 231, 255]
        } else {
            [30, 41, 59]
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
    intensities: &'a [f32],
    labels: &'a [f32],
}

fn draw_panel(svg: &mut String, index: usize, panel: FigurePanel<'_>) -> Result<()> {
    let offset_x = u32::try_from(index)
        .context("GrowCut panel index exceeds u32")?
        .checked_mul(PANEL_WIDTH)
        .context("GrowCut panel offset overflows u32")?;
    let encoded = panel_png(panel.intensities, panel.labels)?;
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

fn labeled_count(labels: &[f32]) -> usize {
    labels.iter().filter(|&&label| label > 0.0).count()
}

fn labeled_errors(labels: &[f32], truth: &[f32]) -> Result<usize> {
    if labels.len() != truth.len() {
        bail!(
            "GrowCut snapshot length {} differs from truth length {}",
            labels.len(),
            truth.len()
        );
    }
    Ok(labels
        .iter()
        .zip(truth)
        .filter(|&(&label, &expected)| label > 0.0 && label != expected)
        .count())
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
    early: &[f32],
    middle: &[f32],
    result: &[f32],
    dice: f64,
    errors: usize,
) -> Result<()> {
    let figure_width = PANEL_WIDTH
        .checked_mul(4)
        .context("GrowCut figure width overflows u32")?;
    let figure_height = PANEL_HEIGHT
        .checked_add(EXPLANATION_HEIGHT)
        .context("GrowCut figure height overflows u32")?;
    let voxel_count = phantom.seeds.len();
    let early_fraction = 100.0 * labeled_count(early) as f64 / voxel_count as f64;
    let middle_fraction = 100.0 * labeled_count(middle) as f64 / voxel_count as f64;
    let early_summary = format!("{early_fraction:.1}% labeled; grayscale is undecided");
    let middle_summary = format!("{middle_fraction:.1}% labeled; fronts meet the edge");
    let metric = format!("Dice = {dice:.3}; errors = {errors}");
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {figure_width} {figure_height}\">\n<style>.title{{font:600 15px sans-serif;fill:#172033}}.subtitle{{font:11px sans-serif;fill:#475569}}.explanation-title{{font:600 16px sans-serif;fill:#172033}}.explanation-heading{{font:600 13px sans-serif;fill:#172033}}.explanation{{font:12px sans-serif;fill:#334155}}.footer{{font:600 12px sans-serif;fill:#172033}}</style>\n"
    );
    for (index, panel) in [
        FigurePanel {
            title: "1. Plant sparse seeds",
            subtitle: "orange = background; cyan = target",
            intensities: &phantom.intensities,
            labels: &phantom.seeds,
        },
        FigurePanel {
            title: "2. Grow for 8 sweeps",
            subtitle: &early_summary,
            intensities: &phantom.intensities,
            labels: early,
        },
        FigurePanel {
            title: "3. Compete at the edge",
            subtitle: &middle_summary,
            intensities: &phantom.intensities,
            labels: middle,
        },
        FigurePanel {
            title: "4. Stop when stable",
            subtitle: &metric,
            intensities: &phantom.intensities,
            labels: result,
        },
    ]
    .into_iter()
    .enumerate()
    {
        draw_panel(&mut svg, index, panel)?;
    }
    let explanation_top = PANEL_HEIGHT;
    let title_line = explanation_top + 30;
    let box_top = explanation_top + 44;
    let footer_line = explanation_top + 154;
    writeln!(
        svg,
        "<rect y=\"{explanation_top}\" width=\"{figure_width}\" height=\"{EXPLANATION_HEIGHT}\" fill=\"#f8fafc\" stroke=\"#cbd5e1\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"20\" y=\"{title_line}\" class=\"explanation-title\">Why the colored fronts stop at the circular tissue edge</text>"
    )?;
    writeln!(
        svg,
        "<rect x=\"20\" y=\"{box_top}\" width=\"440\" height=\"90\" rx=\"8\" fill=\"#ecfeff\" stroke=\"#67e8f9\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"36\" y=\"{}\" class=\"explanation-heading\">Within one tissue: the neighbor can win</text>",
        box_top + 23
    )?;
    writeln!(
        svg,
        "<text x=\"36\" y=\"{}\" class=\"explanation\">|0.85 - 0.85| = 0.00, so g = 1 - 0.00 / 0.70 = 1.00</text>",
        box_top + 47
    )?;
    writeln!(
        svg,
        "<text x=\"36\" y=\"{}\" class=\"explanation\">Attack = neighbor confidence x 1.00: its label can spread.</text>",
        box_top + 70
    )?;
    writeln!(
        svg,
        "<rect x=\"500\" y=\"{box_top}\" width=\"440\" height=\"90\" rx=\"8\" fill=\"#fff7ed\" stroke=\"#fdba74\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"516\" y=\"{}\" class=\"explanation-heading\">Across the tissue edge: the attack is blocked</text>",
        box_top + 23
    )?;
    writeln!(
        svg,
        "<text x=\"516\" y=\"{}\" class=\"explanation\">|0.85 - 0.15| = 0.70, so g = 1 - 0.70 / 0.70 = 0.00</text>",
        box_top + 47
    )?;
    writeln!(
        svg,
        "<text x=\"516\" y=\"{}\" class=\"explanation\">Attack = neighbor confidence x 0.00 = 0: it cannot cross.</text>",
        box_top + 70
    )?;
    writeln!(
        svg,
        "<text x=\"20\" y=\"{footer_line}\" class=\"footer\">Each synchronous sweep lets orange and cyan attack adjacent voxels; every voxel keeps the strongest successful attack.</text>"
    )?;
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
    let early = GrowCutFilter::new(EARLY_ITERATIONS)
        .apply_native(&image, &seeds, &backend)
        .context("apply early GrowCut snapshot")?;
    let middle = GrowCutFilter::new(MIDDLE_ITERATIONS)
        .apply_native(&image, &seeds, &backend)
        .context("apply middle GrowCut snapshot")?;
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
    let early_values = early.data_slice()?;
    let middle_values = middle.data_slice()?;
    let result_values = result.data_slice()?;
    let seed_count = labeled_count(&phantom.seeds);
    let early_count = labeled_count(early_values);
    let middle_count = labeled_count(middle_values);
    let final_count = labeled_count(result_values);
    if !(seed_count < early_count && early_count < middle_count && middle_count < final_count) {
        bail!(
            "GrowCut snapshots do not show strict propagation: seeds={seed_count}, \
             iteration {EARLY_ITERATIONS}={early_count}, iteration {MIDDLE_ITERATIONS}={middle_count}, \
             final={final_count}"
        );
    }
    let early_errors = labeled_errors(early_values, &phantom.truth)?;
    let middle_errors = labeled_errors(middle_values, &phantom.truth)?;
    if early_errors != 0 || middle_errors != 0 {
        bail!(
            "GrowCut crossed the zero-strength tissue boundary before convergence: \
             iteration {EARLY_ITERATIONS} errors={early_errors}, \
             iteration {MIDDLE_ITERATIONS} errors={middle_errors}"
        );
    }
    let (dice, errors) = dice_and_errors(result_values, &phantom.truth)?;
    if errors != 0 || dice != 1.0 {
        bail!("GrowCut missed the analytical oracle: Dice={dice:.6}, label errors={errors}");
    }
    write_figure(
        &output,
        &phantom,
        early_values,
        middle_values,
        result_values,
        dice,
        errors,
    )?;
    println!(
        "wrote {} (Dice={dice:.3}; label errors={errors})",
        output.display()
    );
    Ok(())
}
