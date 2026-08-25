//! Generate the binary morphology figure used by the RITK mdBook.
//!
//! The phantom contains two distinct defect classes: foreground specks and a
//! thin protrusion for opening to remove, plus background holes for closing to
//! fill. The example verifies the anti-extensive and extensive contracts before
//! rendering data-derived change maps.
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use coeus_core::SequentialBackend;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use ritk_filter::{BinaryMorphologicalClosing, BinaryMorphologicalOpening};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

type Backend = SequentialBackend;

const SHAPE: [usize; 3] = [5, 96, 96];
const DISPLAY_SLICE: usize = 2;
const RADIUS: usize = 1;
const PANEL_WIDTH: u32 = 210;
const PANEL_HEIGHT: u32 = 245;
const IMAGE_SIDE: u32 = 176;
const IMAGE_OFFSET_X: u32 = 17;
const IMAGE_OFFSET_Y: u32 = 48;

fn phantom() -> Result<Vec<f32>> {
    let [depth, height, width] = SHAPE;
    let voxel_count = depth
        .checked_mul(height)
        .and_then(|count| count.checked_mul(width))
        .context("binary morphology phantom size overflows usize")?;
    let mut values = vec![0.0; voxel_count];
    for z in 0..depth {
        for y in 20..76 {
            for x in 20..76 {
                values[(z * height + y) * width + x] = 1.0;
            }
        }
        for x in 76..85 {
            values[(z * height + 47) * width + x] = 1.0;
        }
        for y in 8..10 {
            for x in 8..10 {
                values[(z * height + y) * width + x] = 1.0;
            }
        }
        for &(y, x) in &[(36, 36), (36, 37), (37, 36), (60, 55)] {
            values[(z * height + y) * width + x] = 0.0;
        }
    }
    Ok(values)
}

fn middle_slice(values: &[f32]) -> Result<&[f32]> {
    let [depth, height, width] = SHAPE;
    if DISPLAY_SLICE >= depth {
        bail!("display slice {DISPLAY_SLICE} is outside phantom depth {depth}");
    }
    let plane = height
        .checked_mul(width)
        .context("binary morphology plane size overflows usize")?;
    let start = DISPLAY_SLICE
        .checked_mul(plane)
        .context("binary morphology slice offset overflows usize")?;
    values
        .get(start..start + plane)
        .context("binary morphology data length does not match its shape")
}

enum Panel<'a> {
    Mask(&'a [f32]),
    Removed { input: &'a [f32], opened: &'a [f32] },
    Added { input: &'a [f32], closed: &'a [f32] },
}

fn panel_png(panel: Panel<'_>) -> Result<String> {
    let [_, height, width] = SHAPE;
    let pixel_count = height
        .checked_mul(width)
        .context("binary morphology panel size overflows usize")?;
    let mut raster = Vec::with_capacity(
        pixel_count
            .checked_mul(3)
            .context("binary morphology RGB size overflows usize")?,
    );
    for index in 0..pixel_count {
        let color = match panel {
            Panel::Mask(values) => {
                if values
                    .get(index)
                    .copied()
                    .context("mask panel length mismatch")?
                    > 0.5
                {
                    [235, 240, 247]
                } else {
                    [15, 23, 42]
                }
            }
            Panel::Removed { input, opened } => {
                let source = input
                    .get(index)
                    .copied()
                    .context("opening source panel length mismatch")?;
                let result = opened
                    .get(index)
                    .copied()
                    .context("opening result panel length mismatch")?;
                if source > 0.5 && result <= 0.5 {
                    [239, 68, 68]
                } else if result > 0.5 {
                    [203, 213, 225]
                } else {
                    [15, 23, 42]
                }
            }
            Panel::Added { input, closed } => {
                let source = input
                    .get(index)
                    .copied()
                    .context("closing source panel length mismatch")?;
                let result = closed
                    .get(index)
                    .copied()
                    .context("closing result panel length mismatch")?;
                if source <= 0.5 && result > 0.5 {
                    [34, 197, 94]
                } else if result > 0.5 {
                    [203, 213, 225]
                } else {
                    [15, 23, 42]
                }
            }
        };
        raster.extend_from_slice(&color);
    }
    let width = u32::try_from(width).context("binary morphology width exceeds u32")?;
    let height = u32::try_from(height).context("binary morphology height exceeds u32")?;
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&raster, width, height, ColorType::Rgb8)
        .context("encode binary morphology panel as PNG")?;
    Ok(STANDARD.encode(png))
}

struct FigurePanel<'a> {
    title: &'a str,
    subtitle: &'a str,
    content: Panel<'a>,
}

fn draw_panel(svg: &mut String, index: usize, panel: FigurePanel<'_>) -> Result<()> {
    let offset_x = u32::try_from(index)
        .context("binary morphology panel index exceeds u32")?
        .checked_mul(PANEL_WIDTH)
        .context("binary morphology panel offset overflows u32")?;
    let encoded = panel_png(panel.content)?;
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
        "<text x=\"{}\" y=\"38\" text-anchor=\"middle\" class=\"subtitle\">{}</text>",
        PANEL_WIDTH / 2,
        panel.subtitle
    )?;
    writeln!(
        svg,
        "<image x=\"{IMAGE_OFFSET_X}\" y=\"{IMAGE_OFFSET_Y}\" width=\"{IMAGE_SIDE}\" height=\"{IMAGE_SIDE}\" href=\"data:image/png;base64,{encoded}\" image-rendering=\"pixelated\"/>"
    )?;
    svg.push_str("</g>\n");
    Ok(())
}

fn write_figure(
    path: &Path,
    input: &[f32],
    opened: &[f32],
    closed: &[f32],
    removed: usize,
    added: usize,
) -> Result<()> {
    let input = middle_slice(input)?;
    let opened = middle_slice(opened)?;
    let closed = middle_slice(closed)?;
    let figure_width = PANEL_WIDTH
        .checked_mul(5)
        .context("binary morphology figure width overflows u32")?;
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {figure_width} {PANEL_HEIGHT}\">\n<style>.title{{font:600 15px sans-serif;fill:#172033}}.subtitle{{font:11px sans-serif;fill:#475569}}</style>\n"
    );
    for (index, panel) in [
        FigurePanel {
            title: "Input mask",
            subtitle: "specks, spur, and holes",
            content: Panel::Mask(input),
        },
        FigurePanel {
            title: "Opening",
            subtitle: "erode then dilate",
            content: Panel::Mask(opened),
        },
        FigurePanel {
            title: "Opening change",
            subtitle: &format!("{removed} foreground voxels removed"),
            content: Panel::Removed { input, opened },
        },
        FigurePanel {
            title: "Closing",
            subtitle: "dilate then erode",
            content: Panel::Mask(closed),
        },
        FigurePanel {
            title: "Closing change",
            subtitle: &format!("{added} background voxels filled"),
            content: Panel::Added { input, closed },
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
        .with_context(|| format!("write binary morphology figure {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/binary_morphology.svg"));
    let backend = Backend::default();
    let input_values = phantom()?;
    let input = Image::from_flat_on(
        input_values,
        SHAPE,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &backend,
    )
    .context("construct binary morphology phantom")?;
    let opened = BinaryMorphologicalOpening::new(RADIUS)
        .apply_native(&input, &backend)
        .context("apply binary opening")?;
    let closed = BinaryMorphologicalClosing::new(RADIUS)
        .apply_native(&input, &backend)
        .context("apply binary closing")?;
    if opened.shape() != input.shape() || closed.shape() != input.shape() {
        bail!("binary opening or closing changed the phantom geometry");
    }
    let input_values = input.data_slice()?;
    let opened_values = opened.data_slice()?;
    let closed_values = closed.data_slice()?;
    let input_slice = middle_slice(input_values)?;
    let opened_slice = middle_slice(opened_values)?;
    let closed_slice = middle_slice(closed_values)?;
    let mut removed = 0_usize;
    let mut added = 0_usize;
    for ((source, opened), closed) in input_slice.iter().zip(opened_slice).zip(closed_slice) {
        if *opened > *source {
            bail!("binary opening violated anti-extensivity on the displayed interior slice");
        }
        if *closed < *source {
            bail!("binary closing violated extensivity on the displayed interior slice");
        }
        removed += usize::from(*source > 0.5 && *opened <= 0.5);
        added += usize::from(*source <= 0.5 && *closed > 0.5);
    }
    if removed == 0 || added == 0 || opened_slice == closed_slice {
        bail!(
            "binary morphology phantom did not distinguish opening and closing: removed={removed}, added={added}"
        );
    }
    write_figure(
        &output,
        input_values,
        opened_values,
        closed_values,
        removed,
        added,
    )?;
    println!(
        "wrote {} (radius {RADIUS}; opening removed {removed}; closing filled {added})",
        output.display()
    );
    Ok(())
}
