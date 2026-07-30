//! Generate the native JPEG-LS figure used by the RITK mdBook.
//!
//! The example encodes one deterministic 12-bit medical-style phantom through
//! RITK's public lossless and near-lossless paths, decodes both codestreams,
//! verifies their exact analytical error contracts, and renders source,
//! reconstructions, and a magnified near-lossless error panel.

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use ritk_codecs::jpeg_ls::encoder::{encode_grayscale_jpeg_ls, JpegLsEncodeError};
use ritk_codecs::{decode_jpeg_ls_fragment, PixelLayout, PixelSignedness};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const WIDTH: usize = 96;
const HEIGHT: usize = 96;
const PRECISION: u32 = 12;
const NEAR: u32 = 3;
const PANEL_WIDTH: u32 = 260;
const PANEL_HEIGHT: u32 = 280;
const IMAGE_SIZE: u32 = 190;

fn phantom() -> Result<Vec<u16>> {
    let sample_count = WIDTH
        .checked_mul(HEIGHT)
        .context("phantom dimensions overflow usize")?;
    let mut samples = Vec::with_capacity(sample_count);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let x_coordinate = i32::try_from(x).context("x coordinate exceeds i32")?;
            let y_coordinate = i32::try_from(y).context("y coordinate exceeds i32")?;
            let outer_dx = x_coordinate - 48;
            let outer_dy = y_coordinate - 50;
            let outer = outer_dx * outer_dx + outer_dy * outer_dy <= 39 * 39;
            let inner = outer_dx * outer_dx + outer_dy * outer_dy <= 25 * 25;
            let lesion_dx = x_coordinate - 63;
            let lesion_dy = y_coordinate - 37;
            let lesion = lesion_dx * lesion_dx + lesion_dy * lesion_dy <= 8 * 8;
            let vessel = (x_coordinate - 30).abs() <= 2 && (23..=74).contains(&y_coordinate);
            let texture = (3 * x_coordinate + 5 * y_coordinate) % 17;
            let background = 90 + 2 * x_coordinate + y_coordinate + texture;
            let tissue = if inner {
                2450 + 4 * x_coordinate - 2 * y_coordinate + texture
            } else if outer {
                1080 + 2 * x_coordinate + y_coordinate + texture
            } else {
                background
            };
            let value = if lesion {
                3610 + texture
            } else if vessel && outer {
                3190 + texture
            } else {
                tissue
            };
            samples.push(
                u16::try_from(value.clamp(0, 4095))
                    .expect("invariant: clamped phantom sample fits u16"),
            );
        }
    }
    Ok(samples)
}

fn layout() -> PixelLayout {
    PixelLayout {
        rows: HEIGHT,
        cols: WIDTH,
        samples_per_pixel: 1,
        bits_allocated: u16::try_from(PRECISION).expect("invariant: example precision fits u16"),
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    }
}

fn encode_decode(source: &[u16], near: u32) -> Result<(Vec<u8>, Vec<f32>)> {
    let stream = encode_grayscale_jpeg_ls(
        source,
        u32::try_from(HEIGHT).context("height exceeds u32")?,
        u32::try_from(WIDTH).context("width exceeds u32")?,
        PRECISION,
        near,
    )
    .context("encode native JPEG-LS stream")?;
    let reconstruction =
        decode_jpeg_ls_fragment(&stream, layout()).context("decode native JPEG-LS stream")?;
    Ok((stream, reconstruction))
}

fn to_png(values: &[f32], lower: f32, upper: f32) -> Result<String> {
    if values.len() != WIDTH * HEIGHT {
        bail!(
            "panel sample count mismatch: got {}, expected {}",
            values.len(),
            WIDTH * HEIGHT
        );
    }
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        bail!("invalid panel display range [{lower}, {upper}]");
    }

    let display_size = usize::try_from(IMAGE_SIZE).context("display size exceeds usize")?;
    let raster_capacity = display_size
        .checked_mul(display_size)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("panel raster size overflows usize")?;
    let mut raster = Vec::with_capacity(raster_capacity);
    for output_y in 0..display_size {
        let source_y = output_y * HEIGHT / display_size;
        for output_x in 0..display_size {
            let source_x = output_x * WIDTH / display_size;
            let value = *values
                .get(source_y * WIDTH + source_x)
                .context("panel index exceeds source geometry")?;
            let normalized = ((value - lower) / (upper - lower)).clamp(0.0, 1.0);
            // The display mapping is clamped to the complete u8 range before
            // conversion; this is a raster boundary, not codec arithmetic.
            let gray = (normalized * 255.0).round() as u8;
            raster.extend_from_slice(&[gray, gray, gray]);
        }
    }

    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&raster, IMAGE_SIZE, IMAGE_SIZE, ColorType::Rgb8)
        .context("encode panel as PNG")?;
    Ok(STANDARD.encode(png))
}

struct Panel<'a> {
    values: &'a [f32],
    title: &'a str,
    subtitle: &'a str,
    range: (f32, f32),
    column: u32,
}

fn draw_panel(svg: &mut String, panel: Panel<'_>) -> Result<()> {
    let encoded = to_png(panel.values, panel.range.0, panel.range.1)?;
    let offset_x = panel
        .column
        .checked_mul(PANEL_WIDTH)
        .context("panel x offset overflows")?;
    writeln!(svg, "<g transform=\"translate({offset_x},0)\">")?;
    writeln!(
        svg,
        "<rect x=\"8\" y=\"8\" width=\"244\" height=\"264\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"31\" class=\"title\">{}</text>",
        panel.title
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"48\" class=\"subtitle\">{}</text>",
        panel.subtitle
    )?;
    writeln!(
        svg,
        "<image x=\"35\" y=\"62\" width=\"{IMAGE_SIZE}\" height=\"{IMAGE_SIZE}\" href=\"data:image/png;base64,{encoded}\" image-rendering=\"pixelated\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"268\" class=\"note\">display [{:.0}, {:.0}]</text>",
        panel.range.0, panel.range.1
    )?;
    writeln!(svg, "</g>")?;
    Ok(())
}

fn write_figure(
    path: &Path,
    source: &[u16],
    lossless: &[f32],
    near_lossless: &[f32],
    lossless_bytes: usize,
    near_lossless_bytes: usize,
) -> Result<()> {
    let source_values: Vec<f32> = source.iter().map(|&value| f32::from(value)).collect();
    let lossless_mismatches = source_values
        .iter()
        .zip(lossless)
        .filter(|(expected, actual)| expected != actual)
        .count();
    if lossless_mismatches != 0 {
        bail!("lossless reconstruction has {lossless_mismatches} mismatched samples");
    }

    let absolute_error: Vec<f32> = source_values
        .iter()
        .zip(near_lossless)
        .map(|(&expected, &actual)| (actual - expected).abs())
        .collect();
    let maximum_error = absolute_error.iter().copied().fold(0.0f32, f32::max);
    let near_bound =
        f32::from(u16::try_from(NEAR).expect("invariant: the example NEAR value fits in a u16"));
    if maximum_error > near_bound {
        bail!("near-lossless maximum error {maximum_error} exceeds NEAR={NEAR}");
    }
    let changed_samples = absolute_error.iter().filter(|&&error| error > 0.0).count();
    if changed_samples == 0 {
        bail!("near-lossless reconstruction produced no visible differences");
    }

    let figure_width = PANEL_WIDTH * 4;
    let figure_height = PANEL_HEIGHT + 82;
    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {figure_width} {figure_height}\">"
    )?;
    writeln!(
        svg,
        "<rect width=\"{figure_width}\" height=\"{figure_height}\" fill=\"#f8fafc\"/>"
    )?;
    svg.push_str(
        "<style>.title{font:600 15px sans-serif;fill:#172033}.subtitle{font:11px sans-serif;fill:#475569}.note{font:11px sans-serif;fill:#475569}.metric{font:13px sans-serif;fill:#172033}.panel{fill:#fff;stroke:#cbd5e1;stroke-width:1}</style>\n",
    );
    draw_panel(
        &mut svg,
        Panel {
            values: &source_values,
            title: "12-bit source phantom",
            subtitle: "deterministic tissue and texture",
            range: (0.0, 4095.0),
            column: 0,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            values: lossless,
            title: "Lossless (NEAR = 0)",
            subtitle: "decoded native stream",
            range: (0.0, 4095.0),
            column: 1,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            values: near_lossless,
            title: "Near-lossless (NEAR = 3)",
            subtitle: "same 12-bit display range",
            range: (0.0, 4095.0),
            column: 2,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            values: &absolute_error,
            title: "Absolute error × contrast",
            subtitle: "difference, not anatomy",
            range: (0.0, near_bound),
            column: 3,
        },
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"303\" class=\"metric\">Lossless: {lossless_bytes} bytes · mismatches: {lossless_mismatches}</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"328\" class=\"metric\">Near-lossless: {near_lossless_bytes} bytes · changed samples: {changed_samples}/{}</text>",
        source.len()
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"353\" class=\"metric\">Measured max |decoded − source| = {maximum_error:.0} ≤ NEAR = {NEAR}</text>"
    )?;
    writeln!(svg, "</svg>")?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg).with_context(|| format!("write figure {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/jpeg_ls_codec.svg"));
    let source = phantom()?;

    let malformed = encode_grayscale_jpeg_ls(
        &source[..source.len() - 1],
        u32::try_from(HEIGHT).context("height exceeds u32")?,
        u32::try_from(WIDTH).context("width exceeds u32")?,
        PRECISION,
        0,
    );
    if !matches!(malformed, Err(JpegLsEncodeError::PixelCountMismatch { .. })) {
        bail!("malformed sample count did not return PixelCountMismatch");
    }

    let (lossless_stream, lossless) = encode_decode(&source, 0)?;
    let (near_stream, near_lossless) = encode_decode(&source, NEAR)?;
    write_figure(
        &output,
        &source,
        &lossless,
        &near_lossless,
        lossless_stream.len(),
        near_stream.len(),
    )?;
    println!("wrote {}", output.display());
    Ok(())
}
