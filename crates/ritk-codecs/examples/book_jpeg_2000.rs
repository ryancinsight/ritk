//! Generate the native JPEG 2000 figure used by the RITK mdBook.
//!
//! The example encodes one deterministic 12-bit medical-style phantom through
//! RITK's public reversible 5/3 and irreversible 9/7 paths, decodes each
//! codestream, verifies exact reversible reconstruction, and renders the
//! visible size/error tradeoff between unit and coarse scalar quantization.
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use ritk_codecs::jpeg_2000::encoder::{encode_grayscale_j2k, Jpeg2000Encoding, QuantizationStep};
use ritk_codecs::{decode_jpeg2000_fragment, PixelLayout, PixelSignedness};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const WIDTH: usize = 96;
const HEIGHT: usize = 96;
const PRECISION: u32 = 12;
const LEVELS: u8 = 3;
const COARSE_STEP: f32 = 32.0;
const PANEL_WIDTH: u32 = 260;
const PANEL_HEIGHT: u32 = 280;
const IMAGE_SIZE: u32 = 190;

fn phantom() -> Result<Vec<i32>> {
    let sample_count = WIDTH
        .checked_mul(HEIGHT)
        .context("phantom dimensions overflow usize")?;
    let mut samples = Vec::with_capacity(sample_count);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let x_coordinate = i32::try_from(x).context("x coordinate exceeds i32")?;
            let y_coordinate = i32::try_from(y).context("y coordinate exceeds i32")?;
            let dx = x_coordinate - 48;
            let dy = y_coordinate - 50;
            let outer = dx * dx + dy * dy <= 38 * 38;
            let inner = dx * dx + dy * dy <= 24 * 24;
            let lesion_dx = x_coordinate - 63;
            let lesion_dy = y_coordinate - 38;
            let lesion = lesion_dx * lesion_dx + lesion_dy * lesion_dy <= 7 * 7;
            let ridge = (x_coordinate - 30).abs() <= 2 && (24..=72).contains(&y_coordinate);
            let background = 120 + 3 * x_coordinate + 2 * y_coordinate;
            let tissue = if inner {
                2480 + 5 * x_coordinate - 3 * y_coordinate
            } else if outer {
                1120 + 2 * x_coordinate + y_coordinate
            } else {
                background
            };
            let value = if lesion {
                3600
            } else if ridge && outer {
                3200
            } else {
                tissue
            };
            samples.push(value.clamp(0, 4095));
        }
    }
    Ok(samples)
}

fn layout() -> PixelLayout {
    PixelLayout {
        rows: HEIGHT,
        cols: WIDTH,
        samples_per_pixel: 1,
        bits_allocated: u16::try_from(PRECISION).expect("invariant: example precision fits in u16"),
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    }
}

fn encode_decode(source: &[i32], encoding: Jpeg2000Encoding) -> Result<(Vec<u8>, Vec<f32>)> {
    let codestream = encode_grayscale_j2k(
        source,
        u32::try_from(HEIGHT).context("height exceeds u32")?,
        u32::try_from(WIDTH).context("width exceeds u32")?,
        PRECISION,
        PixelSignedness::Unsigned,
        encoding,
    )
    .context("encode native JPEG 2000 codestream")?;
    let reconstruction = decode_jpeg2000_fragment(&codestream, layout())
        .context("decode native JPEG 2000 codestream")?;
    Ok((codestream, reconstruction))
}

#[derive(Clone, Copy)]
enum ColorMap {
    Grayscale,
    ErrorHeat,
}

fn to_png(values: &[f32], lower: f32, upper: f32, color_map: ColorMap) -> Result<String> {
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
            match color_map {
                ColorMap::Grayscale => {
                    let gray = (normalized * 255.0).round() as u8;
                    raster.extend_from_slice(&[gray, gray, gray]);
                }
                ColorMap::ErrorHeat => {
                    let red = (normalized * 2.0).clamp(0.0, 1.0);
                    let green = (normalized * 2.0 - 1.0).clamp(0.0, 1.0);
                    raster.extend_from_slice(&[
                        (red * 255.0).round() as u8,
                        (green * 255.0).round() as u8,
                        0,
                    ]);
                }
            }
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
    color_map: ColorMap,
    column: u32,
}

struct ErrorMetrics {
    absolute_error: Vec<f32>,
    maximum_error: f32,
    psnr: f64,
}

struct Reconstruction<'a> {
    values: &'a [f32],
    codestream_bytes: usize,
}

struct FigureData<'a> {
    source: &'a [i32],
    reversible: Reconstruction<'a>,
    unit: Reconstruction<'a>,
    coarse: Reconstruction<'a>,
}

fn error_metrics(source: &[f32], reconstruction: &[f32]) -> Result<ErrorMetrics> {
    if reconstruction.len() != source.len() {
        bail!(
            "reconstruction sample count mismatch: got {}, expected {}",
            reconstruction.len(),
            source.len()
        );
    }
    let absolute_error: Vec<f32> = source
        .iter()
        .zip(reconstruction)
        .map(|(&expected, &actual)| (actual - expected).abs())
        .collect();
    let maximum_error = absolute_error.iter().copied().fold(0.0f32, f32::max);
    if maximum_error == 0.0 {
        bail!("irreversible reconstruction produced no visible error");
    }
    let mse = absolute_error
        .iter()
        .map(|&error| f64::from(error) * f64::from(error))
        .sum::<f64>()
        / f64::from(u32::try_from(absolute_error.len()).context("sample count exceeds u32")?);
    let peak = f64::from((1u32 << PRECISION) - 1);
    let psnr = 10.0 * (peak * peak / mse).log10();
    Ok(ErrorMetrics {
        absolute_error,
        maximum_error,
        psnr,
    })
}

fn draw_panel(svg: &mut String, panel: Panel<'_>) -> Result<()> {
    let Panel {
        values,
        title,
        subtitle,
        range,
        color_map,
        column,
    } = panel;
    let encoded = to_png(values, range.0, range.1, color_map)?;
    let offset_x = column
        .checked_mul(PANEL_WIDTH)
        .context("panel x offset overflows")?;
    writeln!(svg, "<g transform=\"translate({offset_x},0)\">")?;
    writeln!(
        svg,
        "<rect x=\"8\" y=\"8\" width=\"244\" height=\"264\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"31\" class=\"title\">{title}</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"48\" class=\"subtitle\">{subtitle}</text>"
    )?;
    writeln!(
        svg,
        "<image x=\"35\" y=\"62\" width=\"{IMAGE_SIZE}\" height=\"{IMAGE_SIZE}\" href=\"data:image/png;base64,{encoded}\" image-rendering=\"pixelated\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"268\" class=\"note\">display [{:.2}, {:.2}]</text>",
        range.0, range.1
    )?;
    writeln!(svg, "</g>")?;
    Ok(())
}

fn write_figure(path: &Path, data: FigureData<'_>) -> Result<()> {
    let FigureData {
        source,
        reversible,
        unit,
        coarse,
    } = data;
    // Every source sample is 12-bit, so conversion to f32 is exact.
    let source_values: Vec<f32> = source.iter().map(|&value| value as f32).collect();
    let reversible_mismatches = source_values
        .iter()
        .zip(reversible.values)
        .filter(|(expected, actual)| expected != actual)
        .count();
    if reversible_mismatches != 0 {
        bail!("reversible reconstruction has {reversible_mismatches} mismatched samples");
    }

    let unit_metrics = error_metrics(&source_values, unit.values)?;
    let coarse_metrics = error_metrics(&source_values, coarse.values)?;
    if coarse.codestream_bytes >= unit.codestream_bytes {
        bail!(
            "coarse quantization did not reduce codestream size: coarse={}, unit={}",
            coarse.codestream_bytes,
            unit.codestream_bytes
        );
    }
    if coarse_metrics.maximum_error <= unit_metrics.maximum_error {
        bail!(
            "coarse quantization did not increase maximum error: coarse={}, unit={}",
            coarse_metrics.maximum_error,
            unit_metrics.maximum_error
        );
    }

    let figure_width = PANEL_WIDTH * 5;
    let figure_height = PANEL_HEIGHT + 110;
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
            subtitle: "deterministic structures and edges",
            range: (0.0, 4095.0),
            color_map: ColorMap::Grayscale,
            column: 0,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            values: reversible.values,
            title: "Reversible 5/3",
            subtitle: "decoded native codestream",
            range: (0.0, 4095.0),
            color_map: ColorMap::Grayscale,
            column: 1,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            values: unit.values,
            title: "9/7, Δ = 1",
            subtitle: "higher fidelity · larger stream",
            range: (0.0, 4095.0),
            color_map: ColorMap::Grayscale,
            column: 2,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            values: coarse.values,
            title: "9/7, Δ = 32",
            subtitle: "coarser detail · smaller stream",
            range: (0.0, 4095.0),
            color_map: ColorMap::Grayscale,
            column: 3,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            values: &coarse_metrics.absolute_error,
            title: "Δ = 32 absolute error",
            subtitle: "black → red → yellow; magnified",
            range: (0.0, coarse_metrics.maximum_error),
            color_map: ColorMap::ErrorHeat,
            column: 4,
        },
    )?;
    writeln!(
        svg,
        "<text x=\"24\" y=\"310\" class=\"metric\">5/3: {} bytes · mismatched samples: {reversible_mismatches}</text>",
        reversible.codestream_bytes
    )?;
    writeln!(
        svg,
        "<text x=\"24\" y=\"336\" class=\"metric\">9/7 Δ=1: {} bytes · max error: {:.3} · PSNR: {:.2} dB</text>",
        unit.codestream_bytes,
        unit_metrics.maximum_error,
        unit_metrics.psnr
    )?;
    writeln!(
        svg,
        "<text x=\"24\" y=\"362\" class=\"metric\">9/7 Δ=32: {} bytes · max error: {:.3} · PSNR: {:.2} dB</text>",
        coarse.codestream_bytes,
        coarse_metrics.maximum_error,
        coarse_metrics.psnr
    )?;
    writeln!(
        svg,
        "<text x=\"690\" y=\"336\" class=\"metric\">larger Δ → fewer coefficient magnitudes → fewer EBCOT bits</text>"
    )?;
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
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/jpeg_2000_codec.svg"));
    let source = phantom()?;
    let (reversible_codestream, reversible) = encode_decode(
        &source,
        Jpeg2000Encoding::Lossless {
            decomposition_levels: LEVELS,
        },
    )?;
    let (unit_codestream, unit) = encode_decode(
        &source,
        Jpeg2000Encoding::Lossy {
            decomposition_levels: LEVELS,
            quantization_step: QuantizationStep::UNIT,
        },
    )?;
    let (coarse_codestream, coarse) = encode_decode(
        &source,
        Jpeg2000Encoding::Lossy {
            decomposition_levels: LEVELS,
            quantization_step: QuantizationStep::new(COARSE_STEP)
                .context("construct coarse quantization step")?,
        },
    )?;
    write_figure(
        &output,
        FigureData {
            source: &source,
            reversible: Reconstruction {
                values: &reversible,
                codestream_bytes: reversible_codestream.len(),
            },
            unit: Reconstruction {
                values: &unit,
                codestream_bytes: unit_codestream.len(),
            },
            coarse: Reconstruction {
                values: &coarse,
                codestream_bytes: coarse_codestream.len(),
            },
        },
    )?;
    println!("wrote {}", output.display());
    Ok(())
}
