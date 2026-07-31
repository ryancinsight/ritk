//! Generate the TIFF round-trip figure used by the RITK mdBook.

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use coeus_core::SequentialBackend;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tiff::{read_tiff, write_tiff};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const SHAPE: [usize; 3] = [5, 64, 64];
const DISPLAY_SLICE: usize = 2;
const DISPLAY_SIZE: u32 = 180;
const PANEL_WIDTH: u32 = 238;

type Volume = Image<f32, SequentialBackend, 3>;

fn phantom_values() -> Result<Vec<f32>> {
    let [depth, rows, columns] = SHAPE;
    let voxel_count = depth
        .checked_mul(rows)
        .and_then(|plane| plane.checked_mul(columns))
        .context("TIFF phantom dimensions overflow usize")?;
    let mut values = Vec::with_capacity(voxel_count);
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..columns {
                let x_offset = i32::try_from(x).context("x coordinate exceeds i32")? - 32;
                let y_offset = i32::try_from(y).context("y coordinate exceeds i32")? - 32;
                let radius = x_offset * x_offset + y_offset * y_offset;
                let tissue = if radius <= 12 * 12 {
                    900.0
                } else if radius <= 25 * 25 {
                    420.0
                } else {
                    30.0
                };
                let page_marker = 70.0 * z as f32;
                let diagonal = if x.abs_diff(y + z * 3) <= 1 {
                    180.0
                } else {
                    0.0
                };
                values.push(tissue + page_marker + diagonal);
            }
        }
    }
    Ok(values)
}

fn make_volume(values: Vec<f32>) -> Result<Volume> {
    Image::from_flat_on(
        values,
        SHAPE,
        Point::new([10.0, -20.0, 30.0]),
        Spacing::new([0.5, 0.75, 2.0]),
        Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        &SequentialBackend,
    )
    .context("construct deterministic TIFF example volume")
}

fn verify_round_trip(source: &Volume, decoded: &Volume) -> Result<()> {
    if decoded.shape() != source.shape() {
        bail!(
            "TIFF shape mismatch: decoded {:?}, source {:?}",
            decoded.shape(),
            source.shape()
        );
    }
    let source_values = source.data_slice().context("borrow source voxels")?;
    let decoded_values = decoded.data_slice().context("borrow decoded voxels")?;
    if let Some(index) = source_values
        .iter()
        .zip(decoded_values)
        .position(|(expected, actual)| expected.to_bits() != actual.to_bits())
    {
        bail!(
            "TIFF voxel {index} differs: source {}, decoded {}",
            source_values[index],
            decoded_values[index]
        );
    }
    let default_origin = Point::new([0.0; 3]);
    let default_spacing = Spacing::new([1.0; 3]);
    let default_direction = Direction::identity();
    if decoded.origin() != &default_origin
        || decoded.spacing() != &default_spacing
        || decoded.direction() != &default_direction
    {
        bail!("TIFF reader did not assign documented default geometry");
    }
    Ok(())
}

fn slice(volume: &Volume, z: usize) -> Result<&[f32]> {
    let [depth, rows, columns] = volume.shape();
    if z >= depth {
        bail!("slice {z} exceeds depth {depth}");
    }
    let page_samples = rows
        .checked_mul(columns)
        .context("TIFF slice dimensions overflow usize")?;
    let start = z
        .checked_mul(page_samples)
        .context("TIFF slice offset overflows usize")?;
    let end = start
        .checked_add(page_samples)
        .context("TIFF slice end overflows usize")?;
    volume
        .data_slice()
        .context("borrow TIFF volume voxels")?
        .get(start..end)
        .context("TIFF slice exceeds volume storage")
}

fn grayscale_png(values: &[f32], lower: f32, upper: f32) -> Result<String> {
    let [_, rows, columns] = SHAPE;
    let display_size = usize::try_from(DISPLAY_SIZE).context("display size exceeds usize")?;
    let capacity = display_size
        .checked_mul(display_size)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("TIFF panel size overflows usize")?;
    let mut raster = Vec::with_capacity(capacity);
    for output_y in 0..display_size {
        let source_y = output_y * rows / display_size;
        for output_x in 0..display_size {
            let source_x = output_x * columns / display_size;
            let normalized =
                ((values[source_y * columns + source_x] - lower) / (upper - lower)).clamp(0.0, 1.0);
            let gray = (normalized * 255.0).round() as u8;
            raster.extend_from_slice(&[gray, gray, gray]);
        }
    }
    encode_png(&raster)
}

fn difference_png(source: &[f32], decoded: &[f32]) -> Result<(String, f32)> {
    let [_, rows, columns] = SHAPE;
    let display_size = usize::try_from(DISPLAY_SIZE).context("display size exceeds usize")?;
    let capacity = display_size
        .checked_mul(display_size)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("TIFF difference panel size overflows usize")?;
    let maximum_error = source
        .iter()
        .zip(decoded)
        .map(|(&expected, &actual)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let mut raster = Vec::with_capacity(capacity);
    for output_y in 0..display_size {
        let source_y = output_y * rows / display_size;
        for output_x in 0..display_size {
            let source_x = output_x * columns / display_size;
            let index = source_y * columns + source_x;
            let color = if source[index].to_bits() == decoded[index].to_bits() {
                [8, 47, 73]
            } else {
                [239, 68, 68]
            };
            raster.extend_from_slice(&color);
        }
    }
    Ok((encode_png(&raster)?, maximum_error))
}

fn encode_png(raster: &[u8]) -> Result<String> {
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(raster, DISPLAY_SIZE, DISPLAY_SIZE, ColorType::Rgb8)
        .context("encode TIFF book panel")?;
    Ok(STANDARD.encode(png))
}

fn draw_image_panel(
    svg: &mut String,
    column: u32,
    title: &str,
    subtitle: &str,
    png: &str,
    footer: &str,
) -> Result<()> {
    let x = column
        .checked_mul(PANEL_WIDTH)
        .context("TIFF panel offset overflows u32")?;
    writeln!(svg, "<g transform=\"translate({x},0)\">")?;
    writeln!(
        svg,
        "<rect x=\"7\" y=\"7\" width=\"224\" height=\"266\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"29\" class=\"title\">{title}</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"46\" class=\"small\">{subtitle}</text>"
    )?;
    writeln!(
        svg,
        "<image x=\"29\" y=\"57\" width=\"{DISPLAY_SIZE}\" height=\"{DISPLAY_SIZE}\" href=\"data:image/png;base64,{png}\" image-rendering=\"pixelated\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"259\" class=\"small\">{footer}</text></g>"
    )?;
    Ok(())
}

fn draw_metadata_panel(svg: &mut String, file_bytes: u64) -> Result<()> {
    let x = 3 * PANEL_WIDTH;
    writeln!(svg, "<g transform=\"translate({x},0)\">")?;
    writeln!(
        svg,
        "<rect x=\"7\" y=\"7\" width=\"224\" height=\"266\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"29\" class=\"title\">Metadata boundary</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"55\" class=\"label\">Source geometry</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"75\" class=\"small\">origin [10, −20, 30] mm</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"92\" class=\"small\">spacing [0.5, 0.75, 2] mm</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"109\" class=\"small\">direction: 90° axial rotation</text>"
    )?;
    writeln!(
        svg,
        "<path d=\"M119 121 v28 m0 0 l-7-9 m7 9 l7-9\" class=\"arrow\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"170\" class=\"label\">Decoded defaults</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"190\" class=\"small\">origin [0, 0, 0]</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"207\" class=\"small\">spacing [1, 1, 1]</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"224\" class=\"small\">direction: identity</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"253\" class=\"small\">file size: {file_bytes} bytes</text></g>"
    )?;
    Ok(())
}

fn write_figure(output: &Path, source: &Volume, decoded: &Volume, file_bytes: u64) -> Result<()> {
    let source_slice = slice(source, DISPLAY_SLICE)?;
    let decoded_slice = slice(decoded, DISPLAY_SLICE)?;
    let source_png = grayscale_png(source_slice, 0.0, 1_300.0)?;
    let decoded_png = grayscale_png(decoded_slice, 0.0, 1_300.0)?;
    let (difference_png, maximum_error) = difference_png(source_slice, decoded_slice)?;
    if maximum_error != 0.0 {
        bail!("TIFF figure has nonzero voxel error {maximum_error}");
    }

    let width = 4 * PANEL_WIDTH;
    let height = 340;
    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\">"
    )?;
    writeln!(
        svg,
        "<rect width=\"{width}\" height=\"{height}\" fill=\"#f8fafc\"/>"
    )?;
    svg.push_str("<style>.panel{fill:#fff;stroke:#cbd5e1}.title{font:600 15px sans-serif;fill:#172033}.label{font:600 12px sans-serif;fill:#172033}.small{font:11px sans-serif;fill:#475569}.metric{font:13px sans-serif;fill:#172033}.success{font:600 13px sans-serif;fill:#047857}.arrow{fill:none;stroke:#0f766e;stroke-width:2}</style>\n");
    draw_image_panel(
        &mut svg,
        0,
        "Source z = 2",
        "third slice in [z, y, x]",
        &source_png,
        "display range [0, 1300]",
    )?;
    draw_image_panel(
        &mut svg,
        1,
        "Decoded page 3",
        "IFD page 3 maps to z = 2",
        &decoded_png,
        "same display range",
    )?;
    draw_image_panel(
        &mut svg,
        2,
        "Bitwise difference",
        "red would mark a mismatch",
        &difference_png,
        "max |decoded − source| = 0",
    )?;
    draw_metadata_panel(&mut svg, file_bytes)?;
    writeln!(svg, "<text x=\"18\" y=\"302\" class=\"success\">Pixels and five-page order are exact; physical geometry is intentionally reset at this TIFF boundary.</text>")?;
    writeln!(svg, "<text x=\"18\" y=\"325\" class=\"metric\">Shape [5, 64, 64] · Gray32Float pages · one IFD per z-slice · page 1 → z=0, page 3 → z=2, page 5 → z=4</text></svg>")?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).context("create TIFF figure directory")?;
    }
    std::fs::write(output, svg).context("write TIFF book figure")
}

fn output_path() -> PathBuf {
    std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/tiff_roundtrip.svg"))
}

fn main() -> Result<()> {
    let directory = tempfile::tempdir().context("create TIFF example directory")?;
    let path = directory.path().join("phantom.tiff");
    let source = make_volume(phantom_values()?)?;
    write_tiff(&source, &path, &SequentialBackend).context("write TIFF phantom")?;
    let decoded = read_tiff(&path, &SequentialBackend).context("read TIFF phantom")?;
    verify_round_trip(&source, &decoded)?;
    let file_bytes = std::fs::metadata(&path)
        .context("inspect TIFF file size")?
        .len();
    write_figure(&output_path(), &source, &decoded, file_bytes)
}
