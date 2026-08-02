//! Generate the Analyze 7.5 round-trip figure used by the RITK mdBook.

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use coeus_core::SequentialBackend;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use ritk_analyze::{read_analyze, write_analyze};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

const SHAPE: [usize; 3] = [4, 64, 64];
const DISPLAY_SLICE: usize = 2;
const DISPLAY_SIZE: u32 = 180;
const PANEL_WIDTH: u32 = 238;

type Volume = Image<f32, SequentialBackend, 3>;

fn phantom_values() -> Result<Vec<f32>> {
    let [depth, rows, columns] = SHAPE;
    let voxel_count = depth
        .checked_mul(rows)
        .and_then(|plane| plane.checked_mul(columns))
        .context("Analyze phantom dimensions overflow usize")?;
    let mut values = Vec::with_capacity(voxel_count);
    for z in 0..depth {
        for y in 0..rows {
            for x in 0..columns {
                let dx = i32::try_from(x).context("x coordinate exceeds i32")? - 32;
                let dy = i32::try_from(y).context("y coordinate exceeds i32")? - 32;
                let radius = dx * dx + dy * dy;
                let anatomy = if radius <= 11 * 11 {
                    820.0
                } else if radius <= 23 * 23 {
                    430.0
                } else if radius <= 29 * 29 {
                    130.0
                } else {
                    20.0
                };
                let asymmetric_marker = if (9..=15).contains(&x) && (25..=39).contains(&y) {
                    170.0
                } else {
                    0.0
                };
                let diagonal = if x.abs_diff(y + z * 2) <= 1 {
                    90.0
                } else {
                    0.0
                };
                values.push(anatomy + asymmetric_marker + diagonal + 25.0 * z as f32);
            }
        }
    }
    Ok(values)
}

fn make_volume(values: Vec<f32>) -> Result<Volume> {
    Image::from_flat_on(
        values,
        SHAPE,
        Point::new([3.0, -4.5, 8.0]),
        Spacing::new([2.0, 1.5, 0.75]),
        Direction::identity(),
        &SequentialBackend,
    )
    .context("construct deterministic Analyze example volume")
}

fn verify_round_trip(source: &Volume, decoded: &Volume) -> Result<()> {
    if decoded.shape() != source.shape()
        || decoded.spacing() != source.spacing()
        || decoded.origin() != source.origin()
        || decoded.direction() != source.direction()
    {
        bail!(
            "Analyze geometry mismatch: source shape {:?}, decoded shape {:?}",
            source.shape(),
            decoded.shape()
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
            "Analyze voxel {index} differs: source {}, decoded {}",
            source_values[index],
            decoded_values[index]
        );
    }
    Ok(())
}

fn slice(volume: &Volume) -> Result<&[f32]> {
    let [depth, rows, columns] = volume.shape();
    if DISPLAY_SLICE >= depth {
        bail!("display slice {DISPLAY_SLICE} exceeds depth {depth}");
    }
    let plane = rows
        .checked_mul(columns)
        .context("Analyze slice dimensions overflow usize")?;
    let start = DISPLAY_SLICE
        .checked_mul(plane)
        .context("Analyze slice offset overflows usize")?;
    volume
        .data_slice()
        .context("borrow Analyze volume voxels")?
        .get(start..start + plane)
        .context("Analyze display slice exceeds volume storage")
}

fn grayscale_png(values: &[f32], lower: f32, upper: f32) -> Result<String> {
    let [_, rows, columns] = SHAPE;
    let display = usize::try_from(DISPLAY_SIZE).context("display size exceeds usize")?;
    let mut raster = Vec::with_capacity(display * display * 3);
    for output_y in 0..display {
        let source_y = output_y * rows / display;
        for output_x in 0..display {
            let source_x = output_x * columns / display;
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
    let display = usize::try_from(DISPLAY_SIZE).context("display size exceeds usize")?;
    let maximum_error = source
        .iter()
        .zip(decoded)
        .map(|(&expected, &actual)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let mut raster = Vec::with_capacity(display * display * 3);
    for output_y in 0..display {
        let source_y = output_y * rows / display;
        for output_x in 0..display {
            let source_x = output_x * columns / display;
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
        .context("encode Analyze book panel")?;
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
    let x = column * PANEL_WIDTH;
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
    writeln!(svg, "<image x=\"29\" y=\"57\" width=\"{DISPLAY_SIZE}\" height=\"{DISPLAY_SIZE}\" href=\"data:image/png;base64,{png}\" image-rendering=\"pixelated\"/>")?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"259\" class=\"small\">{footer}</text></g>"
    )?;
    Ok(())
}

fn draw_contract_panel(
    svg: &mut String,
    source: &Volume,
    header_bytes: u64,
    image_bytes: u64,
) -> Result<()> {
    let x = 3 * PANEL_WIDTH;
    writeln!(svg, "<g transform=\"translate({x},0)\">")?;
    writeln!(
        svg,
        "<rect x=\"7\" y=\"7\" width=\"224\" height=\"266\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"29\" class=\"title\">File pair + geometry</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"57\" class=\"label\">phantom.hdr</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"76\" class=\"small\">{header_bytes} bytes · dimensions</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"93\" class=\"small\">spacing · datatype · originator</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"124\" class=\"label\">phantom.img</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"143\" class=\"small\">{image_bytes} bytes · little-endian f32</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"174\" class=\"label\">Round-trip geometry</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"193\" class=\"small\">shape {:?}</text>",
        source.shape()
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"210\" class=\"small\">spacing [{}, {}, {}] mm</text>",
        source.spacing()[0],
        source.spacing()[1],
        source.spacing()[2]
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"227\" class=\"small\">origin [{}, {}, {}] mm</text>",
        source.origin()[0],
        source.origin()[1],
        source.origin()[2]
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"251\" class=\"warning\">direction is identity only</text></g>"
    )?;
    Ok(())
}

fn write_figure(
    output: &Path,
    source: &Volume,
    decoded: &Volume,
    header_bytes: u64,
    image_bytes: u64,
) -> Result<()> {
    let source_slice = slice(source)?;
    let decoded_slice = slice(decoded)?;
    let source_png = grayscale_png(source_slice, 0.0, 1_100.0)?;
    let decoded_png = grayscale_png(decoded_slice, 0.0, 1_100.0)?;
    let (difference_png, maximum_error) = difference_png(source_slice, decoded_slice)?;
    if maximum_error != 0.0 {
        bail!("Analyze figure has nonzero voxel error {maximum_error}");
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
    svg.push_str("<style>.panel{fill:#fff;stroke:#cbd5e1}.title{font:600 15px sans-serif;fill:#172033}.label{font:600 12px sans-serif;fill:#172033}.small{font:11px sans-serif;fill:#475569}.metric{font:13px sans-serif;fill:#172033}.success{font:600 13px sans-serif;fill:#047857}.warning{font:600 11px sans-serif;fill:#b45309}</style>\n");
    draw_image_panel(
        &mut svg,
        0,
        "Source z = 2",
        "before file encoding",
        &source_png,
        "display range [0, 1100]",
    )?;
    draw_image_panel(
        &mut svg,
        1,
        "Decoded z = 2",
        "after .hdr/.img read",
        &decoded_png,
        "same display range",
    )?;
    draw_image_panel(
        &mut svg,
        2,
        "Absolute difference",
        "red would mark a mismatch",
        &difference_png,
        "max |decoded − source| = 0",
    )?;
    draw_contract_panel(&mut svg, source, header_bytes, image_bytes)?;
    writeln!(svg, "<text x=\"18\" y=\"302\" class=\"success\">The source and decoded slices are bit-identical; the difference panel is uniformly zero.</text>")?;
    writeln!(svg, "<text x=\"18\" y=\"325\" class=\"metric\">One 3-D volume · X-fastest payload · exact f32 values · trailing payload bytes rejected</text></svg>")?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).context("create Analyze figure directory")?;
    }
    std::fs::write(output, svg).context("write Analyze book figure")
}

fn output_path() -> PathBuf {
    std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/analyze_roundtrip.svg"))
}

fn main() -> Result<()> {
    let directory = tempfile::tempdir().context("create Analyze example directory")?;
    let path = directory.path().join("phantom.hdr");
    let source = make_volume(phantom_values()?)?;
    write_analyze(&path, &source, &SequentialBackend).context("write Analyze phantom")?;
    let decoded = read_analyze(&path, &SequentialBackend).context("read Analyze phantom")?;
    verify_round_trip(&source, &decoded)?;

    let header_bytes = std::fs::metadata(&path)
        .context("inspect Analyze header size")?
        .len();
    let image_path = path.with_extension("img");
    let image_bytes = std::fs::metadata(&image_path)
        .context("inspect Analyze payload size")?
        .len();
    write_figure(&output_path(), &source, &decoded, header_bytes, image_bytes)?;

    OpenOptions::new()
        .append(true)
        .open(&image_path)
        .context("open Analyze payload for malformed-tail check")?
        .write_all(&[0])
        .context("append malformed Analyze tail")?;
    let error = read_analyze(&path, &SequentialBackend)
        .expect_err("Analyze reader must reject trailing payload bytes");
    if !error.to_string().contains("length mismatch") {
        bail!("unexpected malformed-tail error: {error:#}");
    }
    Ok(())
}
