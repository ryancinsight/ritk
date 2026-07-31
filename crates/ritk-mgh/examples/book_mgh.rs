//! Generate the MGH/MGZ round-trip figure used by the RITK mdBook.
//!
//! The example constructs a deterministic three-dimensional MR-like phantom
//! with non-default spatial metadata, writes it through RITK's public MGH and
//! MGZ APIs, reads both files back, verifies bit-exact voxels and geometry, and
//! demonstrates that a complete multi-frame payload is rejected rather than
//! silently truncated to its first frame.

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use coeus_core::SequentialBackend;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use ritk_image::Image;
use ritk_mgh::{read_mgh, write_mgh};
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const SHAPE: [usize; 3] = [15, 96, 96];
const HEADER_BYTES: usize = 284;
const FRAME_COUNT_OFFSET: usize = 16;
const PANEL_WIDTH: u32 = 238;
const PANEL_HEIGHT: u32 = 270;
const DISPLAY_SIZE: u32 = 174;

type Volume = Image<f32, SequentialBackend, 3>;

fn phantom_values() -> Result<Vec<f32>> {
    let [nz, ny, nx] = SHAPE;
    let voxel_count = nz
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nx))
        .context("phantom dimensions overflow usize")?;
    let mut values = Vec::with_capacity(voxel_count);

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let x_coordinate = i32::try_from(x).context("x coordinate exceeds i32")? - 48;
                let y_coordinate = i32::try_from(y).context("y coordinate exceeds i32")? - 48;
                let z_coordinate = i32::try_from(z).context("z coordinate exceeds i32")? - 7;
                let head = x_coordinate * x_coordinate
                    + y_coordinate * y_coordinate
                    + 7 * z_coordinate * z_coordinate
                    <= 42 * 42;
                let white_matter = x_coordinate * x_coordinate
                    + 2 * y_coordinate * y_coordinate
                    + 9 * z_coordinate * z_coordinate
                    <= 27 * 27;
                let left_ventricle = (x_coordinate + 10).pow(2) + 3 * y_coordinate.pow(2) <= 8 * 8;
                let right_ventricle = (x_coordinate - 10).pow(2) + 3 * y_coordinate.pow(2) <= 8 * 8;
                let lesion = (x_coordinate - 19).pow(2)
                    + (y_coordinate + 14).pow(2)
                    + 4 * z_coordinate.pow(2)
                    <= 7 * 7;
                let texture =
                    (5 * x_coordinate + 3 * y_coordinate + 11 * z_coordinate).rem_euclid(23) as f32;

                let value = if !head {
                    20.0 + texture
                } else if left_ventricle || right_ventricle {
                    140.0 + texture
                } else if lesion {
                    3_300.0 + texture
                } else if white_matter {
                    2_350.0 + texture
                } else {
                    1_120.0 + texture
                };
                values.push(value);
            }
        }
    }
    Ok(values)
}

fn make_volume(values: Vec<f32>) -> Result<Volume> {
    Image::from_flat_on(
        values,
        SHAPE,
        Point::new([12.5, -18.25, 32.0]),
        Spacing::new([0.75, 0.75, 1.5]),
        Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        &SequentialBackend,
    )
    .context("construct deterministic MGH example volume")
}

fn verify_round_trip(source: &Volume, decoded: &Volume, label: &str) -> Result<()> {
    if decoded.shape() != source.shape() {
        bail!(
            "{label} shape mismatch: decoded {:?}, source {:?}",
            decoded.shape(),
            source.shape()
        );
    }
    if decoded.origin() != source.origin()
        || decoded.spacing() != source.spacing()
        || decoded.direction() != source.direction()
    {
        bail!(
            "{label} geometry mismatch: origin {:?}/{:?}, spacing {:?}/{:?}, \
             direction {:?}/{:?}",
            decoded.origin(),
            source.origin(),
            decoded.spacing(),
            source.spacing(),
            decoded.direction(),
            source.direction()
        );
    }

    let source_values = source.data_slice().context("borrow source voxels")?;
    let decoded_values = decoded.data_slice().context("borrow decoded voxels")?;
    let mismatch = source_values
        .iter()
        .zip(decoded_values)
        .position(|(expected, actual)| expected.to_bits() != actual.to_bits());
    if let Some(index) = mismatch {
        bail!(
            "{label} voxel {index} differs: source {}, decoded {}",
            source_values[index],
            decoded_values[index]
        );
    }
    Ok(())
}

fn central_slice(volume: &Volume) -> Result<Vec<f32>> {
    let [nz, ny, nx] = volume.shape();
    let slice_voxels = ny
        .checked_mul(nx)
        .context("slice dimensions overflow usize")?;
    let start = (nz / 2)
        .checked_mul(slice_voxels)
        .context("slice offset overflows usize")?;
    let end = start
        .checked_add(slice_voxels)
        .context("slice end overflows usize")?;
    Ok(volume
        .data_slice()
        .context("borrow volume voxels")?
        .get(start..end)
        .context("central slice exceeds volume data")?
        .to_vec())
}

fn absolute_error(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        bail!(
            "difference inputs have unequal lengths: {} and {}",
            left.len(),
            right.len()
        );
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(&expected, &actual)| (actual - expected).abs())
        .collect())
}

fn grayscale_png(values: &[f32], lower: f32, upper: f32) -> Result<String> {
    let [_, height, width] = SHAPE;
    if values.len() != height * width {
        bail!(
            "panel sample count mismatch: got {}, expected {}",
            values.len(),
            height * width
        );
    }
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        bail!("invalid display range [{lower}, {upper}]");
    }

    let display_size = usize::try_from(DISPLAY_SIZE).context("display size exceeds usize")?;
    let capacity = display_size
        .checked_mul(display_size)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("raster size overflows usize")?;
    let mut raster = Vec::with_capacity(capacity);
    for output_y in 0..display_size {
        let source_y = output_y * height / display_size;
        for output_x in 0..display_size {
            let source_x = output_x * width / display_size;
            let value = values[source_y * width + source_x];
            let normalized = ((value - lower) / (upper - lower)).clamp(0.0, 1.0);
            let gray = (normalized * 255.0).round() as u8;
            raster.extend_from_slice(&[gray, gray, gray]);
        }
    }

    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&raster, DISPLAY_SIZE, DISPLAY_SIZE, ColorType::Rgb8)
        .context("encode grayscale panel")?;
    Ok(STANDARD.encode(png))
}

fn error_png(values: &[f32]) -> Result<String> {
    let [_, height, width] = SHAPE;
    if values.len() != height * width {
        bail!("error panel sample count does not match slice geometry");
    }
    let display_size = usize::try_from(DISPLAY_SIZE).context("display size exceeds usize")?;
    let capacity = display_size
        .checked_mul(display_size)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("error raster size overflows usize")?;
    let mut raster = Vec::with_capacity(capacity);
    for output_y in 0..display_size {
        let source_y = output_y * height / display_size;
        for output_x in 0..display_size {
            let source_x = output_x * width / display_size;
            let error = values[source_y * width + source_x];
            let color = if error == 0.0 {
                [8, 47, 73]
            } else {
                [239, 68, 68]
            };
            raster.extend_from_slice(&color);
        }
    }

    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&raster, DISPLAY_SIZE, DISPLAY_SIZE, ColorType::Rgb8)
        .context("encode error panel")?;
    Ok(STANDARD.encode(png))
}

struct Panel<'a> {
    title: &'a str,
    subtitle: &'a str,
    encoded_png: &'a str,
    footer: &'a str,
    column: u32,
}

fn draw_panel(svg: &mut String, panel: Panel<'_>) -> Result<()> {
    let offset_x = panel
        .column
        .checked_mul(PANEL_WIDTH)
        .context("panel offset overflows u32")?;
    writeln!(svg, "<g transform=\"translate({offset_x},0)\">")?;
    writeln!(
        svg,
        "<rect x=\"7\" y=\"7\" width=\"224\" height=\"256\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"29\" class=\"title\">{}</text>",
        panel.title
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"46\" class=\"subtitle\">{}</text>",
        panel.subtitle
    )?;
    writeln!(
        svg,
        "<image x=\"32\" y=\"55\" width=\"{DISPLAY_SIZE}\" height=\"{DISPLAY_SIZE}\" \
         href=\"data:image/png;base64,{}\" image-rendering=\"pixelated\"/>",
        panel.encoded_png
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"251\" class=\"footer\">{}</text>",
        panel.footer
    )?;
    writeln!(svg, "</g>")?;
    Ok(())
}

fn verify_multi_frame_rejection(single_frame_path: &Path, output_path: &Path) -> Result<()> {
    let mut bytes = std::fs::read(single_frame_path).context("read single-frame MGH fixture")?;
    if bytes.len() < HEADER_BYTES {
        bail!("writer produced an MGH shorter than its header");
    }
    let frame = bytes[HEADER_BYTES..].to_vec();
    bytes
        .get_mut(FRAME_COUNT_OFFSET..FRAME_COUNT_OFFSET + 4)
        .context("MGH frame-count field is absent")?
        .copy_from_slice(&2_i32.to_be_bytes());
    bytes.extend_from_slice(&frame);
    std::fs::write(output_path, bytes).context("write complete two-frame MGH fixture")?;

    let error = read_mgh(output_path, &SequentialBackend)
        .err()
        .context("two-frame MGH was accepted as a three-dimensional image")?;
    let message = format!("{error:#}");
    if !message.contains("2 frames") {
        bail!("multi-frame rejection does not name the frame count: {message}");
    }
    Ok(())
}

fn write_figure(
    output: &Path,
    source: &Volume,
    decoded_mgh: &Volume,
    decoded_mgz: &Volume,
    mgh_bytes: u64,
    mgz_bytes: u64,
) -> Result<()> {
    let source_slice = central_slice(source)?;
    let mgh_slice = central_slice(decoded_mgh)?;
    let mgz_slice = central_slice(decoded_mgz)?;
    let mgh_error = absolute_error(&source_slice, &mgh_slice)?;
    let mgz_error = absolute_error(&source_slice, &mgz_slice)?;
    let combined_error: Vec<f32> = mgh_error
        .iter()
        .zip(&mgz_error)
        .map(|(&left, &right)| left.max(right))
        .collect();
    let maximum_error = combined_error.iter().copied().fold(0.0f32, f32::max);
    if maximum_error != 0.0 {
        bail!("round-trip figure has nonzero voxel error {maximum_error}");
    }

    let source_png = grayscale_png(&source_slice, 0.0, 3_500.0)?;
    let mgh_png = grayscale_png(&mgh_slice, 0.0, 3_500.0)?;
    let mgz_png = grayscale_png(&mgz_slice, 0.0, 3_500.0)?;
    let error_png = error_png(&combined_error)?;
    let width = PANEL_WIDTH * 4;
    let height = PANEL_HEIGHT + 110;
    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\">"
    )?;
    writeln!(
        svg,
        "<rect width=\"{width}\" height=\"{height}\" fill=\"#f8fafc\"/>"
    )?;
    svg.push_str(
        "<style>.panel{fill:#fff;stroke:#cbd5e1}.title{font:600 15px sans-serif;fill:#172033}.subtitle,.footer{font:11px sans-serif;fill:#475569}.metric{font:13px sans-serif;fill:#172033}.success{font:600 13px sans-serif;fill:#047857}.bar-bg{fill:#e2e8f0}.bar{fill:#0f766e}</style>\n",
    );
    draw_panel(
        &mut svg,
        Panel {
            title: "Source volume",
            subtitle: "central MR-like slice",
            encoded_png: &source_png,
            footer: "display range [0, 3500]",
            column: 0,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            title: "Decoded .mgh",
            subtitle: "uncompressed big-endian",
            encoded_png: &mgh_png,
            footer: "same display range",
            column: 1,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            title: "Decoded .mgz",
            subtitle: "gzip-compressed MGH",
            encoded_png: &mgz_png,
            footer: "same display range",
            column: 2,
        },
    )?;
    draw_panel(
        &mut svg,
        Panel {
            title: "Absolute difference",
            subtitle: "red would mark any mismatch",
            encoded_png: &error_png,
            footer: "max |decoded − source| = 0",
            column: 3,
        },
    )?;

    let ratio = if mgh_bytes == 0 {
        0.0
    } else {
        mgz_bytes as f64 / mgh_bytes as f64
    };
    let bar_width = (ratio.clamp(0.0, 1.0) * 300.0).round() as u32;
    writeln!(
        svg,
        "<text x=\"18\" y=\"282\" class=\"metric\">Geometry preserved: origin [12.5, −18.25, 32.0] mm · spacing [0.75, 0.75, 1.5] mm · 90° axial rotation</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"307\" class=\"metric\">File bytes: MGH {mgh_bytes} · MGZ {mgz_bytes} ({:.1}% of MGH)</text>",
        ratio * 100.0
    )?;
    writeln!(
        svg,
        "<rect x=\"600\" y=\"294\" width=\"300\" height=\"15\" rx=\"3\" class=\"bar-bg\"/><rect x=\"600\" y=\"294\" width=\"{bar_width}\" height=\"15\" rx=\"3\" class=\"bar\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"353\" class=\"success\">Safety boundary: complete two-frame input rejected (the 3-D Image contract accepts exactly one frame)</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"375\" class=\"subtitle\">The anatomy panels should match. The difference panel and numeric contract prove exact reconstruction, rather than asking the reader to judge two similar images.</text>"
    )?;
    writeln!(svg, "</svg>")?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).context("create figure output directory")?;
    }
    std::fs::write(output, svg).context("write MGH book figure")
}

fn output_path() -> PathBuf {
    std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/mgh_roundtrip.svg"))
}

fn main() -> Result<()> {
    let directory = tempfile::tempdir().context("create MGH example temporary directory")?;
    let mgh_path = directory.path().join("phantom.mgh");
    let mgz_path = directory.path().join("phantom.mgz");
    let multi_frame_path = directory.path().join("two_frames.mgh");
    let source = make_volume(phantom_values()?)?;

    write_mgh(&source, &mgh_path, &SequentialBackend).context("write MGH volume")?;
    write_mgh(&source, &mgz_path, &SequentialBackend).context("write MGZ volume")?;
    let decoded_mgh = read_mgh(&mgh_path, &SequentialBackend).context("read MGH volume")?;
    let decoded_mgz = read_mgh(&mgz_path, &SequentialBackend).context("read MGZ volume")?;
    verify_round_trip(&source, &decoded_mgh, "MGH")?;
    verify_round_trip(&source, &decoded_mgz, "MGZ")?;
    verify_multi_frame_rejection(&mgh_path, &multi_frame_path)?;

    write_figure(
        &output_path(),
        &source,
        &decoded_mgh,
        &decoded_mgz,
        std::fs::metadata(&mgh_path)
            .context("inspect MGH size")?
            .len(),
        std::fs::metadata(&mgz_path)
            .context("inspect MGZ size")?
            .len(),
    )
}
