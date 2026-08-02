//! Generate the MINC2 round-trip figure used by the RITK mdBook.

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use coeus_core::SequentialBackend;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};
use ritk_image::Image;
use ritk_minc::{read_minc, write_minc};
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

#[path = "../src/scaled_fixture.rs"]
mod scaled_fixture;

const SHAPE: [usize; 3] = [9, 64, 64];
const DISPLAY_SLICE: usize = 4;
const DISPLAY_SIZE: u32 = 180;
const PANEL_WIDTH: u32 = 238;

type Volume = Image<f32, SequentialBackend, 3>;

fn phantom_values() -> Result<Vec<f32>> {
    let [depth, rows, columns] = SHAPE;
    let voxel_count = depth
        .checked_mul(rows)
        .and_then(|plane| plane.checked_mul(columns))
        .context("MINC2 phantom dimensions overflow usize")?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(voxel_count)
        .context("reserve MINC2 phantom voxels")?;

    for z in 0..depth {
        for y in 0..rows {
            for x in 0..columns {
                let dx = i16::try_from(x).context("x coordinate exceeds i16")? - 32;
                let dy = i16::try_from(y).context("y coordinate exceeds i16")? - 32;
                let dz = i16::try_from(z).context("z coordinate exceeds i16")? - 4;
                let radius = dx * dx + dy * dy + 5 * dz * dz;
                let anatomy = if radius <= 10 * 10 {
                    860.0
                } else if radius <= 22 * 22 {
                    520.0
                } else if radius <= 29 * 29 {
                    180.0
                } else {
                    18.0
                };
                let left_ventricle = (dx + 8).pow(2) + 3 * dy.pow(2) <= 6 * 6;
                let right_ventricle = (dx - 8).pow(2) + 3 * dy.pow(2) <= 6 * 6;
                let ventricles = if left_ventricle || right_ventricle {
                    -310.0
                } else {
                    0.0
                };
                let lesion = if (dx - 17).pow(2) + (dy + 13).pow(2) + 3 * dz.pow(2) <= 6 * 6 {
                    340.0
                } else {
                    0.0
                };
                let marker = if (8..=14).contains(&x) && (41..=48).contains(&y) {
                    145.0
                } else {
                    0.0
                };
                let texture_index = (7 * dx + 11 * dy + 13 * dz).rem_euclid(19);
                let texture = f32::from(texture_index) * 1.5;
                values.push(anatomy + ventricles + lesion + marker + texture);
            }
        }
    }
    Ok(values)
}

fn make_volume(values: Vec<f32>) -> Result<Volume> {
    Image::from_flat_on(
        values,
        SHAPE,
        Point::new([18.0, -12.5, 27.25]),
        Spacing::new([2.0, 0.8, 0.8]),
        Direction::from_rows([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        &SequentialBackend,
    )
    .context("construct deterministic MINC2 example volume")
}

fn verify_round_trip(source: &Volume, decoded: &Volume) -> Result<()> {
    if decoded.shape() != source.shape()
        || decoded.origin() != source.origin()
        || decoded.spacing() != source.spacing()
        || decoded.direction() != source.direction()
    {
        bail!(
            "MINC2 geometry mismatch: source shape {:?}, decoded shape {:?}",
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
            "MINC2 voxel {index} differs: source {}, decoded {}",
            source_values[index],
            decoded_values[index]
        );
    }
    Ok(())
}

fn display_slice(volume: &Volume) -> Result<&[f32]> {
    let [depth, rows, columns] = volume.shape();
    if DISPLAY_SLICE >= depth {
        bail!("display slice {DISPLAY_SLICE} exceeds depth {depth}");
    }
    let plane = rows
        .checked_mul(columns)
        .context("MINC2 slice dimensions overflow usize")?;
    let start = DISPLAY_SLICE
        .checked_mul(plane)
        .context("MINC2 slice offset overflows usize")?;
    let end = start
        .checked_add(plane)
        .context("MINC2 slice end overflows usize")?;
    volume
        .data_slice()
        .context("borrow MINC2 volume voxels")?
        .get(start..end)
        .context("MINC2 display slice exceeds volume storage")
}

fn encode_png(raster: &[u8]) -> Result<String> {
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(raster, DISPLAY_SIZE, DISPLAY_SIZE, ColorType::Rgb8)
        .context("encode MINC2 book panel")?;
    Ok(STANDARD.encode(png))
}

fn grayscale_png(values: &[f32], lower: f32, upper: f32) -> Result<String> {
    let [_, rows, columns] = SHAPE;
    let display = usize::try_from(DISPLAY_SIZE).context("display size exceeds usize")?;
    let capacity = display
        .checked_mul(display)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("MINC2 panel size overflows usize")?;
    let mut raster = Vec::new();
    raster
        .try_reserve_exact(capacity)
        .context("reserve MINC2 grayscale panel")?;
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

fn difference_png(source: &[f32], decoded: &[f32]) -> Result<(String, f32, usize)> {
    let [_, rows, columns] = SHAPE;
    if source.len() != decoded.len() {
        bail!("MINC2 difference inputs have unequal lengths");
    }
    let maximum_error = source
        .iter()
        .zip(decoded)
        .map(|(&expected, &actual)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let differing = source
        .iter()
        .zip(decoded)
        .filter(|(expected, actual)| expected.to_bits() != actual.to_bits())
        .count();
    let display = usize::try_from(DISPLAY_SIZE).context("display size exceeds usize")?;
    let capacity = display
        .checked_mul(display)
        .and_then(|pixels| pixels.checked_mul(3))
        .context("MINC2 difference panel size overflows usize")?;
    let mut raster = Vec::new();
    raster
        .try_reserve_exact(capacity)
        .context("reserve MINC2 difference panel")?;
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
    Ok((encode_png(&raster)?, maximum_error, differing))
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
        .context("MINC2 panel offset overflows u32")?;
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

fn draw_contract_panel(svg: &mut String, source: &Volume, file_bytes: u64) -> Result<()> {
    let x = 3_u32
        .checked_mul(PANEL_WIDTH)
        .context("MINC2 contract panel offset overflows u32")?;
    writeln!(svg, "<g transform=\"translate({x},0)\">")?;
    writeln!(
        svg,
        "<rect x=\"7\" y=\"7\" width=\"224\" height=\"266\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"29\" class=\"title\">HDF5 + geometry</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"59\" class=\"label\">phantom.mnc</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"78\" class=\"small\">{file_bytes} bytes · contiguous f32</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"102\" class=\"small\">/minc-2.0/dimensions</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"119\" class=\"small\">/minc-2.0/image/0/image</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"150\" class=\"label\">Exact round-trip</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"169\" class=\"small\">shape {:?} (z, y, x)</text>",
        source.shape()
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"188\" class=\"small\">spacing [{}, {}, {}] mm</text>",
        source.spacing()[0],
        source.spacing()[1],
        source.spacing()[2]
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"207\" class=\"small\">origin [{}, {}, {}] mm</text>",
        source.origin()[0],
        source.origin()[1],
        source.origin()[2]
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"232\" class=\"success\">direction matrix preserved</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"17\" y=\"251\" class=\"small\">8 KiB writer scratch</text></g>"
    )?;
    Ok(())
}

fn draw_scaling_panel(svg: &mut String, stored: &[i16; 8], scaled: &Volume) -> Result<()> {
    let decoded = scaled
        .data_slice()
        .context("borrow scaled fixture voxels")?;
    let expected = [-1_000.0, -500.0, 0.0, 1_000.0, 0.0, 50.0, 100.0, 200.0];
    if decoded != expected {
        bail!("scaled MINC2 fixture mismatch: got {decoded:?}, expected {expected:?}");
    }

    writeln!(svg, "<g transform=\"translate(0,355)\">")?;
    writeln!(
        svg,
        "<rect x=\"7\" y=\"7\" width=\"938\" height=\"178\" class=\"panel\"/>"
    )?;
    writeln!(svg, "<text x=\"18\" y=\"31\" class=\"title\">Quantitative i16 scaling: stored codes and decoded intensities must differ</text>")?;
    writeln!(svg, "<text x=\"18\" y=\"55\" class=\"metric\">real = image-min + (stored − valid-min) × (image-max − image-min) / (valid-max − valid-min)</text>")?;
    writeln!(svg, "<text x=\"18\" y=\"79\" class=\"small\">valid_range = [0, 100] · first spatial axis selects one image range per slice</text>")?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"107\" class=\"label\">Slice 0 · image range [−1000, 1000]</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"275\" y=\"107\" class=\"small\">stored [{}, {}, {}, {}]</text>",
        stored[0], stored[1], stored[2], stored[3]
    )?;
    writeln!(
        svg,
        "<text x=\"550\" y=\"107\" class=\"success\">read_minc → [−1000, −500, 0, 1000]</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"18\" y=\"137\" class=\"label\">Slice 1 · image range [0, 200]</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"275\" y=\"137\" class=\"small\">stored [{}, {}, {}, {}]</text>",
        stored[4], stored[5], stored[6], stored[7]
    )?;
    writeln!(
        svg,
        "<text x=\"550\" y=\"137\" class=\"success\">read_minc → [0, 50, 100, 200]</text>"
    )?;
    writeln!(svg, "<text x=\"18\" y=\"166\" class=\"small\">These values come from the public reader, not from labels assembled for the figure.</text></g>")?;
    Ok(())
}

fn write_figure(
    output: &Path,
    source: &Volume,
    decoded: &Volume,
    file_bytes: u64,
    stored: &[i16; 8],
    scaled: &Volume,
) -> Result<()> {
    let source_slice = display_slice(source)?;
    let decoded_slice = display_slice(decoded)?;
    let source_png = grayscale_png(source_slice, 0.0, 1_200.0)?;
    let decoded_png = grayscale_png(decoded_slice, 0.0, 1_200.0)?;
    let (difference_png, maximum_error, differing) = difference_png(source_slice, decoded_slice)?;
    if maximum_error != 0.0 || differing != 0 {
        bail!("MINC2 figure contains {differing} differing voxels; max error {maximum_error}");
    }

    let width = 4_u32
        .checked_mul(PANEL_WIDTH)
        .context("MINC2 figure width overflows u32")?;
    let height = 555;
    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\">"
    )?;
    writeln!(
        svg,
        "<rect width=\"{width}\" height=\"{height}\" fill=\"#f8fafc\"/>"
    )?;
    svg.push_str("<style>.panel{fill:#fff;stroke:#cbd5e1}.title{font:600 15px sans-serif;fill:#172033}.label{font:600 12px sans-serif;fill:#172033}.small{font:11px sans-serif;fill:#475569}.metric{font:13px sans-serif;fill:#172033}.success{font:600 12px sans-serif;fill:#047857}</style>\n");
    draw_image_panel(
        &mut svg,
        0,
        "Source z = 4",
        "before MINC2 encoding",
        &source_png,
        "display range [0, 1200]",
    )?;
    draw_image_panel(
        &mut svg,
        1,
        "Decoded z = 4",
        "after public read_minc",
        &decoded_png,
        "same display range",
    )?;
    draw_image_panel(
        &mut svg,
        2,
        "Bitwise difference",
        "red would mark corruption",
        &difference_png,
        "0 differing voxels · max |Δ| = 0",
    )?;
    draw_contract_panel(&mut svg, source, file_bytes)?;
    writeln!(svg, "<text x=\"18\" y=\"302\" class=\"success\">The identical-looking panels are confirmed by the explicit zero-difference mask and a full-volume bit comparison.</text>")?;
    writeln!(svg, "<text x=\"18\" y=\"325\" class=\"metric\">One 3-D volume · x-fastest contiguous payload · exact f32 voxels · physical geometry retained</text>")?;
    draw_scaling_panel(&mut svg, stored, scaled)?;
    writeln!(svg, "</svg>")?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).context("create MINC2 figure directory")?;
    }
    std::fs::write(output, svg).context("write MINC2 book figure")
}

fn output_path() -> PathBuf {
    std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/minc_roundtrip.svg"))
}

fn main() -> Result<()> {
    let directory = tempfile::tempdir().context("create MINC2 example directory")?;
    let path = directory.path().join("phantom.mnc");
    let source = make_volume(phantom_values()?)?;
    write_minc(&source, &path, &SequentialBackend).context("write MINC2 phantom")?;
    let decoded = read_minc(&path, &SequentialBackend).context("read MINC2 phantom")?;
    verify_round_trip(&source, &decoded)?;
    let file_bytes = std::fs::metadata(&path)
        .context("inspect MINC2 file size")?
        .len();
    let scaled_path = directory.path().join("scaled-int16.mnc");
    let stored = [0_i16, 25, 50, 100, 0, 25, 50, 100];
    scaled_fixture::write_scaled_integer_fixture(
        &scaled_path,
        &stored,
        [2, 2, 2],
        [0, 100],
        scaled_fixture::ImageRangeFixture::Complete {
            minima: &[-1_000.0, 0.0],
            maxima: &[1_000.0, 200.0],
        },
    )
    .context("write deterministic scaled-integer MINC2 fixture")?;
    let scaled = read_minc(&scaled_path, &SequentialBackend)
        .context("read deterministic scaled-integer MINC2 fixture")?;
    write_figure(
        &output_path(),
        &source,
        &decoded,
        file_bytes,
        &stored,
        &scaled,
    )
}
