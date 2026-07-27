//! Generate the processing-pipeline figure used by the RITK mdBook.
//!
//! The pipeline is deliberately deterministic and uses one scalar phantom for
//! every panel. Each stage runs through RITK's public Coeus-native image API;
//! the SVG renderer only visualizes the resulting voxel values.

use anyhow::{bail, Context, Result};
use coeus_core::SequentialBackend;
use eunomia::CastFrom;
use ritk_filter::{
    BinaryDilateFilter, BinaryErodeFilter, CurvatureFlowConfig, CurvatureFlowImageFilter,
    DiffusionConfig, GradientMagnitudeFilter, GrayscaleOpeningFilter, SigmoidImageFilter,
    ThresholdImageFilter,
};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

type Backend = SequentialBackend;
type ScalarImage = Image<f32, Backend, 3>;

const DIMS: [usize; 3] = [3, 128, 128];
const PANEL_WIDTH: u32 = 256;
const PANEL_HEIGHT: u32 = 280;
const PANEL_COLUMNS: u32 = 4;
const PANEL_ROWS: u32 = 3;
const DISPLAY_SIZE: usize = 208;

fn phantom() -> Result<Vec<f32>> {
    let [depth, height, width] = DIMS;
    let capacity = depth
        .checked_mul(height)
        .and_then(|size| size.checked_mul(width))
        .context("phantom size overflows")?;
    let mut values = Vec::with_capacity(capacity);
    for _ in 0..depth {
        for y_index in 0..height {
            for x_index in 0..width {
                let x = f32::from(u16::try_from(x_index).context("x coordinate exceeds u16")?);
                let y = f32::from(u16::try_from(y_index).context("y coordinate exceeds u16")?);
                let main = ((x - 58.0).powi(2) + (y - 66.0).powi(2)) / (2.0 * 28.0_f32.powi(2));
                let secondary =
                    ((x - 91.0).powi(2) + (y - 45.0).powi(2)) / (2.0 * 13.0_f32.powi(2));
                let crescent = if (x - 46.0).abs() < 17.0 && (y - 93.0).abs() < 9.0 {
                    0.16
                } else {
                    0.0
                };
                let hole = ((x - 57.0).powi(2) + (y - 65.0).powi(2)) < 6.0_f32.powi(2);
                let residue = i16::try_from((x_index * 19 + y_index * 31) % 29)
                    .context("phantom residue exceeds i16")?;
                let noise = f32::from(residue - 14) / 320.0;
                let tissue = 0.72 * (-main).exp() + 0.35 * (-secondary).exp() + crescent + noise;
                values.push(if hole { 0.22 } else { tissue.clamp(0.0, 1.0) });
            }
        }
    }
    Ok(values)
}

fn image(values: Vec<f32>, backend: &Backend) -> Result<ScalarImage> {
    Image::from_flat_on(
        values,
        DIMS,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        backend,
    )
    .context("construct phantom image")
}

fn min_max(values: &[f32]) -> Result<(f32, f32)> {
    let mut range = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(None, |range: Option<(f32, f32)>, value| {
            Some(match range {
                Some((lower, upper)) => (lower.min(value), upper.max(value)),
                None => (value, value),
            })
        });
    let range = range.take().context("figure input has no finite values")?;
    if range.0 == range.1 {
        bail!("figure display range is degenerate at {}", range.0);
    }
    Ok(range)
}

fn gray(value: f32, lower: f32, upper: f32) -> u8 {
    if !value.is_finite() || !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return 0;
    }
    let mapped = ((value - lower) / (upper - lower)).clamp(0.0, 1.0) * 255.0;
    u8::cast_from(mapped.round())
}

fn draw_image_panel(
    svg: &mut String,
    values: &[f32],
    title: &str,
    subtitle: &str,
    display_range: (f32, f32),
    index: usize,
) -> Result<()> {
    let panel_columns = usize::try_from(PANEL_COLUMNS).context("panel columns exceed usize")?;
    let column = index % panel_columns;
    let row = index / panel_columns;
    let offset_x = PANEL_WIDTH
        .checked_mul(u32::try_from(column).context("panel column exceeds u32")?)
        .context("figure width overflows")?;
    let offset_y = PANEL_HEIGHT
        .checked_mul(u32::try_from(row).context("panel row exceeds u32")?)
        .context("figure height overflows")?;
    let [depth, height, width] = DIMS;
    let display_slice = depth / 2;
    let slice_offset = display_slice
        .checked_mul(height)
        .and_then(|offset| offset.checked_mul(width))
        .context("display slice offset overflows")?;
    let cell_x = f64::from(u32::try_from(DISPLAY_SIZE)?) / f64::from(u32::try_from(width)?);
    let cell_y = f64::from(u32::try_from(DISPLAY_SIZE)?) / f64::from(u32::try_from(height)?);
    writeln!(svg, "<g transform=\"translate({offset_x},{offset_y})\">")?;
    writeln!(
        svg,
        "<rect x=\"8\" y=\"8\" width=\"240\" height=\"264\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"16\" y=\"27\" class=\"title\">{title}</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"16\" y=\"43\" class=\"subtitle\">{subtitle}</text>"
    )?;
    for y in 0..height {
        for x in 0..width {
            let value = *values
                .get(slice_offset + y * width + x)
                .context("panel shape does not match phantom")?;
            let intensity = gray(value, display_range.0, display_range.1);
            let x0 = 24.0 + f64::from(u32::try_from(x)?) * cell_x;
            let y0 = 54.0 + f64::from(u32::try_from(y)?) * cell_y;
            writeln!(
                svg,
                "<rect x=\"{x0:.3}\" y=\"{y0:.3}\" width=\"{cell_x:.3}\" height=\"{cell_y:.3}\" fill=\"rgb({intensity},{intensity},{intensity})\"/>"
            )?;
        }
    }
    writeln!(
        svg,
        "<text x=\"16\" y=\"276\" class=\"note\">center z={display_slice}; display [{:.3}, {:.3}]</text>",
        display_range.0, display_range.1
    )?;
    svg.push_str("</g>\n");
    Ok(())
}

fn draw_contract_panel(svg: &mut String, index: usize) -> Result<()> {
    let panel_columns = usize::try_from(PANEL_COLUMNS).context("panel columns exceed usize")?;
    let column = index % panel_columns;
    let row = index / panel_columns;
    let offset_x = PANEL_WIDTH * u32::try_from(column)?;
    let offset_y = PANEL_HEIGHT * u32::try_from(row)?;
    writeln!(svg, "<g transform=\"translate({offset_x},{offset_y})\">")?;
    writeln!(
        svg,
        "<rect x=\"8\" y=\"8\" width=\"240\" height=\"264\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"16\" y=\"31\" class=\"title\">Pipeline contract</text>"
    )?;
    for (line, text) in [
        (62, "all stages preserve [z, y, x] geometry"),
        (92, "sigmoid: bounded intensity remap"),
        (122, "threshold: suppress outside [0, .58]"),
        (152, "morphology: binary topology changes"),
        (182, "gradient: physical intensity slope"),
        (212, "diffusion: denoise with edge stopping"),
        (242, "change panels: absolute output deltas"),
    ] {
        writeln!(
            svg,
            "<text x=\"16\" y=\"{line}\" class=\"contract\">{text}</text>"
        )?;
    }
    writeln!(
        svg,
        "<text x=\"16\" y=\"264\" class=\"note\">three slices; center slice shown; native API</text>"
    )?;
    svg.push_str("</g>\n");
    Ok(())
}

fn write_figure(
    path: &Path,
    input: &[f32],
    sigmoid: &[f32],
    threshold: &[f32],
    gradient: &[f32],
    opened: &[f32],
    closed: &[f32],
    grayscale_opened: &[f32],
    diffused: &[f32],
    curvature_flow: &[f32],
) -> Result<()> {
    let gradient_range = min_max(gradient)?;
    let change: Vec<f32> = input
        .iter()
        .zip(diffused.iter())
        .map(|(&source, &filtered)| (filtered - source).abs())
        .collect();
    let change_range = min_max(&change)?;
    let curvature_change: Vec<f32> = input
        .iter()
        .zip(curvature_flow.iter())
        .map(|(&source, &filtered)| (filtered - source).abs())
        .collect();
    let curvature_change_range = min_max(&curvature_change)?;
    let figure_width = PANEL_WIDTH * PANEL_COLUMNS;
    let figure_height = PANEL_HEIGHT * PANEL_ROWS;
    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {figure_width} {figure_height}\">"
    )?;
    writeln!(
        svg,
        "<rect width=\"{figure_width}\" height=\"{figure_height}\" fill=\"#ffffff\"/>"
    )?;
    svg.push_str("<style>.title{font:600 14px sans-serif;fill:#172033}.subtitle{font:11px sans-serif;fill:#475569}.note{font:11px sans-serif;fill:#475569}.contract{font:11px sans-serif;fill:#172033}.panel{fill:#ffffff;stroke:#cbd5e1;stroke-width:1}</style>\n");
    draw_image_panel(
        &mut svg,
        input,
        "Input phantom",
        "raw scalar values",
        (0.0, 1.0),
        0,
    )?;
    draw_image_panel(
        &mut svg,
        sigmoid,
        "Sigmoid",
        "alpha .42; beta .10",
        (0.0, 1.0),
        1,
    )?;
    draw_image_panel(
        &mut svg,
        threshold,
        "Threshold suppression",
        "retain [0.00, 0.58]",
        (0.0, 1.0),
        2,
    )?;
    draw_image_panel(
        &mut svg,
        gradient,
        "Gradient magnitude",
        "central differences",
        (0.0, gradient_range.1),
        3,
    )?;
    draw_image_panel(
        &mut svg,
        opened,
        "Binary opening",
        "erode → dilate; r = 1",
        (0.0, 1.0),
        4,
    )?;
    draw_image_panel(
        &mut svg,
        closed,
        "Binary closing",
        "dilate → erode; r = 1",
        (0.0, 1.0),
        5,
    )?;
    draw_image_panel(
        &mut svg,
        grayscale_opened,
        "Grayscale opening",
        "local min → max; r = 2",
        (0.0, 1.0),
        6,
    )?;
    draw_image_panel(
        &mut svg,
        diffused,
        "Perona–Malik",
        "12 steps; K = .08",
        (0.0, 1.0),
        7,
    )?;
    draw_image_panel(
        &mut svg,
        &change,
        "Diffusion change",
        "absolute value delta",
        (0.0, change_range.1),
        8,
    )?;
    draw_image_panel(
        &mut svg,
        curvature_flow,
        "Curvature flow",
        "5 steps; dt = .0625",
        (0.0, 1.0),
        9,
    )?;
    draw_image_panel(
        &mut svg,
        &curvature_change,
        "Curvature change",
        "absolute value delta",
        (0.0, curvature_change_range.1),
        10,
    )?;
    draw_contract_panel(&mut svg, 11)?;
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
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/processing_pipeline.svg"));
    let backend = SequentialBackend;
    let input_values = phantom()?;
    let input = image(input_values.clone(), &backend)?;
    let sigmoid = SigmoidImageFilter::new(0.42, 0.10, 0.0, 1.0)
        .apply_native(&input, &backend)
        .context("apply sigmoid remap")?;
    let threshold = ThresholdImageFilter::outside(0.0, 0.58, 0.0)
        .apply_native(&sigmoid, &backend)
        .context("apply threshold suppression")?;
    let gradient = GradientMagnitudeFilter::unit()
        .apply_native(&input)
        .context("compute gradient magnitude")?;
    let binary_seed = ritk_filter::BinaryThresholdImageFilter::new(0.62, 1.0, 1.0, 0.0)
        .apply_native(&sigmoid, &backend)
        .context("create binary mask")?;
    let opened = BinaryDilateFilter::new(1)
        .apply_native(
            &BinaryErodeFilter::new(1)
                .apply_native(&binary_seed, &backend)
                .context("erode binary mask")?,
            &backend,
        )
        .context("dilate eroded mask")?;
    let closed = BinaryErodeFilter::new(1)
        .apply_native(
            &BinaryDilateFilter::new(1)
                .apply_native(&binary_seed, &backend)
                .context("dilate binary mask")?,
            &backend,
        )
        .context("erode dilated mask")?;
    let grayscale_opened = GrayscaleOpeningFilter::new(2)
        .apply_native(&sigmoid, &backend)
        .context("apply grayscale opening")?;
    let diffused = DiffusionConfig {
        num_iterations: 12,
        time_step: 0.0625,
        conductance: 0.08,
        ..DiffusionConfig::default()
    }
    .apply_native(&input, &backend)
    .context("apply Perona-Malik diffusion")?;
    let curvature_flow = CurvatureFlowImageFilter::new(CurvatureFlowConfig {
        num_iterations: 5,
        time_step: 0.0625,
    })
    .apply_native(&input, &backend)
    .context("apply curvature flow")?;
    write_figure(
        &output,
        &input_values,
        sigmoid.data_slice()?,
        threshold.data_slice()?,
        gradient.data_slice()?,
        opened.data_slice()?,
        closed.data_slice()?,
        grayscale_opened.data_slice()?,
        diffused.data_slice()?,
        curvature_flow.data_slice()?,
    )?;
    println!("wrote {}", output.display());
    Ok(())
}
