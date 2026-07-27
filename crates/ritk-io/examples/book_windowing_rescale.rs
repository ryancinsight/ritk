//! Generate the CT windowing and rescaling figure used by the RITK mdBook.
//!
//! The example uses the RIRE Patient 001 CT volume and renders the outputs of
//! the native intensity filters with explicit HU windows and output ranges.
//! The histogram is computed from the same axial slice as the image panels so
//! the clinical windows are visible as data-derived intervals rather than
//! unexplained display settings.

use anyhow::{bail, Context, Result};
use coeus_core::SequentialBackend;
use ritk_filter::{IntensityWindowingFilter, RescaleIntensityFilter};
use ritk_io::{format::metaimage::native::MetaImageReader, ImageReader};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

type Backend = SequentialBackend;

const CT_PATH: &str = "test_data/registration/rire/training_001_ct.mha";
const SOFT_TISSUE_WINDOW: (f32, f32) = (-160.0, 240.0);
const LUNG_WINDOW: (f32, f32) = (-1000.0, 400.0);
const CT_DISPLAY_WINDOW: (f32, f32) = (-1000.0, 1000.0);
const PANEL_WIDTH: u32 = 280;
const PANEL_HEIGHT: u32 = 290;
const IMAGE_SIZE: u32 = 160;
const IMAGE_LEFT: u32 = 46;
const IMAGE_TOP: u32 = 52;
const HISTOGRAM_BINS: usize = 64;

fn percentile(values: &[f32], hundredths: usize) -> Result<f32> {
    let mut finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        bail!("cannot compute a percentile from an empty finite sample");
    }
    finite.sort_by(f32::total_cmp);
    let index = finite.len().saturating_mul(hundredths) / 100;
    finite
        .get(index.min(finite.len() - 1))
        .copied()
        .context("percentile index is outside the sorted sample")
}

fn slice(values: &[f32], shape: [usize; 3], z: usize) -> Result<&[f32]> {
    let [depth, height, width] = shape;
    if z >= depth {
        bail!("axial slice {z} is outside volume depth {depth}");
    }
    let plane = height
        .checked_mul(width)
        .context("axial plane size overflows usize")?;
    let start = z
        .checked_mul(plane)
        .context("axial slice offset overflows usize")?;
    values
        .get(start..start + plane)
        .context("image data length does not match its declared shape")
}

fn intensity_to_gray(value: f32, lower: f32, upper: f32) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    f64::from(((value - lower) / (upper - lower)).clamp(0.0, 1.0)) * 255.0
}

struct ImagePanel<'a> {
    values: &'a [f32],
    shape: [usize; 3],
    display_range: (f32, f32),
    title: &'a str,
    subtitle: &'a str,
    note: &'a str,
    offset_x: u32,
    offset_y: u32,
}

fn draw_image_panel(svg: &mut String, panel: ImagePanel<'_>) -> Result<()> {
    let ImagePanel {
        values,
        shape,
        display_range,
        title,
        subtitle,
        note,
        offset_x,
        offset_y,
    } = panel;
    let [_, height, width] = shape;
    let display_size = usize::try_from(IMAGE_SIZE).context("image size exceeds usize")?;
    let (lower, upper) = display_range;
    writeln!(svg, "<g transform=\"translate({offset_x},{offset_y})\">")?;
    writeln!(
        svg,
        "<text x=\"16\" y=\"18\" class=\"title\">{title}</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"16\" y=\"34\" class=\"subtitle\">{subtitle}</text>"
    )?;
    writeln!(svg, "<text x=\"16\" y=\"242\" class=\"note\">{note}</text>")?;
    writeln!(
        svg,
        "<rect x=\"16\" y=\"42\" width=\"248\" height=\"208\" class=\"panel\"/>"
    )?;
    for output_y in 0..display_size {
        let source_y = output_y.saturating_mul(height) / display_size;
        for output_x in 0..display_size {
            let source_x = output_x.saturating_mul(width) / display_size;
            let value = *values
                .get(source_y * width + source_x)
                .context("image panel shape mismatch")?;
            let intensity = intensity_to_gray(value, lower, upper);
            let x0 = f64::from(IMAGE_LEFT) + f64::from(u32::try_from(output_x)?);
            let y0 = f64::from(IMAGE_TOP) + f64::from(u32::try_from(output_y)?);
            writeln!(
                svg,
                "<rect x=\"{x0:.0}\" y=\"{y0:.0}\" width=\"1\" height=\"1\" fill=\"rgb({intensity:.0},{intensity:.0},{intensity:.0})\"/>"
            )?;
        }
    }
    writeln!(svg, "</g>")?;
    Ok(())
}

fn draw_histogram_panel(
    svg: &mut String,
    values: &[f32],
    title: &str,
    offset_x: u32,
    offset_y: u32,
) -> Result<()> {
    let lower = CT_DISPLAY_WINDOW.0;
    let upper = CT_DISPLAY_WINDOW.1;
    let bin_count = f32::from(u16::try_from(HISTOGRAM_BINS)?);
    let bin_width_hu = (upper - lower) / bin_count;
    let mut bins = [0_usize; HISTOGRAM_BINS];
    for (index, bin) in bins.iter_mut().enumerate() {
        let index_f32 = f32::from(u16::try_from(index)?);
        let bin_lower = lower + index_f32 * bin_width_hu;
        let bin_upper = bin_lower + bin_width_hu;
        *bin = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .filter(|value| {
                if index + 1 == HISTOGRAM_BINS {
                    *value >= bin_lower && *value <= bin_upper
                } else {
                    *value >= bin_lower && *value < bin_upper
                }
            })
            .count();
    }
    let maximum = bins
        .iter()
        .copied()
        .max()
        .context("histogram has no bins")?
        .max(1);
    let chart_left = 28.0_f64;
    let chart_top = 52.0_f64;
    let chart_width = 232.0_f64;
    let chart_height = 184.0_f64;
    let bin_width = chart_width / f64::from(u32::try_from(HISTOGRAM_BINS)?);
    writeln!(svg, "<g transform=\"translate({offset_x},{offset_y})\">")?;
    writeln!(
        svg,
        "<text x=\"16\" y=\"18\" class=\"title\">{title}</text>"
    )?;
    writeln!(svg, "<text x=\"16\" y=\"34\" class=\"subtitle\">same axial slice; clipped to [-1000, 1000] HU</text>")?;
    writeln!(
        svg,
        "<rect x=\"16\" y=\"42\" width=\"248\" height=\"208\" class=\"panel\"/>"
    )?;
    for (index, &count) in bins.iter().enumerate() {
        let x = chart_left + f64::from(u32::try_from(index)?) * bin_width;
        let height =
            f64::from(u32::try_from(count)?) / f64::from(u32::try_from(maximum)?) * chart_height;
        let y = chart_top + chart_height - height;
        writeln!(svg, "<rect x=\"{x:.3}\" y=\"{y:.3}\" width=\"{:.3}\" height=\"{height:.3}\" fill=\"#64748b\"/>", (bin_width - 0.5).max(0.0))?;
    }
    writeln!(
        svg,
        "<line x1=\"{chart_left}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"axis\"/>",
        chart_top + chart_height,
        chart_left + chart_width,
        chart_top + chart_height
    )?;
    for &(window, color, label) in &[
        (SOFT_TISSUE_WINDOW, "#dc2626", "soft tissue"),
        (LUNG_WINDOW, "#2563eb", "lung"),
    ] {
        let x0 = chart_left + f64::from((window.0 - lower) / (upper - lower)) * chart_width;
        let x1 = chart_left + f64::from((window.1 - lower) / (upper - lower)) * chart_width;
        writeln!(svg, "<line x1=\"{x0:.3}\" y1=\"{chart_top}\" x2=\"{x0:.3}\" y2=\"{}\" stroke=\"{color}\" stroke-width=\"2\"/>", chart_top + chart_height)?;
        writeln!(svg, "<line x1=\"{x1:.3}\" y1=\"{chart_top}\" x2=\"{x1:.3}\" y2=\"{}\" stroke=\"{color}\" stroke-width=\"2\"/>", chart_top + chart_height)?;
        writeln!(
            svg,
            "<text x=\"{:.3}\" y=\"{}\" class=\"legend\" fill=\"{color}\">{label}</text>",
            (x0 + x1) * 0.5 - 24.0,
            chart_top + chart_height + 22.0
        )?;
    }
    writeln!(
        svg,
        "<text x=\"{chart_left}\" y=\"{}\" class=\"axis-label\">-1000 HU</text>",
        chart_top + chart_height + 38.0
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"{}\" class=\"axis-label\">1000 HU</text>",
        chart_left + chart_width - 48.0,
        chart_top + chart_height + 38.0
    )?;
    writeln!(svg, "</g>")?;
    Ok(())
}

fn draw_contract_panel(svg: &mut String, offset_x: u32, offset_y: u32) -> Result<()> {
    writeln!(svg, "<g transform=\"translate({offset_x},{offset_y})\">")?;
    writeln!(
        svg,
        "<text x=\"16\" y=\"18\" class=\"title\">Filter contract</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"16\" y=\"34\" class=\"subtitle\">same voxels; different intensity maps</text>"
    )?;
    writeln!(
        svg,
        "<rect x=\"16\" y=\"42\" width=\"248\" height=\"208\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"20\" y=\"76\" class=\"contract\">window: clamp HU, then map to [0, 1]</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"20\" y=\"102\" class=\"contract\">below lower bound → 0</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"20\" y=\"126\" class=\"contract\">above upper bound → 1</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"20\" y=\"174\" class=\"contract\">rescale: global min/max → [0, 255]</text>"
    )?;
    writeln!(svg, "<rect x=\"20\" y=\"202\" width=\"44\" height=\"16\" fill=\"#334155\"/><text x=\"72\" y=\"215\" class=\"legend\">output minimum</text>")?;
    writeln!(svg, "<rect x=\"20\" y=\"228\" width=\"44\" height=\"16\" fill=\"#e2e8f0\"/><text x=\"72\" y=\"241\" class=\"legend\">output maximum</text>")?;
    writeln!(svg, "</g>")?;
    Ok(())
}

fn write_figure(
    path: &Path,
    input: &[f32],
    shape: [usize; 3],
    soft: &[f32],
    lung: &[f32],
    rescaled: &[f32],
) -> Result<()> {
    let axial_slice = shape[0] / 2;
    let input_slice = slice(input, shape, axial_slice)?;
    let soft_slice = slice(soft, shape, axial_slice)?;
    let lung_slice = slice(lung, shape, axial_slice)?;
    let rescaled_slice = slice(rescaled, shape, axial_slice)?;
    let (input_lower, input_upper) = min_max(input)?;
    let soft_clipped = clipped_percentage(input_slice, SOFT_TISSUE_WINDOW)?;
    let lung_clipped = clipped_percentage(input_slice, LUNG_WINDOW)?;
    let global_subtitle =
        format!("RescaleIntensityFilter [{input_lower:.0}, {input_upper:.0}] HU → [0, 255]");
    let soft_note =
        format!("IntensityWindowingFilter; {soft_clipped:.1}% of source voxels saturated");
    let lung_note =
        format!("IntensityWindowingFilter; {lung_clipped:.1}% of source voxels saturated");
    let figure_width = PANEL_WIDTH
        .checked_mul(3)
        .context("figure width overflows u32")?;
    let figure_height = PANEL_HEIGHT
        .checked_mul(2)
        .context("figure height overflows u32")?;
    let mut svg = String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 ");
    writeln!(svg, "{figure_width} {figure_height}\">")?;
    writeln!(svg, "<rect width=\"{figure_width}\" height=\"{figure_height}\" fill=\"#ffffff\"/>\n<style>.title{{font:600 15px sans-serif;fill:#172033}}.subtitle{{font:12px sans-serif;fill:#475569}}.note{{font:11px sans-serif;fill:#475569}}.panel{{fill:#ffffff;stroke:#cbd5e1;stroke-width:1}}.axis{{stroke:#172033;stroke-width:1}}.axis-label,.legend{{font:11px sans-serif;fill:#172033}}.contract{{font:12px sans-serif;fill:#172033}}</style>")?;
    draw_image_panel(
        &mut svg,
        ImagePanel {
            values: input_slice,
            shape: [1, shape[1], shape[2]],
            display_range: CT_DISPLAY_WINDOW,
            title: "Input CT",
            subtitle: "display window [-1000, 1000] HU",
            note: "fixed display mapping for comparison",
            offset_x: 0,
            offset_y: 0,
        },
    )?;
    draw_image_panel(
        &mut svg,
        ImagePanel {
            values: soft_slice,
            shape: [1, shape[1], shape[2]],
            display_range: (0.0, 1.0),
            title: "Soft-tissue window",
            subtitle: "HU [-160, 240] → output [0, 1]",
            note: &soft_note,
            offset_x: PANEL_WIDTH,
            offset_y: 0,
        },
    )?;
    draw_image_panel(
        &mut svg,
        ImagePanel {
            values: lung_slice,
            shape: [1, shape[1], shape[2]],
            display_range: (0.0, 1.0),
            title: "Lung window",
            subtitle: "HU [-1000, 400] → output [0, 1]",
            note: &lung_note,
            offset_x: PANEL_WIDTH * 2,
            offset_y: 0,
        },
    )?;
    draw_image_panel(
        &mut svg,
        ImagePanel {
            values: rescaled_slice,
            shape: [1, shape[1], shape[2]],
            display_range: (0.0, 255.0),
            title: "Global rescale",
            subtitle: &global_subtitle,
            note: "RescaleIntensityFilter; same source geometry",
            offset_x: 0,
            offset_y: PANEL_HEIGHT,
        },
    )?;
    draw_histogram_panel(
        &mut svg,
        input_slice,
        "Input distribution",
        PANEL_WIDTH,
        PANEL_HEIGHT,
    )?;
    draw_contract_panel(&mut svg, PANEL_WIDTH * 2, PANEL_HEIGHT)?;
    svg.push_str("</svg>\n");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg)
        .with_context(|| format!("write windowing figure {}", path.display()))?;
    Ok(())
}

fn min_max(values: &[f32]) -> Result<(f32, f32)> {
    let mut iter = values.iter().copied().filter(|value| value.is_finite());
    let first = iter.next().context("image contains no finite values")?;
    Ok(iter.fold((first, first), |(lower, upper), value| {
        (lower.min(value), upper.max(value))
    }))
}

fn clipped_percentage(values: &[f32], window: (f32, f32)) -> Result<f64> {
    let finite_count = values.iter().filter(|value| value.is_finite()).count();
    let clipped_count = values
        .iter()
        .filter(|value| value.is_finite())
        .filter(|value| **value < window.0 || **value > window.1)
        .count();
    let finite_count = u32::try_from(finite_count).context("finite voxel count exceeds u32")?;
    let clipped_count = u32::try_from(clipped_count).context("clipped voxel count exceeds u32")?;
    if finite_count == 0 {
        bail!("cannot compute clipping percentage from an empty finite sample");
    }
    Ok(f64::from(clipped_count) * 100.0 / f64::from(finite_count))
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/windowing_rescale.svg"));
    let backend = Backend::default();
    let input = MetaImageReader::new(Backend::default())
        .read(CT_PATH)
        .with_context(|| format!("read RIRE CT volume {CT_PATH}"))?;
    let input_values = input.data_slice()?.to_vec();
    let soft_filter =
        IntensityWindowingFilter::new(SOFT_TISSUE_WINDOW.0, SOFT_TISSUE_WINDOW.1, 0.0, 1.0);
    let lung_filter = IntensityWindowingFilter::new(LUNG_WINDOW.0, LUNG_WINDOW.1, 0.0, 1.0);
    let soft = soft_filter
        .apply_native(&input, &backend)
        .context("apply soft-tissue window")?;
    let lung = lung_filter
        .apply_native(&input, &backend)
        .context("apply lung window")?;
    let rescaled = RescaleIntensityFilter::new(0.0, 255.0)
        .apply_native(&input, &backend)
        .context("rescale CT intensities to [0, 255]")?;
    if soft.shape() != input.shape()
        || lung.shape() != input.shape()
        || rescaled.shape() != input.shape()
    {
        bail!("intensity filters changed CT geometry");
    }
    let soft_values = soft.data_slice()?.to_vec();
    let lung_values = lung.data_slice()?.to_vec();
    let rescaled_values = rescaled.data_slice()?.to_vec();
    for (name, values, expected) in [
        ("soft-tissue window", &soft_values, (0.0, 1.0)),
        ("lung window", &lung_values, (0.0, 1.0)),
        ("global rescale", &rescaled_values, (0.0, 255.0)),
    ] {
        let (lower, upper) = min_max(values)?;
        if lower < expected.0 || upper > expected.1 {
            bail!(
                "{name} output range [{lower}, {upper}] exceeds [{}, {}]",
                expected.0,
                expected.1
            );
        }
    }
    write_figure(
        &output,
        &input_values,
        input.shape(),
        &soft_values,
        &lung_values,
        &rescaled_values,
    )?;
    let axial_slice = slice(&input_values, input.shape(), input.shape()[0] / 2)?;
    let (input_lower, input_upper) = min_max(axial_slice)?;
    let p02 = percentile(axial_slice, 2)?;
    let p98 = percentile(axial_slice, 98)?;
    println!("wrote {} (RIRE CT axial slice {}; source min/max {:.1}/{:.1} HU; slice p2/p98 {:.1}/{:.1} HU)", output.display(), input.shape()[0] / 2, input_lower, input_upper, p02, p98);
    Ok(())
}
