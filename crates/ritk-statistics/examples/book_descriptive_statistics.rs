//! Generate the descriptive-statistics figure used by the RITK mdBook.
//!
//! The example creates one deterministic medical-style intensity field,
//! compares full-image and masked-foreground populations, verifies RITK's
//! results against an independent sorted reference, and renders the source,
//! mask, distributions, quartiles, mean, and median to one SVG.
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

use anyhow::{bail, Context, Result};
use ritk_statistics::{
    compute_statistics_from_slice, histogram_from_slice, masked_statistics_from_slices,
    ImageStatistics,
};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;
const HISTOGRAM_MIN: f32 = 0.0;
const HISTOGRAM_MAX: f32 = 240.0;
const HISTOGRAM_BINS: usize = 24;
const FIGURE_WIDTH: u32 = 1180;
const FIGURE_HEIGHT: u32 = 650;

struct Phantom {
    intensities: Vec<f32>,
    mask: Vec<f32>,
}

#[derive(Debug)]
struct ReferenceStatistics {
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
    percentiles: [f32; 3],
}

fn phantom() -> Result<Phantom> {
    let sample_count = WIDTH
        .checked_mul(HEIGHT)
        .context("phantom dimensions overflow usize")?;
    let mut intensities = Vec::with_capacity(sample_count);
    let mut mask = Vec::with_capacity(sample_count);

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let x_coordinate = i32::try_from(x).context("x coordinate exceeds i32")?;
            let y_coordinate = i32::try_from(y).context("y coordinate exceeds i32")?;
            let dx = x_coordinate - 32;
            let dy = y_coordinate - 24;
            let foreground = dx * dx * 9 + dy * dy * 16 <= 22 * 22 * 9;
            let lesion_dx = x_coordinate - 41;
            let lesion_dy = y_coordinate - 19;
            let lesion = lesion_dx * lesion_dx + lesion_dy * lesion_dy <= 5 * 5;
            let texture = ((7 * x_coordinate + 11 * y_coordinate) % 13) as f32 - 6.0;

            let intensity = if lesion && foreground {
                205.0 + texture
            } else if foreground && x_coordinate >= 32 {
                151.0 + texture + 0.18 * dy as f32
            } else if foreground {
                82.0 + texture - 0.12 * dy as f32
            } else {
                17.0 + texture * 0.45
            };
            intensities.push(intensity);
            mask.push(if foreground { 1.0 } else { 0.0 });
        }
    }

    Ok(Phantom { intensities, mask })
}

fn reference_statistics(values: &[f32], ddof: usize) -> Result<ReferenceStatistics> {
    if values.is_empty() || ddof >= values.len() {
        bail!("reference statistics require N > ddof");
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("reference statistics require finite samples");
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let sample_count = sorted.len();
    let sum = sorted.iter().map(|&value| f64::from(value)).sum::<f64>();
    let mean_wide = sum / sample_count as f64;
    let sum_squared_deviation = sorted
        .iter()
        .map(|&value| {
            let deviation = f64::from(value) - mean_wide;
            deviation * deviation
        })
        .sum::<f64>();
    let std = (sum_squared_deviation / (sample_count - ddof) as f64).sqrt() as f32;
    let upper_quartile_rank = (sample_count / 4) * 3 + ((sample_count % 4) * 3) / 4;

    Ok(ReferenceStatistics {
        min: sorted[0],
        max: sorted[sample_count - 1],
        mean: mean_wide as f32,
        std,
        percentiles: [
            sorted[sample_count / 4],
            sorted[sample_count / 2],
            sorted[upper_quartile_rank],
        ],
    })
}

fn verify_statistics(
    label: &str,
    actual: &ImageStatistics,
    reference: &ReferenceStatistics,
) -> Result<()> {
    let tolerance = 32.0 * f32::EPSILON;
    if actual.min != reference.min
        || actual.max != reference.max
        || actual.percentiles != reference.percentiles
        || (actual.mean - reference.mean).abs() > tolerance * reference.mean.abs().max(1.0)
        || (actual.std - reference.std).abs() > tolerance * reference.std.abs().max(1.0)
    {
        bail!("{label} statistics disagree with the independent sorted reference");
    }
    Ok(())
}

fn usize_to_f64(value: usize, label: &str) -> Result<f64> {
    let narrowed = u32::try_from(value).with_context(|| format!("{label} exceeds u32"))?;
    Ok(f64::from(narrowed))
}

fn draw_image_panel(svg: &mut String, values: &[f32], origin_x: f64) -> Result<()> {
    let cell = 4.6_f64;
    writeln!(
        svg,
        "<g><rect x=\"{origin_x}\" y=\"55\" width=\"316\" height=\"296\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"82\" class=\"panel-title\">Intensity field</text>",
        origin_x + 18.0
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"101\" class=\"caption\">background + two tissues + lesion</text>",
        origin_x + 18.0
    )?;

    for (index, &value) in values.iter().enumerate() {
        let x = index % WIDTH;
        let y = index / WIDTH;
        let normalized =
            ((value - HISTOGRAM_MIN) / (HISTOGRAM_MAX - HISTOGRAM_MIN)).clamp(0.0, 1.0);
        let lightness = 10.0 + 82.0 * f64::from(normalized);
        let pixel_x = origin_x + 11.0 + usize_to_f64(x, "pixel x")? * cell;
        let pixel_y = 112.0 + usize_to_f64(y, "pixel y")? * cell;
        writeln!(
            svg,
            "<rect x=\"{pixel_x:.2}\" y=\"{pixel_y:.2}\" width=\"{cell:.2}\" height=\"{cell:.2}\" fill=\"hsl(210 18% {lightness:.2}%)\"/>"
        )?;
    }
    writeln!(
        svg,
        "<text x=\"{}\" y=\"340\" class=\"caption\">same [0, 240] display range as histogram</text></g>",
        origin_x + 18.0
    )?;
    Ok(())
}

fn draw_mask_panel(svg: &mut String, mask: &[f32], origin_x: f64) -> Result<()> {
    let cell = 4.6_f64;
    writeln!(
        svg,
        "<g><rect x=\"{origin_x}\" y=\"55\" width=\"316\" height=\"296\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"82\" class=\"panel-title\">Foreground mask</text>",
        origin_x + 18.0
    )?;
    writeln!(
        svg,
        "<text x=\"{}\" y=\"101\" class=\"caption\">orange samples define the masked population</text>",
        origin_x + 18.0
    )?;

    for (index, &value) in mask.iter().enumerate() {
        let x = index % WIDTH;
        let y = index / WIDTH;
        let pixel_x = origin_x + 11.0 + usize_to_f64(x, "mask x")? * cell;
        let pixel_y = 112.0 + usize_to_f64(y, "mask y")? * cell;
        let fill = if value > 0.5 { "#f97316" } else { "#172033" };
        writeln!(
            svg,
            "<rect x=\"{pixel_x:.2}\" y=\"{pixel_y:.2}\" width=\"{cell:.2}\" height=\"{cell:.2}\" fill=\"{fill}\"/>"
        )?;
    }
    writeln!(
        svg,
        "<text x=\"{}\" y=\"340\" class=\"caption\">masked statistics exclude dark background</text></g>",
        origin_x + 18.0
    )?;
    Ok(())
}

fn intensity_x(value: f32, plot_x: f64, plot_width: f64) -> f64 {
    let normalized = ((value - HISTOGRAM_MIN) / (HISTOGRAM_MAX - HISTOGRAM_MIN)).clamp(0.0, 1.0);
    plot_x + f64::from(normalized) * plot_width
}

fn draw_distribution_panel(
    svg: &mut String,
    full: &ImageStatistics,
    masked: &ImageStatistics,
    full_counts: &[usize],
    masked_counts: &[usize],
    full_total: usize,
    masked_total: usize,
) -> Result<()> {
    let panel_x = 680.0_f64;
    let plot_x = 720.0_f64;
    let plot_y = 116.0_f64;
    let plot_width = 420.0_f64;
    let plot_height = 205.0_f64;
    writeln!(
        svg,
        "<g><rect x=\"{panel_x}\" y=\"55\" width=\"480\" height=\"296\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"700\" y=\"82\" class=\"panel-title\">Full image versus masked distribution</text>"
    )?;
    writeln!(
        svg,
        "<rect x=\"700\" y=\"92\" width=\"12\" height=\"12\" fill=\"#2563eb\" opacity=\"0.72\"/><text x=\"718\" y=\"103\" class=\"legend\">full image</text>"
    )?;
    writeln!(
        svg,
        "<rect x=\"805\" y=\"92\" width=\"12\" height=\"12\" fill=\"#f97316\" opacity=\"0.72\"/><text x=\"823\" y=\"103\" class=\"legend\">masked foreground</text>"
    )?;
    writeln!(
        svg,
        "<line x1=\"{plot_x}\" y1=\"{plot_y}\" x2=\"{plot_x}\" y2=\"{}\" class=\"axis\"/><line x1=\"{plot_x}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"axis\"/>",
        plot_y + plot_height,
        plot_y + plot_height,
        plot_x + plot_width,
        plot_y + plot_height
    )?;

    let bin_width = plot_width / usize_to_f64(HISTOGRAM_BINS, "histogram bins")?;
    let maximum_share = full_counts
        .iter()
        .map(|&count| count as f64 / full_total as f64)
        .chain(
            masked_counts
                .iter()
                .map(|&count| count as f64 / masked_total as f64),
        )
        .fold(0.0_f64, f64::max);
    if maximum_share <= 0.0 {
        bail!("histogram has no visible mass");
    }

    for (index, (&full_count, &masked_count)) in full_counts.iter().zip(masked_counts).enumerate() {
        let x = plot_x + usize_to_f64(index, "histogram index")? * bin_width;
        let full_height = (full_count as f64 / full_total as f64) / maximum_share * plot_height;
        let masked_height =
            (masked_count as f64 / masked_total as f64) / maximum_share * plot_height;
        writeln!(
            svg,
            "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{full_height:.2}\" fill=\"#2563eb\" opacity=\"0.62\"/>",
            x + 0.8,
            plot_y + plot_height - full_height,
            bin_width - 1.6
        )?;
        writeln!(
            svg,
            "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{masked_height:.2}\" fill=\"#f97316\" opacity=\"0.62\"/>",
            x + bin_width * 0.27,
            plot_y + plot_height - masked_height,
            bin_width * 0.46
        )?;
    }

    for tick in [0.0_f32, 60.0, 120.0, 180.0, 240.0] {
        let x = intensity_x(tick, plot_x, plot_width);
        writeln!(
            svg,
            "<line x1=\"{x:.2}\" y1=\"{}\" x2=\"{x:.2}\" y2=\"{}\" class=\"tick\"/><text x=\"{x:.2}\" y=\"339\" text-anchor=\"middle\" class=\"axis-label\">{tick:.0}</text>",
            plot_y + plot_height,
            plot_y + plot_height + 5.0
        )?;
    }
    writeln!(
        svg,
        "<text x=\"690\" y=\"225\" transform=\"rotate(-90 690 225)\" class=\"axis-label\">share of selected samples</text><text x=\"930\" y=\"345\" text-anchor=\"middle\" class=\"axis-label\">intensity</text>"
    )?;

    let full_q1 = intensity_x(full.percentiles[0], plot_x, plot_width);
    let full_q3 = intensity_x(full.percentiles[2], plot_x, plot_width);
    let masked_q1 = intensity_x(masked.percentiles[0], plot_x, plot_width);
    let masked_q3 = intensity_x(masked.percentiles[2], plot_x, plot_width);
    writeln!(
        svg,
        "<rect x=\"{full_q1:.2}\" y=\"308\" width=\"{:.2}\" height=\"5\" fill=\"#2563eb\"/><rect x=\"{masked_q1:.2}\" y=\"315\" width=\"{:.2}\" height=\"5\" fill=\"#f97316\"/>",
        full_q3 - full_q1,
        masked_q3 - masked_q1
    )?;
    for (statistics, color) in [(full, "#1d4ed8"), (masked, "#c2410c")] {
        let mean_x = intensity_x(statistics.mean, plot_x, plot_width);
        let median_x = intensity_x(statistics.percentiles[1], plot_x, plot_width);
        writeln!(
            svg,
            "<line x1=\"{mean_x:.2}\" y1=\"112\" x2=\"{mean_x:.2}\" y2=\"321\" stroke=\"{color}\" stroke-width=\"2\"/><line x1=\"{median_x:.2}\" y1=\"112\" x2=\"{median_x:.2}\" y2=\"321\" stroke=\"{color}\" stroke-width=\"2\" stroke-dasharray=\"5 3\"/>"
        )?;
    }
    writeln!(svg, "</g>")?;
    Ok(())
}

fn draw_statistics_table(
    svg: &mut String,
    full: &ImageStatistics,
    masked: &ImageStatistics,
    full_count: usize,
    masked_count: usize,
) -> Result<()> {
    writeln!(
        svg,
        "<rect x=\"20\" y=\"382\" width=\"1140\" height=\"238\" class=\"panel\"/><text x=\"42\" y=\"415\" class=\"panel-title\">The mask changes the population, not merely the display</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"42\" y=\"441\" class=\"table-head\">Population</text><text x=\"245\" y=\"441\" class=\"table-head\">N</text><text x=\"335\" y=\"441\" class=\"table-head\">Min</text><text x=\"425\" y=\"441\" class=\"table-head\">Q1</text><text x=\"515\" y=\"441\" class=\"table-head\">Median</text><text x=\"625\" y=\"441\" class=\"table-head\">Mean</text><text x=\"735\" y=\"441\" class=\"table-head\">Q3</text><text x=\"825\" y=\"441\" class=\"table-head\">Max</text><text x=\"915\" y=\"441\" class=\"table-head\">Std</text>"
    )?;
    for (row, label, color, count, statistics) in [
        (0.0, "Full image", "#2563eb", full_count, full),
        (48.0, "Masked foreground", "#f97316", masked_count, masked),
    ] {
        let y = 474.0 + row;
        writeln!(
            svg,
            "<circle cx=\"51\" cy=\"{}\" r=\"6\" fill=\"{color}\"/><text x=\"66\" y=\"{y}\" class=\"table-value\">{label}</text><text x=\"245\" y=\"{y}\" class=\"table-value\">{count}</text><text x=\"335\" y=\"{y}\" class=\"table-value\">{:.1}</text><text x=\"425\" y=\"{y}\" class=\"table-value\">{:.1}</text><text x=\"515\" y=\"{y}\" class=\"table-value\">{:.1}</text><text x=\"625\" y=\"{y}\" class=\"table-value\">{:.1}</text><text x=\"735\" y=\"{y}\" class=\"table-value\">{:.1}</text><text x=\"825\" y=\"{y}\" class=\"table-value\">{:.1}</text><text x=\"915\" y=\"{y}\" class=\"table-value\">{:.1}</text>",
            y - 5.0,
            statistics.min,
            statistics.percentiles[0],
            statistics.percentiles[1],
            statistics.mean,
            statistics.percentiles[2],
            statistics.max,
            statistics.std
        )?;
    }
    writeln!(
        svg,
        "<line x1=\"42\" y1=\"542\" x2=\"1138\" y2=\"542\" stroke=\"#cbd5e1\"/><text x=\"42\" y=\"570\" class=\"callout\">Blue solid = mean · blue dotted = median · blue band = interquartile range</text><text x=\"42\" y=\"596\" class=\"callout orange\">Orange solid = mean · orange dotted = median · orange band = interquartile range</text>"
    )?;
    Ok(())
}

fn write_figure(path: &Path, phantom: &Phantom) -> Result<()> {
    let full = compute_statistics_from_slice(&phantom.intensities, 0)?;
    let masked = masked_statistics_from_slices(&phantom.intensities, &phantom.mask, 0)?;
    let foreground: Vec<f32> = phantom
        .intensities
        .iter()
        .zip(&phantom.mask)
        .filter_map(|(&value, &mask)| (mask > 0.5).then_some(value))
        .collect();
    let full_reference = reference_statistics(&phantom.intensities, 0)?;
    let masked_reference = reference_statistics(&foreground, 0)?;
    verify_statistics("full-image", &full, &full_reference)?;
    verify_statistics("masked", &masked, &masked_reference)?;

    let full_histogram = histogram_from_slice(
        &phantom.intensities,
        HISTOGRAM_MIN,
        HISTOGRAM_MAX,
        HISTOGRAM_BINS,
    )?;
    let masked_histogram =
        histogram_from_slice(&foreground, HISTOGRAM_MIN, HISTOGRAM_MAX, HISTOGRAM_BINS)?;
    if full_histogram.total() != phantom.intensities.len()
        || masked_histogram.total() != foreground.len()
    {
        bail!("histogram totals do not match their source populations");
    }
    if masked.mean - full.mean < 45.0 || masked.percentiles[1] - full.percentiles[1] < 45.0 {
        bail!("phantom does not make full and masked populations visually distinct");
    }

    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {FIGURE_WIDTH} {FIGURE_HEIGHT}\">"
    )?;
    writeln!(
        svg,
        "<rect width=\"{FIGURE_WIDTH}\" height=\"{FIGURE_HEIGHT}\" fill=\"#f8fafc\"/><style>.panel{{fill:#fff;stroke:#cbd5e1;stroke-width:1}}.panel-title{{font:600 17px sans-serif;fill:#172033}}.caption,.legend,.axis-label{{font:12px sans-serif;fill:#475569}}.axis{{stroke:#334155;stroke-width:1.2}}.tick{{stroke:#64748b}}.table-head{{font:600 13px sans-serif;fill:#475569}}.table-value{{font:14px sans-serif;fill:#172033}}.callout{{font:13px sans-serif;fill:#1d4ed8}}.orange{{fill:#c2410c}}</style>"
    )?;
    draw_image_panel(&mut svg, &phantom.intensities, 20.0)?;
    draw_mask_panel(&mut svg, &phantom.mask, 350.0)?;
    draw_distribution_panel(
        &mut svg,
        &full,
        &masked,
        &full_histogram.counts,
        &masked_histogram.counts,
        phantom.intensities.len(),
        foreground.len(),
    )?;
    draw_statistics_table(
        &mut svg,
        &full,
        &masked,
        phantom.intensities.len(),
        foreground.len(),
    )?;
    writeln!(svg, "</svg>")?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg).with_context(|| format!("write {}", path.display()))?;
    println!(
        "wrote {} (full mean {:.2}, masked mean {:.2}, full median {:.2}, masked median {:.2})",
        path.display(),
        full.mean,
        masked.mean,
        full.percentiles[1],
        masked.percentiles[1]
    );
    Ok(())
}

fn output_path() -> PathBuf {
    std::env::args_os().nth(1).map_or_else(
        || PathBuf::from("docs/book/figures/descriptive_statistics.svg"),
        PathBuf::from,
    )
}

fn main() -> Result<()> {
    write_figure(&output_path(), &phantom()?)
}
