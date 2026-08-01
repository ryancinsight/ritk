//! Generate the diffusion-MRI and tractography figure used by the RITK book.
//!
//! The example synthesizes a known single-fiber diffusion tensor, estimates
//! its analytical Q-ball ODF through the public API, and tracks a curved
//! direction field into Gaia polylines. Assertions bind the rendered result to
//! the known axis and tract boundary rather than treating a plausible picture
//! as correctness evidence.

use anyhow::{Context, Result, bail};
use ritk_diffusion::odf::{OdField, OdfConfig, estimate_odf};
use ritk_diffusion_scheme::{GradientFrame, GradientScheme};
use ritk_spatial::{Point, Vector};
use ritk_tractography::{
    TerminationReason, TrackingDirection, TractographyConfig, euler_tractography,
};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const GRADIENT_COUNT: usize = 48;
const B_VALUE: f64 = 1_500.0;
const BASELINE: f64 = 1_000.0;
const AXIAL_DIFFUSIVITY: f64 = 0.0015;
const RADIAL_DIFFUSIVITY: f64 = 0.0003;
const PANEL_WIDTH: f64 = 320.0;
const FIGURE_HEIGHT: f64 = 470.0;

fn fibonacci_directions(count: usize) -> Vec<[f64; 3]> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..count)
        .map(|index| {
            let y = 1.0 - 2.0 * (index as f64 + 0.5) / count as f64;
            let radius = (1.0 - y * y).sqrt();
            let azimuth = golden_angle * index as f64;
            [radius * azimuth.cos(), y, radius * azimuth.sin()]
        })
        .collect()
}

fn tensor_signal(direction: [f64; 3]) -> f64 {
    let axial_fraction = direction[0] * direction[0];
    let apparent_diffusivity =
        RADIAL_DIFFUSIVITY + (AXIAL_DIFFUSIVITY - RADIAL_DIFFUSIVITY) * axial_fraction;
    BASELINE * (-B_VALUE * apparent_diffusivity).exp()
}

struct TensorOdf {
    directions: Vec<[f64; 3]>,
    signals: Vec<f64>,
    field: OdField,
    peak_error: f64,
}

fn estimate_tensor_odf() -> Result<TensorOdf> {
    let directions = fibonacci_directions(GRADIENT_COUNT);
    let mut pairs = Vec::with_capacity(GRADIENT_COUNT + 2);
    pairs.push((0.0, Vector::new([0.0, 0.0, 0.0])));
    pairs.push((0.0, Vector::new([0.0, 0.0, 0.0])));
    pairs.extend(
        directions
            .iter()
            .copied()
            .map(|direction| (B_VALUE, Vector::new(direction))),
    );
    let scheme =
        GradientScheme::from_seconds_per_square_millimeter(pairs, GradientFrame::ImageAxis)?;
    let mut signals = vec![BASELINE, BASELINE];
    signals.extend(directions.iter().copied().map(tensor_signal));
    let odf = estimate_odf(&scheme, &signals, OdfConfig::default())?;

    const POLAR_INTERVALS: usize = 180;
    const AZIMUTH_SAMPLES: usize = 360;
    let mut peak = None;
    for polar_index in 0..=POLAR_INTERVALS {
        let theta = std::f64::consts::PI * polar_index as f64 / POLAR_INTERVALS as f64;
        for azimuth_index in 0..AZIMUTH_SAMPLES {
            let phi = std::f64::consts::TAU * azimuth_index as f64 / AZIMUTH_SAMPLES as f64;
            let direction = [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ];
            let value = odf.evaluate_at_direction(direction)?;
            if peak.is_none_or(|(_, best)| value > best) {
                peak = Some((direction, value));
            }
        }
    }
    let (peak_direction, _) = peak.context("ODF full-sphere scan has no samples")?;
    let antipodal_error = peak_direction[0].abs().clamp(-1.0, 1.0).acos();
    if antipodal_error.to_degrees() > 2.0 {
        bail!(
            "Q-ball peak misses the analytical x axis by {:.3} degrees",
            antipodal_error.to_degrees()
        );
    }
    Ok(TensorOdf {
        directions,
        signals: signals[2..].to_vec(),
        field: odf,
        peak_error: antipodal_error,
    })
}

fn bundle_center(x: f64) -> f64 {
    20.0 + 5.0 * (x / 8.0).sin()
}

fn direction_field(point: &Point<3>) -> Option<Vector<3>> {
    let [x, y, z] = point.to_array();
    if !(2.0..=38.0).contains(&x) || (y - bundle_center(x)).abs() > 4.0 || z.abs() > 0.5 {
        return None;
    }
    let slope = 0.625 * (x / 8.0).cos();
    let norm = (1.0 + slope * slope).sqrt();
    Some(Vector::new([1.0 / norm, slope / norm, 0.0]))
}

fn tracking_paths() -> Result<Vec<Vec<[f64; 2]>>> {
    let seed_x = 20.0;
    let seeds = [-2.5, -1.25, 0.0, 1.25, 2.5]
        .map(|offset| Point::new([seed_x, bundle_center(seed_x) + offset, 0.0]));
    let config = TractographyConfig::new(0.35, 160, 20.0, TrackingDirection::Bidirectional)?;
    let result = euler_tractography(&seeds, config, direction_field)?;
    if result.streamlines_generated() != seeds.len() {
        bail!(
            "expected {} analytical streamlines, generated {}",
            seeds.len(),
            result.streamlines_generated()
        );
    }
    if result.streamlines().iter().any(|streamline| {
        streamline.forward_termination() != TerminationReason::FieldBoundary
            || streamline.backward_termination() != Some(TerminationReason::FieldBoundary)
    }) {
        bail!("expected every analytical streamline half to terminate at the field boundary");
    }
    let paths = result
        .streamlines()
        .iter()
        .map(|streamline| {
            streamline
                .geometry()
                .points()
                .iter()
                .map(|point| [point.x, point.y])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if paths
        .iter()
        .flat_map(|path| path.iter())
        .any(|[x, y]| direction_field(&Point::new([*x, *y, 0.0])).is_none())
    {
        bail!("tractography emitted a point outside the analytical bundle");
    }
    Ok(paths)
}

fn panel_heading(svg: &mut String, panel: usize, title: &str, subtitle: &str) -> Result<()> {
    let x = panel as f64 * PANEL_WIDTH;
    writeln!(svg, "<g transform=\"translate({x},0)\">")?;
    writeln!(
        svg,
        "<rect width=\"{PANEL_WIDTH}\" height=\"{FIGURE_HEIGHT}\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"160\" y=\"28\" class=\"title\" text-anchor=\"middle\">{title}</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"160\" y=\"48\" class=\"subtitle\" text-anchor=\"middle\">{subtitle}</text>"
    )?;
    Ok(())
}

fn draw_acquisition(svg: &mut String, directions: &[[f64; 3]]) -> Result<()> {
    panel_heading(
        svg,
        0,
        "1. Acquire directional signal",
        "48 unit gradients at b = 1500 s/mm²",
    )?;
    writeln!(
        svg,
        "<circle cx=\"160\" cy=\"205\" r=\"112\" class=\"sphere\"/>"
    )?;
    writeln!(
        svg,
        "<ellipse cx=\"160\" cy=\"205\" rx=\"112\" ry=\"34\" class=\"latitude\"/>"
    )?;
    writeln!(
        svg,
        "<line x1=\"48\" y1=\"205\" x2=\"272\" y2=\"205\" class=\"axis\"/>"
    )?;
    writeln!(
        svg,
        "<line x1=\"160\" y1=\"93\" x2=\"160\" y2=\"317\" class=\"axis\"/>"
    )?;
    for [x, y, z] in directions {
        let projected_x = 160.0 + 106.0 * x;
        let projected_y = 205.0 - 106.0 * y;
        let radius = 2.5 + 1.5 * (z + 1.0) / 2.0;
        writeln!(
            svg,
            "<circle cx=\"{projected_x:.2}\" cy=\"{projected_y:.2}\" r=\"{radius:.2}\" class=\"gradient\"/>"
        )?;
    }
    writeln!(
        svg,
        "<text x=\"160\" y=\"350\" class=\"callout\" text-anchor=\"middle\">Each point is one measured direction</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"160\" y=\"373\" class=\"body\" text-anchor=\"middle\">b0 images establish S₀; weighted images</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"160\" y=\"391\" class=\"body\" text-anchor=\"middle\">attenuate according to tissue orientation.</text></g>"
    )?;
    Ok(())
}

fn draw_signal_and_odf(
    svg: &mut String,
    directions: &[[f64; 3]],
    signals: &[f64],
    odf: &OdField,
    error: f64,
) -> Result<()> {
    panel_heading(
        svg,
        1,
        "2. Estimate orientation",
        "signal attenuation → analytical Q-ball ODF",
    )?;
    writeln!(
        svg,
        "<line x1=\"50\" y1=\"182\" x2=\"275\" y2=\"182\" class=\"axis\"/>"
    )?;
    writeln!(
        svg,
        "<line x1=\"50\" y1=\"76\" x2=\"50\" y2=\"182\" class=\"axis\"/>"
    )?;
    for (direction, signal) in directions.iter().zip(signals) {
        let x = 50.0 + 225.0 * direction[0].abs();
        let y = 182.0 - 106.0 * signal / BASELINE;
        writeln!(
            svg,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"2.5\" class=\"signal\"/>"
        )?;
    }
    writeln!(
        svg,
        "<text x=\"162\" y=\"201\" class=\"label\" text-anchor=\"middle\">|gradient · fiber axis|</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"20\" y=\"130\" class=\"label\" transform=\"rotate(-90 20 130)\">S / S₀</text>"
    )?;

    let samples = (0..181)
        .map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / 180.0;
            odf.evaluate_at_direction([angle.cos(), angle.sin(), 0.0])
                .map(|value| (angle, value))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let minimum = samples
        .iter()
        .map(|sample| sample.1)
        .fold(f64::INFINITY, f64::min);
    let maximum = samples
        .iter()
        .map(|sample| sample.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = maximum - minimum;
    if !range.is_finite() || range <= 0.0 {
        bail!("ODF polar plot has no finite dynamic range");
    }
    let mut path = String::new();
    for (index, (angle, value)) in samples.iter().enumerate() {
        let radius = 30.0 + 64.0 * (value - minimum) / range;
        let x = 160.0 + radius * angle.cos();
        let y = 315.0 - radius * angle.sin();
        write!(path, "{} {x:.2},{y:.2}", if index == 0 { "M" } else { "L" })?;
    }
    path.push_str(" Z");
    writeln!(
        svg,
        "<circle cx=\"160\" cy=\"315\" r=\"94\" class=\"odf-guide\"/>"
    )?;
    writeln!(svg, "<path d=\"{path}\" class=\"odf\"/>")?;
    writeln!(
        svg,
        "<line x1=\"66\" y1=\"315\" x2=\"254\" y2=\"315\" class=\"fiber-axis\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"160\" y=\"430\" class=\"callout\" text-anchor=\"middle\">ODF peak error = {:.2}°</text></g>",
        error.to_degrees()
    )?;
    Ok(())
}

fn draw_tractography(svg: &mut String, paths: &[Vec<[f64; 2]>]) -> Result<()> {
    panel_heading(
        svg,
        2,
        "3. Integrate streamlines",
        "local orientations form bounded trajectories",
    )?;
    let map = |x: f64, y: f64| [24.0 + 6.8 * x, 410.0 - 8.8 * y];
    let mut upper = String::new();
    let mut lower = String::new();
    for index in 0..=100 {
        let x = 2.0 + 36.0 * index as f64 / 100.0;
        let [ux, uy] = map(x, bundle_center(x) + 4.0);
        let [lx, ly] = map(x, bundle_center(x) - 4.0);
        write!(
            upper,
            "{} {ux:.2},{uy:.2}",
            if index == 0 { "M" } else { "L" }
        )?;
        write!(
            lower,
            "{} {lx:.2},{ly:.2}",
            if index == 0 { "M" } else { "L" }
        )?;
    }
    writeln!(
        svg,
        "<path d=\"{upper}\" class=\"bundle-edge\"/><path d=\"{lower}\" class=\"bundle-edge\"/>"
    )?;
    for x in (4..=36).step_by(4) {
        let x = f64::from(x);
        let y = bundle_center(x);
        let direction =
            direction_field(&Point::new([x, y, 0.0])).context("glyph lies outside bundle")?;
        let [cx, cy] = map(x, y);
        let [dx, dy, _] = direction.to_array();
        writeln!(
            svg,
            "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" class=\"glyph\"/>",
            cx - 8.0 * dx,
            cy + 8.0 * dy,
            cx + 8.0 * dx,
            cy - 8.0 * dy
        )?;
    }
    for path in paths {
        let mut data = String::new();
        for (index, [x, y]) in path.iter().enumerate() {
            let [px, py] = map(*x, *y);
            write!(
                data,
                "{} {px:.2},{py:.2}",
                if index == 0 { "M" } else { "L" }
            )?;
        }
        writeln!(svg, "<path d=\"{data}\" class=\"streamline\"/>")?;
    }
    for offset in [-2.5, -1.25, 0.0, 1.25, 2.5] {
        let [x, y] = map(20.0, bundle_center(20.0) + offset);
        writeln!(
            svg,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"3.5\" class=\"seed\"/>"
        )?;
    }
    writeln!(
        svg,
        "<text x=\"160\" y=\"430\" class=\"callout\" text-anchor=\"middle\">5 seeds → 5 boundary-terminated streamlines</text></g>"
    )?;
    Ok(())
}

fn write_figure(path: &Path) -> Result<()> {
    let tensor = estimate_tensor_odf()?;
    let paths = tracking_paths()?;
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {FIGURE_HEIGHT}\"><style>.panel{{fill:#fff;stroke:#cbd5e1}}.title{{font:600 16px sans-serif;fill:#172033}}.subtitle,.label{{font:11px sans-serif;fill:#64748b}}.body{{font:12px sans-serif;fill:#475569}}.callout{{font:600 12px sans-serif;fill:#172033}}.sphere{{fill:#eff6ff;stroke:#60a5fa;stroke-width:2}}.latitude,.axis,.odf-guide{{fill:none;stroke:#94a3b8;stroke-width:1}}.gradient{{fill:#2563eb;fill-opacity:.75}}.signal{{fill:#f97316;fill-opacity:.8}}.odf{{fill:#bfdbfe;fill-opacity:.75;stroke:#2563eb;stroke-width:2.5}}.fiber-axis{{stroke:#dc2626;stroke-width:2;stroke-dasharray:5 4}}.bundle-edge{{fill:none;stroke:#cbd5e1;stroke-width:2;stroke-dasharray:5 4}}.glyph{{stroke:#94a3b8;stroke-width:1.5}}.streamline{{fill:none;stroke:#2563eb;stroke-width:2.3}}.seed{{fill:#f97316;stroke:#fff;stroke-width:1.5}}</style>",
        PANEL_WIDTH * 3.0
    );
    draw_acquisition(&mut svg, &tensor.directions)?;
    draw_signal_and_odf(
        &mut svg,
        &tensor.directions,
        &tensor.signals,
        &tensor.field,
        tensor.peak_error,
    )?;
    draw_tractography(&mut svg, &paths)?;
    svg.push_str("</svg>\n");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create figure directory {}", parent.display()))?;
    }
    std::fs::write(path, svg).with_context(|| format!("write figure {}", path.display()))?;
    println!(
        "wrote {}: ODF peak error {:.2} degrees, {} streamlines",
        path.display(),
        tensor.peak_error.to_degrees(),
        paths.len()
    );
    Ok(())
}

fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/diffusion_tractography.svg"));
    write_figure(&output)
}
