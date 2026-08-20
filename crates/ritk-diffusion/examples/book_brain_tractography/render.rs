use std::fmt::Write as _;

use anyhow::{Context, Result};
use ritk_connectome::ConnectivityMatrix;

const PANEL: f64 = 340.0;
const GAP: f64 = 30.0;
const TOP: f64 = 42.0;
const MAX_VISIBLE_TRACKS: usize = 900;
const MIN_AXIS_DOMINANCE: f64 = 3.0;

pub struct HumanMetrics {
    pub fitted_voxels: usize,
    pub seeds: usize,
    pub streamlines: usize,
    pub assigned_streamlines: usize,
    pub median_length_mm: f64,
    pub region_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub top_source: String,
    pub top_target: String,
    pub top_weight: f64,
}

pub struct SlicePanel<'a> {
    pub fa: &'a [f64],
    pub pev: &'a [[f64; 3]],
    pub channels: [usize; 3],
    pub rows: usize,
    pub columns: usize,
    pub slice: usize,
    pub depth: usize,
    pub peak: f64,
}

pub fn resolve_colour_channels(direction: &ritk_spatial::Direction<3>) -> Result<[usize; 3]> {
    let mut channels = [0_usize; 3];
    for (component, channel) in channels.iter_mut().enumerate() {
        let column = 2 - component;
        let magnitudes = [
            direction.0[(0, column)].abs(),
            direction.0[(1, column)].abs(),
            direction.0[(2, column)].abs(),
        ];
        let dominant = (0..3)
            .max_by(|left, right| magnitudes[*left].total_cmp(&magnitudes[*right]))
            .unwrap_or(0);
        let runner_up = (0..3)
            .filter(|axis| *axis != dominant)
            .map(|axis| magnitudes[axis])
            .fold(0.0_f64, f64::max);
        anyhow::ensure!(
            magnitudes[dominant] >= runner_up * MIN_AXIS_DOMINANCE,
            "gradient component {component} maps to an ambiguous image axis {column}: {magnitudes:?}"
        );
        *channel = dominant;
    }
    let mut seen = channels;
    seen.sort_unstable();
    anyhow::ensure!(
        seen == [0, 1, 2],
        "colour channels are not a permutation: {channels:?}"
    );
    Ok(channels)
}

pub fn render(
    panel: &SlicePanel<'_>,
    tracks: &ritk_tractography::TractographyResult,
    connectome: &ConnectivityMatrix,
    metrics: &HumanMetrics,
) -> Result<String> {
    let total_width = PANEL * 3.0 + GAP * 2.0;
    let mut svg = String::with_capacity(2 << 20);
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_width} 465">"#
    )?;
    writeln!(
        svg,
        r#"<style>.t{{font:600 15px sans-serif;fill:#172033}}.s{{font:11px sans-serif;fill:#64748b}}.m{{font:600 12px sans-serif;fill:#334155}}</style>"#
    )?;
    writeln!(svg, r##"<rect width="100%" height="100%" fill="#fff"/>"##)?;

    render_anatomy(&mut svg, panel, 0.0, false, tracks)?;
    render_anatomy(&mut svg, panel, PANEL + GAP, true, tracks)?;
    render_connectome(&mut svg, connectome, PANEL * 2.0 + GAP * 2.0)?;

    let assigned_percent = if metrics.streamlines == 0 {
        0.0
    } else {
        #[expect(
            clippy::cast_precision_loss,
            reason = "the bounded example has far fewer than 2^53 streamlines"
        )]
        {
            100.0 * metrics.assigned_streamlines as f64 / metrics.streamlines as f64
        }
    };
    writeln!(
        svg,
        r#"<text x="{}" y="403" class="m" text-anchor="middle">{} fitted voxels · {} seeds · {} streamlines · median {:.1} mm</text>"#,
        total_width / 2.0,
        metrics.fitted_voxels,
        metrics.seeds,
        metrics.streamlines,
        metrics.median_length_mm
    )?;
    writeln!(
        svg,
        r#"<text x="{}" y="421" class="s" text-anchor="middle">{} of {} streamlines assigned ({assigned_percent:.1}%) · {} regions · {} edges · density {:.3}</text>"#,
        total_width / 2.0,
        metrics.assigned_streamlines,
        metrics.streamlines,
        metrics.region_count,
        metrics.edge_count,
        metrics.density
    )?;
    writeln!(
        svg,
        r#"<text x="{}" y="439" class="s" text-anchor="middle">strongest edge: {} ↔ {} ({} streamlines)</text>"#,
        total_width / 2.0,
        metrics.top_source,
        metrics.top_target,
        metrics.top_weight
    )?;
    writeln!(
        svg,
        r#"<text x="{}" y="457" class="s" text-anchor="middle">Stanford HARDI · axial slice {} of {} · 150 directions at b = 2000 s/mm² plus 10 b₀ volumes</text>"#,
        total_width / 2.0,
        panel.slice,
        panel.depth
    )?;
    writeln!(svg, "</svg>")?;
    Ok(svg)
}

fn render_anatomy(
    svg: &mut String,
    panel: &SlicePanel<'_>,
    offset: f64,
    show_tracks: bool,
    tracks: &ritk_tractography::TractographyResult,
) -> Result<()> {
    let title = if show_tracks {
        "2. Whole-brain streamlines"
    } else {
        "1. Directionally encoded FA"
    };
    writeln!(
        svg,
        r#"<text x="{}" y="24" class="t" text-anchor="middle">{title}</text>"#,
        offset + PANEL / 2.0
    )?;
    writeln!(svg, r#"<g transform="translate({offset},{TOP})">"#)?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "image dimensions are far below f64 exact-integer range"
    )]
    let scale = PANEL / (panel.columns.max(panel.rows) as f64);
    for row in 0..panel.rows {
        for column in 0..panel.columns {
            let index = row * panel.columns + column;
            let value = panel.fa[index] / panel.peak;
            if value <= 0.02 {
                continue;
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "image indices are small integers"
            )]
            let (x, y) = (column as f64 * scale, row as f64 * scale);
            let fill = directional_colour(panel.pev[index], value, panel.channels);
            writeln!(
                svg,
                r#"<rect x="{x:.2}" y="{y:.2}" width="{scale:.2}" height="{scale:.2}" fill="{fill}"/>"#
            )?;
        }
    }
    if show_tracks {
        let stride = tracks
            .streamlines_generated()
            .div_ceil(MAX_VISIBLE_TRACKS)
            .max(1);
        for streamline in tracks.streamlines().iter().step_by(stride) {
            let points = streamline.geometry().points();
            let Some((first, last)) = points.first().zip(points.last()) else {
                continue;
            };
            // Geometry is [depth, row, column]; direction colour is [x, y, z].
            let chord = [last.z - first.z, last.y - first.y, last.x - first.x];
            let length = chord.iter().map(|value| value * value).sum::<f64>().sqrt();
            if length <= f64::EPSILON {
                continue;
            }
            let unit = [chord[0] / length, chord[1] / length, chord[2] / length];
            write!(svg, r##"<polyline points=""##)?;
            for point in points {
                write!(svg, "{:.1},{:.1} ", point.z * scale, point.y * scale)?;
            }
            writeln!(
                svg,
                r#"" fill="none" stroke="{}" stroke-width=".8" stroke-opacity=".55"/>"#,
                directional_colour(unit, 1.0, panel.channels)
            )?;
        }
    }
    writeln!(svg, "</g>")?;
    Ok(())
}

fn render_connectome(svg: &mut String, matrix: &ConnectivityMatrix, offset: f64) -> Result<()> {
    writeln!(
        svg,
        r#"<text x="{}" y="24" class="t" text-anchor="middle">3. Endpoint connectome</text>"#,
        offset + PANEL / 2.0
    )?;
    let labels = matrix.region_labels();
    let max_weight = matrix
        .edges()
        .map(|edge| edge.weight)
        .fold(0.0_f64, f64::max);
    anyhow::ensure!(max_weight > 0.0, "connectome has no visible weight");
    #[expect(
        clippy::cast_precision_loss,
        reason = "the atlas region count is far below f64 exact-integer range"
    )]
    let cell = PANEL / labels.len() as f64;
    writeln!(svg, r#"<g transform="translate({offset},{TOP})">"#)?;
    for (row, source) in labels.iter().enumerate() {
        for (column, target) in labels.iter().enumerate() {
            let weight = matrix
                .weight(*source, *target)
                .context("region label disappeared while rendering")?;
            if weight <= 0.0 {
                continue;
            }
            let intensity = weight.ln_1p() / max_weight.ln_1p();
            #[expect(
                clippy::cast_precision_loss,
                reason = "atlas indices are small integers"
            )]
            let (x, y) = (column as f64 * cell, row as f64 * cell);
            writeln!(
                svg,
                r#"<rect x="{x:.2}" y="{y:.2}" width="{cell:.2}" height="{cell:.2}" fill="{}"/>"#,
                heat_colour(intensity)
            )?;
        }
    }
    let hemisphere = labels.partition_point(|label| *label < 46);
    #[expect(
        clippy::cast_precision_loss,
        reason = "the atlas region count is far below f64 exact-integer range"
    )]
    let boundary = hemisphere as f64 * cell;
    writeln!(
        svg,
        r##"<path d="M{boundary:.2} 0V{PANEL}M0 {boundary:.2}H{PANEL}" stroke="#e2e8f0" stroke-width="1"/>"##
    )?;
    writeln!(svg, "</g>")?;
    writeln!(
        svg,
        r#"<text x="{}" y="394" class="s" text-anchor="middle">left hemisphere</text><text x="{}" y="394" class="s" text-anchor="middle">right hemisphere</text>"#,
        offset + boundary / 2.0,
        offset + boundary + (PANEL - boundary) / 2.0
    )?;
    Ok(())
}

fn directional_colour(direction: [f64; 3], weight: f64, channels: [usize; 3]) -> String {
    let level = |component: f64| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped colour intensity scaled by 255 fits u8"
        )]
        {
            (component.abs().clamp(0.0, 1.0) * weight.clamp(0.0, 1.0) * 255.0) as u8
        }
    };
    let mut rgb = [0_u8; 3];
    for (component, channel) in channels.iter().enumerate() {
        rgb[*channel] = level(direction[component]);
    }
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

fn heat_colour(intensity: f64) -> String {
    let t = intensity.clamp(0.0, 1.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped colour components scaled by 255 fit u8"
    )]
    let channel = |start: f64, end: f64| ((start + t * (end - start)) * 255.0) as u8;
    let red = channel(0.07, 0.96);
    let green = channel(0.10, 0.78);
    let blue = channel(0.22, 0.20);
    format!("#{red:02x}{green:02x}{blue:02x}")
}
