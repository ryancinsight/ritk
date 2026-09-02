//! Generate the reusable DTI-volume tractography figure used by the RITK book.
//!
//! The example fits two known tensor regimes on one synthetic image-axis volume,
//! then drives both FA-threshold seeding and DTI-volume tracking through the
//! public `ritk-tractography` API. The SVG is written only after assertions bind
//! its labels and primitive counts to the computed result.
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]

use std::error::Error;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use ritk_diffusion::maps::{DiffusionMapsConfig, DtiVolume, fit_diffusion_maps};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::Vector;
use ritk_tractography::{
    DtiTractographyConfig, TerminationReason, TrackingDirection, TractographyConfig,
    dti_volume_seed_points, dti_volume_tractography,
};

const VOXEL_COUNT: usize = 12;
const TRACKABLE_VOXELS: usize = 8;
const SEED_THRESHOLD: f64 = 0.25;
const TRACK_THRESHOLD: f64 = 0.15;
const MAX_SEEDS: usize = 4;
const PANEL_WIDTH: f64 = 350.0;
const FIGURE_HEIGHT: f64 = 330.0;
const WHITE_MATTER: [f64; 6] = [1.7e-3, 3.0e-4, 3.0e-4, 0.0, 0.0, 0.0];
const LOW_ANISOTROPY: [f64; 6] = [8.0e-4, 8.0e-4, 8.0e-4, 0.0, 0.0, 0.0];

fn image_axis_scheme() -> GradientScheme {
    let direction_count = 30_usize;
    let mut entries = Vec::with_capacity(direction_count + 1);
    entries.push(
        GradientDirection::new(
            DiffusionWeighting::from_seconds_per_square_millimeter(0.0)
                .expect("finite b0 weighting"),
            Vector::new([0.0, 0.0, 0.0]),
        )
        .expect("valid b0 direction"),
    );
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for index in 0..direction_count {
        #[expect(
            clippy::cast_precision_loss,
            reason = "the deterministic example has only thirty directions"
        )]
        let z = 1.0 - 2.0 * (index as f64 + 0.5) / direction_count as f64;
        let radius = (1.0 - z * z).sqrt();
        #[expect(
            clippy::cast_precision_loss,
            reason = "the deterministic example has only thirty directions"
        )]
        let phi = golden_angle * index as f64;
        entries.push(
            GradientDirection::new(
                DiffusionWeighting::from_seconds_per_square_millimeter(1_000.0)
                    .expect("finite diffusion weighting"),
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .expect("unit Fibonacci direction"),
        );
    }
    GradientScheme::new(entries, GradientFrame::ImageAxis).expect("valid image-axis scheme")
}

fn tensor_signal(scheme: &GradientScheme, tensor: [f64; 6]) -> Vec<f64> {
    let [dxx, dyy, dzz, dxy, dxz, dyz] = tensor;
    scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return 1_000.0;
            }
            let [gx, gy, gz] = entry.direction().to_array();
            let apparent = dxx * gx * gx
                + dyy * gy * gy
                + dzz * gz * gz
                + 2.0 * dxy * gx * gy
                + 2.0 * dxz * gx * gz
                + 2.0 * dyz * gy * gz;
            1_000.0 * (-b * apparent).exp()
        })
        .collect()
}

fn fitted_volume() -> Result<DtiVolume, Box<dyn Error>> {
    let scheme = image_axis_scheme();
    let tensors: Vec<[f64; 6]> = (0..VOXEL_COUNT)
        .map(|voxel| {
            if voxel < TRACKABLE_VOXELS {
                WHITE_MATTER
            } else {
                LOW_ANISOTROPY
            }
        })
        .collect();
    let per_voxel: Vec<Vec<f64>> = tensors
        .iter()
        .map(|tensor| tensor_signal(&scheme, *tensor))
        .collect();
    let volumes: Vec<Vec<f64>> = (0..scheme.len())
        .map(|acquisition| per_voxel.iter().map(|voxel| voxel[acquisition]).collect())
        .collect();
    let borrowed: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();
    let maps = fit_diffusion_maps(
        &scheme,
        &borrowed,
        &DiffusionMapsConfig {
            background_fraction: 0.0,
            ..DiffusionMapsConfig::default()
        },
    )?;
    Ok(DtiVolume::new(maps, [1, 1, VOXEL_COUNT], TRACK_THRESHOLD)?)
}

fn render(
    volume: &DtiVolume,
    seeds: &[ritk_spatial::Point<3>],
    tracks: &ritk_tractography::TractographyResult,
) -> Result<String, Box<dyn Error>> {
    let fa = volume.maps().fractional_anisotropy();
    if fa.len() != VOXEL_COUNT {
        return Err("FA map length differs from the plotted volume".into());
    }
    if !fa[..TRACKABLE_VOXELS]
        .iter()
        .all(|value| *value >= SEED_THRESHOLD)
        || !fa[TRACKABLE_VOXELS..]
            .iter()
            .all(|value| *value < TRACK_THRESHOLD)
    {
        return Err("synthetic FA regimes did not survive the fitted-map oracle".into());
    }

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {FIGURE_HEIGHT}\"><style>.panel{{fill:#fff;stroke:#cbd5e1}}.title{{font:600 16px sans-serif;fill:#172033}}.label{{font:11px sans-serif;fill:#475569}}.axis{{stroke:#94a3b8;stroke-width:1}}.fa{{fill:#2563eb;fill-opacity:.75}}.seed{{fill:#f97316;stroke:#fff;stroke-width:1.5}}.streamline{{fill:none;stroke:#2563eb;stroke-width:2.5}}.boundary{{stroke:#dc2626;stroke-width:2;stroke-dasharray:5 4}}",
        PANEL_WIDTH * 2.0
    );
    writeln!(
        svg,
        "<rect width=\"{PANEL_WIDTH}\" height=\"{FIGURE_HEIGHT}\" class=\"panel\"/><rect x=\"{PANEL_WIDTH}\" width=\"{PANEL_WIDTH}\" height=\"{FIGURE_HEIGHT}\" class=\"panel\"/>"
    )?;
    writeln!(
        svg,
        "<text x=\"175\" y=\"28\" class=\"title\" text-anchor=\"middle\">1. FA seeding policy</text>"
    )?;
    writeln!(
        svg,
        "<text x=\"525\" y=\"28\" class=\"title\" text-anchor=\"middle\">2. DTI-volume tracking</text>"
    )?;
    writeln!(
        svg,
        "<line x1=\"40\" y1=\"260\" x2=\"310\" y2=\"260\" class=\"axis\"/><line x1=\"40\" y1=\"65\" x2=\"40\" y2=\"260\" class=\"axis\"/>"
    )?;
    let bar_width = 230.0 / fa.len() as f64;
    for (voxel, value) in fa.iter().copied().enumerate() {
        let height = 170.0 * value;
        let x = 45.0 + voxel as f64 * bar_width;
        let y = 260.0 - height;
        writeln!(
            svg,
            "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{:.2}\" height=\"{height:.2}\" class=\"fa\"/><text x=\"{:.2}\" y=\"278\" class=\"label\" text-anchor=\"middle\">{value:.2}</text>",
            bar_width - 1.0,
            x + (bar_width - 1.0) / 2.0
        )?;
    }
    let threshold_y = 260.0 - 170.0 * SEED_THRESHOLD;
    writeln!(
        svg,
        "<line x1=\"40\" y1=\"{threshold_y:.2}\" x2=\"310\" y2=\"{threshold_y:.2}\" class=\"boundary\"/><text x=\"45\" y=\"{:.2}\" class=\"label\">seed FA ≥ {SEED_THRESHOLD:.2}</text>",
        threshold_y - 6.0
    )?;
    for seed in seeds {
        let z = seed.to_array()[2];
        let x = PANEL_WIDTH + 40.0 + z * 22.0;
        writeln!(
            svg,
            "<circle cx=\"{x:.2}\" cy=\"90\" r=\"4\" class=\"seed\"/>"
        )?;
    }
    for (line_index, streamline) in tracks.streamlines().iter().enumerate() {
        let mut path = String::new();
        for (point_index, point) in streamline.geometry().points().iter().enumerate() {
            let x = PANEL_WIDTH + 40.0 + point.z * 22.0;
            let y = 150.0 + line_index as f64 * 20.0;
            write!(
                path,
                "{} {x:.2},{y:.2}",
                if point_index == 0 { "M" } else { "L" }
            )?;
        }
        writeln!(svg, "<path d=\"{path}\" class=\"streamline\"/>")?;
    }
    writeln!(
        svg,
        "<line x1=\"{:.2}\" y1=\"65\" x2=\"{:.2}\" y2=\"260\" class=\"boundary\"/><text x=\"525\" y=\"285\" class=\"label\" text-anchor=\"middle\">{} seeds → {} streamlines</text>",
        PANEL_WIDTH + 40.0 + TRACKABLE_VOXELS as f64 * 22.0,
        PANEL_WIDTH + 40.0 + TRACKABLE_VOXELS as f64 * 22.0,
        seeds.len(),
        tracks.streamlines_generated()
    )?;
    svg.push_str("</svg>\n");

    assert_eq!(svg.matches("class=\"fa\"").count(), fa.len());
    assert_eq!(svg.matches("class=\"seed\"").count(), seeds.len());
    assert_eq!(
        svg.matches("class=\"streamline\"").count(),
        tracks.streamlines_generated()
    );
    for value in fa {
        assert!(svg.contains(&format!("{value:.2}")));
    }
    Ok(svg)
}

fn write_figure(path: &Path) -> Result<(), Box<dyn Error>> {
    let volume = fitted_volume()?;
    let tracking = TractographyConfig::new(0.5, 32, 60.0, TrackingDirection::Bidirectional)?;
    let policy = DtiTractographyConfig::new(SEED_THRESHOLD, MAX_SEEDS, tracking)?;
    let seeds = dti_volume_seed_points(&volume, SEED_THRESHOLD, MAX_SEEDS)?;
    let tracks = dti_volume_tractography(&volume, policy)?;
    if tracks.seeds_attempted() != seeds.len()
        || tracks.streamlines_generated() != seeds.len()
        || tracks.streamlines().iter().any(|streamline| {
            streamline.forward_termination() != TerminationReason::FieldBoundary
                && streamline.forward_termination() != TerminationReason::StepLimit
        })
    {
        return Err("DTI-volume tracking result violates the deterministic example oracle".into());
    }
    let svg = render(&volume, &seeds, &tracks)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, svg)?;
    println!(
        "wrote {}: {} seeds and {} streamlines",
        path.display(),
        seeds.len(),
        tracks.streamlines_generated()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docs/book/figures/dti_volume_tractography.svg"));
    write_figure(&output)
}
