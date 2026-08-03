//! Fit diffusion tensors to a real DWI acquisition and track through them.
//!
//! The companion `book_diffusion_tractography` example explains the method on a
//! synthetic single-bundle phantom, where the answer is known and the ODF peak
//! error can be asserted at zero. This one runs the same estimators over a real
//! subject and renders what they actually produce, which is a different claim:
//! not "the method is correct" but "the method applied to scanner data yields
//! anatomically recognisable structure".
//!
//! # Data
//!
//! OpenNeuro `ds002087` sub-01, CC0. 104x104x72 at 2 mm, 99 volumes at b = 0
//! and b = 700 s/mm^2. Fetch with `test_data/diffusion/download.sh`, then the
//! DWI volume itself from OpenNeuro S3 (the script leaves it as a git-annex
//! pointer). The example exits without writing when the data is absent, so it
//! stays buildable and runnable in CI, where the dataset is not present.
//!
//! # What the figures show
//!
//! Panel 1 is a fractional-anisotropy map of one axial slice. FA is a scalar in
//! `[0, 1]` measuring how directional the local diffusion is, so the bright
//! structure is where water moves preferentially along one axis — white matter.
//! The corpus callosum should read as a bright band across the midline.
//!
//! Panel 2 overlays streamlines seeded in the high-FA voxels of that slice.
//! Their agreement with the bright structure underneath is the check: tracks
//! that wander off the anisotropic tissue would indicate the direction field
//! and the FA map disagree.
//!
//! Both panels are rendered from the fitted values, not drawn. Nothing here is
//! illustrative.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ritk_diffusion::dti::{DtiConfig, estimate_dti};
use ritk_diffusion_scheme::{GradientScheme, read_fsl_scheme};
use ritk_spatial::{Point, Vector};
use ritk_tractography::{TractographyConfig, euler_tractography};

/// Axial slice rendered, as a fraction of the volume's depth.
///
/// Mid-brain, where the corpus callosum and the internal capsule are both in
/// plane — the structures whose presence makes the map checkable by eye.
const SLICE_FRACTION: f64 = 0.5;

/// FA below this is treated as isotropic tissue and not seeded.
///
/// 0.25 is the conventional white-matter floor: grey matter and CSF sit well
/// below it, coherent bundles well above.
const SEED_FA_FLOOR: f64 = 0.25;

/// Cap on seeds, so the figure stays legible and the run stays bounded.
const MAX_SEEDS: usize = 220;

/// Slices either side of the rendered plane that are fitted and tracked.
///
/// Tracking must be three-dimensional: a principal eigenvector with an
/// out-of-plane component leaves a single slice immediately, which truncates
/// every track that is not incidentally axial. A slab gives tracks room to
/// leave the plane while keeping the demonstration bounded. Widening this to
/// the full depth is the scale-up path and changes nothing but runtime.
const SLAB_RADIUS: usize = 14;

/// Upper bound on a physically admissible diffusivity, in mm^2/s.
///
/// Free water at body temperature diffuses at about 3.0e-3 mm^2/s, and no
/// tissue compartment exceeds free water. An eigenvalue above this is a fit
/// artefact, not a measurement.
const FREE_WATER_CEILING: f64 = 3.2e-3;

/// Lower bound on a physically admissible diffusivity, in mm^2/s.
///
/// Radial diffusivity in coherent white matter is around 2-3e-4 mm^2/s; no
/// tissue restricts water two orders below that. A smallest eigenvalue under
/// this floor is a collapsed, rank-one fit rather than an anisotropic voxel —
/// which is the actual source of impossible FA, since such a tensor is
/// positive-definite and passes a sign check while driving FA toward one.
const RESTRICTED_DIFFUSIVITY_FLOOR: f64 = 1.0e-5;

/// Background threshold as a fraction of the b = 0 signal's upper percentile.
///
/// Outside the head the signal is noise, and a tensor fitted to noise produces
/// spurious anisotropy — the bright rim that otherwise dominates the FA range.
/// Referencing a high percentile rather than the maximum keeps one hot voxel
/// from setting the scale.
const BACKGROUND_FRACTION: f64 = 0.12;

const PANEL: f64 = 420.0;

fn main() -> Result<()> {
    let Some((dwi, bval, bvec)) = locate_dataset() else {
        eprintln!(
            "skipping: DWI dataset not present. Run test_data/diffusion/download.sh, then \
             fetch sub-01_run-1_dwi.nii.gz from OpenNeuro S3."
        );
        return Ok(());
    };

    let scheme = read_scheme(&bval, &bvec)?;
    let series = ritk_io::read_image_series_native(&dwi)
        .map_err(|error| anyhow::anyhow!("reading {}: {error:#}", dwi.display()))?;
    anyhow::ensure!(
        series.len() == scheme.len(),
        "series has {} volumes but the scheme declares {}",
        series.len(),
        scheme.len()
    );

    let [depth, rows, columns] = series[0].shape();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "slice index is a bounded fraction of a small depth"
    )]
    let slice = ((depth as f64) * SLICE_FRACTION) as usize;

    let voxels: Vec<&[f32]> = series
        .iter()
        .map(|volume| volume.data_slice().expect("contiguous host voxels"))
        .collect();

    // ── Background mask from the b = 0 volume ─────────────────────────────
    let b0 = scheme
        .directions()
        .iter()
        .position(|entry| entry.weighting().is_unweighted())
        .context("the scheme declares no b = 0 volume to build a mask from")?;
    let mut sorted: Vec<f32> = voxels[b0].to_vec();
    sorted.sort_by(f32::total_cmp);
    let upper = f64::from(sorted[sorted.len() * 98 / 100]);
    let floor = upper * BACKGROUND_FRACTION;

    // ── Fit one tensor per voxel of the slab ──────────────────────────────
    let first = slice.saturating_sub(SLAB_RADIUS);
    let last = (slice + SLAB_RADIUS + 1).min(depth);
    let slab = last - first;
    let plane = rows * columns;

    let mut fa = vec![0.0_f64; slab * plane];
    let mut pev = vec![[0.0_f64; 3]; slab * plane];
    let mut signals = vec![0.0_f64; scheme.len()];

    for local in 0..slab {
        for row in 0..rows {
            for column in 0..columns {
                let offset = (first + local) * plane + row * columns + column;
                if f64::from(voxels[b0][offset]) < floor {
                    continue;
                }
                for (slot, volume) in signals.iter_mut().zip(&voxels) {
                    *slot = f64::from(volume[offset]);
                }
                if let Ok(tensor) = estimate_dti(&scheme, &signals, DtiConfig::default()) {
                    // Reject the fit, not the voxel. Diffusion eigenvalues are
                    // positive and bounded by free water; a tensor violating
                    // either is a failed estimate however bright its voxel was.
                    // Those degenerate fits are what drive FA toward 1 and put
                    // a speckle of impossible anisotropy through the map.
                    let [largest, _, smallest] = *tensor.eigenvalues();
                    if smallest < RESTRICTED_DIFFUSIVITY_FLOOR || largest > FREE_WATER_CEILING {
                        continue;
                    }
                    let index = local * plane + row * columns + column;
                    fa[index] = tensor.fa();
                    pev[index] = tensor.principal_eigenvector();
                }
            }
        }
    }

    let brightest = fa.iter().copied().fold(0.0_f64, f64::max);
    anyhow::ensure!(
        brightest > SEED_FA_FLOOR,
        "no voxel in the slab reached FA {SEED_FA_FLOOR}; peak was {brightest:.3}.          The fit produced no anisotropic tissue, which means the scheme and the          volumes are misaligned rather than that the brain lacks white matter."
    );

    // ── Track through the fitted field, in three dimensions ──────────────
    let seeds = choose_seeds(&fa, slab, rows, columns);
    let sample = |point: &Point<3>| -> Option<Vector<3>> {
        let (z, y, x) = (point[0].round(), point[1].round(), point[2].round());
        if z < 0.0 || y < 0.0 || x < 0.0 {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "negatives are rejected immediately above and bounds below"
        )]
        let (z, y, x) = (z as usize, y as usize, x as usize);
        if z >= slab || y >= rows || x >= columns {
            return None;
        }
        let index = z * plane + y * columns + x;
        // Below the floor the principal direction is noise, so the track stops
        // rather than continuing on an orientation the data does not support.
        (fa[index] >= SEED_FA_FLOOR).then(|| Vector::new(pev[index]))
    };

    let tracks = euler_tractography(&seeds, TractographyConfig::default(), sample)
        .context("tracking through the fitted tensor field")?;

    let centre = slice - first;
    let figure = render(
        &fa[centre * plane..(centre + 1) * plane],
        rows,
        columns,
        &tracks,
        slice,
        depth,
        brightest,
    )?;
    let out = figure_path();
    std::fs::write(&out, figure).with_context(|| format!("writing {}", out.display()))?;

    println!(
        "wrote {}: slice {slice}/{depth}, peak FA {brightest:.3}, {} seeds, {} streamlines",
        out.display(),
        seeds.len(),
        tracks.streamlines().len()
    );
    Ok(())
}

/// Seed the most anisotropic voxels, spread across the slice.
///
/// Taking the strongest FA voxels concentrates seeds in coherent white matter,
/// which is where a deterministic tensor track is meaningful; a uniform grid
/// would spend most seeds in tissue with no direction to follow.
fn choose_seeds(fa: &[f64], slab: usize, rows: usize, columns: usize) -> Vec<Point<3>> {
    let plane = rows * columns;
    let mut candidates: Vec<(usize, f64)> = fa
        .iter()
        .enumerate()
        .filter(|(_, value)| **value >= SEED_FA_FLOOR)
        .map(|(index, value)| (index, *value))
        .collect();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));

    // Stride the ranked list rather than taking the top block, so seeds are not
    // all packed into one bundle.
    let stride = (candidates.len() / MAX_SEEDS).max(1);
    let _ = slab;
    candidates
        .iter()
        .step_by(stride)
        .take(MAX_SEEDS)
        .map(|(index, _)| {
            let z = index / plane;
            let row = (index % plane) / columns;
            let column = index % columns;
            #[expect(
                clippy::cast_precision_loss,
                reason = "voxel indices are far below f64 exact-integer range"
            )]
            Point::new([z as f64, row as f64, column as f64])
        })
        .collect()
}

fn locate_dataset() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/diffusion");
    let candidates = [
        (
            root.join("sub-01_dwi.nii.gz"),
            root.join("sub-01_dwi.bval"),
            root.join("sub-01_dwi.bvec"),
        ),
        (
            root.join("ds002087_repo/sub-01/dwi/sub-01_run-1_dwi.nii.gz"),
            root.join("ds002087_repo/sub-01/dwi/sub-01_run-1_dwi.bval"),
            root.join("ds002087_repo/sub-01/dwi/sub-01_run-1_dwi.bvec"),
        ),
    ];
    candidates.into_iter().find(|(dwi, bval, bvec)| {
        // A git-annex pointer is a few hundred bytes; a real volume is tens of
        // megabytes. Size is what distinguishes them, not existence.
        bval.exists()
            && bvec.exists()
            && std::fs::metadata(dwi).map(|meta| meta.len()).unwrap_or(0) > 1_000_000
    })
}

fn read_scheme(bval: &Path, bvec: &Path) -> Result<GradientScheme> {
    read_fsl_scheme(
        &std::fs::read_to_string(bval)?,
        &std::fs::read_to_string(bvec)?,
    )
    .context("building the gradient scheme from FSL sidecars")
}

fn figure_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/book/figures/brain_tractography.svg")
}

/// Render the FA map and the streamline overlay as two panels.
fn render(
    fa: &[f64],
    rows: usize,
    columns: usize,
    tracks: &ritk_tractography::TractographyResult,
    slice: usize,
    depth: usize,
    peak: f64,
) -> Result<String> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "voxel counts are far below f64 exact-integer range"
    )]
    let (width, height) = (columns as f64, rows as f64);
    let scale = PANEL / width.max(height);

    let mut svg = String::with_capacity(1 << 20);
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">"#,
        PANEL * 2.0 + 60.0,
        PANEL + 76.0
    )?;
    writeln!(
        svg,
        r#"<style>.t{{font:600 15px sans-serif;fill:#172033}}.s{{font:11px sans-serif;fill:#64748b}}.tr{{fill:none;stroke:#f97316;stroke-width:1.1;stroke-opacity:.85}}</style>"#
    )?;
    writeln!(svg, r##"<rect width="100%" height="100%" fill="#fff"/>"##)?;

    for (panel, title) in [
        (0.0_f64, "1. Fractional anisotropy"),
        (PANEL + 40.0, "2. Streamlines over the same slice"),
    ] {
        writeln!(
            svg,
            r#"<text x="{}" y="24" class="t" text-anchor="middle">{title}</text>"#,
            panel + PANEL / 2.0
        )?;
        writeln!(svg, r#"<g transform="translate({panel},40)">"#)?;
        // FA as greyscale: one rect per voxel, intensity straight from the
        // fitted value normalized by the slice peak.
        for row in 0..rows {
            for column in 0..columns {
                let value = fa[row * columns + column] / peak;
                if value <= 0.02 {
                    continue;
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "value is clamped to [0, 1] by construction above"
                )]
                let level = (value.clamp(0.0, 1.0) * 255.0) as u8;
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "voxel indices are small integers"
                )]
                let (x, y) = (column as f64 * scale, row as f64 * scale);
                writeln!(
                    svg,
                    r##"<rect x="{x:.2}" y="{y:.2}" width="{scale:.2}" height="{scale:.2}" fill="#{level:02x}{level:02x}{level:02x}"/>"##
                )?;
            }
        }
        if panel > 0.0 {
            for streamline in tracks.streamlines() {
                let mut path = String::new();
                for (step, point) in streamline.geometry().points().iter().enumerate() {
                    let command = if step == 0 { 'M' } else { 'L' };
                    write!(
                        path,
                        "{command}{:.1} {:.1} ",
                        point.z * scale,
                        point.y * scale
                    )?;
                }
                writeln!(svg, r#"<path class="tr" d="{}"/>"#, path.trim_end())?;
            }
        }
        writeln!(svg, "</g>")?;
    }

    writeln!(
        svg,
        r#"<text x="{}" y="{}" class="s" text-anchor="middle">OpenNeuro ds002087 sub-01 — axial slice {slice} of {depth}, 99 volumes at b = 0 and 700 s/mm², peak FA {peak:.2}</text>"#,
        PANEL + 30.0,
        PANEL + 66.0
    )?;
    writeln!(svg, "</svg>")?;
    Ok(svg)
}
