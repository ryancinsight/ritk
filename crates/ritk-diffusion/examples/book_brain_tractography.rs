//! Build a human whole-brain tractogram and regional connectome.
//!
//! Stanford HARDI supplies 150 diffusion directions, 10 reference volumes,
//! and an aligned reduced FreeSurfer parcellation. The example fits one tensor
//! per voxel, tracks from cerebral white matter with sign-invariant trilinear
//! orientation interpolation, assigns endpoints to image-present grey-matter
//! regions, and writes the complete connectivity matrix beside the book figure.
//!
//! Human diffusion MRI has no voxelwise fibre ground truth. The output therefore
//! demonstrates a reproducible data path and internally checkable accounting;
//! it is not evidence that streamline counts equal axon counts or that any
//! individual connection is anatomically true.

#[path = "book_brain_tractography/data.rs"]
mod data;
#[path = "book_brain_tractography/render.rs"]
mod render;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use data::{HumanAtlas, HumanDataset};
use render::{HumanMetrics, SlicePanel};
use ritk_diffusion::maps::{
    DiffusionMapsConfig, DirectionInterpolation, DtiVolume, fit_diffusion_maps,
};
use ritk_diffusion_scheme::{GradientScheme, read_fsl_scheme};
use ritk_spatial::{Point, Vector};
use ritk_tractography::{TrackingDirection, TractographyConfig, euler_tractography};

/// Conventional high-confidence white-matter seeding floor.
const SEED_FA_FLOOR: f64 = 0.25;
/// Lower continuation floor lets a track traverse uncertain bundle margins.
const TRACK_FA_FLOOR: f64 = 0.15;
/// Whole-volume cap that keeps the example bounded while sampling both hemispheres.
const MAX_SEEDS: usize = 12_000;
/// Mid-axial plane used for the tractography projection.
const SLICE_FRACTION: f64 = 0.5;

fn main() -> Result<()> {
    let Some(dataset) = HumanDataset::locate() else {
        eprintln!("skipping: Stanford HARDI is absent. Run test_data/diffusion/download.sh");
        return Ok(());
    };

    let scheme = read_scheme(&dataset.bvals, &dataset.bvecs)?;
    let series = ritk_io::read_image_series_native(&dataset.dwi)
        .map_err(|error| anyhow::anyhow!("reading {}: {error:#}", dataset.dwi.display()))?;
    let reference = series
        .first()
        .context("the DWI series contains no volumes")?;
    anyhow::ensure!(
        series.len() == scheme.len(),
        "series has {} volumes but the scheme declares {}",
        series.len(),
        scheme.len()
    );

    let atlas = HumanAtlas::read(&dataset)?;
    let label_image = ritk_io::read_image_native(&dataset.labels)
        .map_err(|error| anyhow::anyhow!("reading {}: {error:#}", dataset.labels.display()))?;
    anyhow::ensure!(
        label_image.shape() == reference.shape() && atlas.shape_zyx == reference.shape(),
        "DWI shape {:?}, label image shape {:?}, and decoded label shape {:?} must agree",
        reference.shape(),
        label_image.shape(),
        atlas.shape_zyx
    );
    anyhow::ensure!(
        label_image.origin() == reference.origin()
            && label_image.spacing() == reference.spacing()
            && label_image.direction() == reference.direction(),
        "the parcellation and DWI NIfTI spatial transforms differ"
    );
    let channels = render::resolve_colour_channels(reference.direction())?;

    let voxels: Vec<&[f32]> = series
        .iter()
        .map(|volume| {
            volume
                .data_slice()
                .context("DWI volume is not contiguous in host memory")
        })
        .collect::<Result<_>>()?;
    let maps = fit_diffusion_maps(&scheme, &voxels, &DiffusionMapsConfig::default())
        .context("fitting the whole-brain tensor field")?;
    let fitted_voxels = maps.fitted_count();
    let anisotropy = maps.fractional_anisotropy();
    let peak_anisotropy = anisotropy.iter().copied().fold(0.0_f64, f64::max);
    anyhow::ensure!(
        peak_anisotropy >= SEED_FA_FLOOR,
        "no voxel reached the seeding floor {SEED_FA_FLOOR}; peak FA was {peak_anisotropy:.3}"
    );

    let seeds = choose_seeds(&anisotropy, &atlas.labels, reference.shape(), MAX_SEEDS);
    anyhow::ensure!(
        !seeds.is_empty(),
        "the fitted field has no cerebral-white-matter voxel at FA {SEED_FA_FLOOR}"
    );

    let volume = DtiVolume::new(maps, reference.shape(), TRACK_FA_FLOOR)
        .context("placing the tensor field on the DWI grid")?
        .with_interpolation(DirectionInterpolation::Trilinear);
    let tracking = TractographyConfig::new(0.5, 1_000, 60.0, TrackingDirection::Bidirectional)
        .context("validating the tractography configuration")?;
    let tracks = euler_tractography(&seeds, tracking, |point| {
        volume.direction_at(point).map(fsl_direction_to_image_index)
    })
    .context("tracking the human tensor field")?;

    // DtiVolume follows Image order [depth, row, column]. Parcellation follows
    // physical axis order [x, y, z], while both share the same voxel grid.
    let label_tracks = tracks
        .map_points(index_to_label_point)
        .context("mapping tracks into the aligned parcellation grid")?;
    let geometries = label_tracks
        .streamlines()
        .iter()
        .map(|streamline| streamline.geometry().clone())
        .collect::<Vec<_>>();
    let connectome = ritk_connectome::build_connectivity_matrix(
        &atlas.parcellation,
        &geometries,
        &ritk_connectome::ConnectomeConfig::new(),
    )
    .context("building the endpoint connectivity matrix")?;
    validate_connectome_accounting(&connectome)?;

    let physical_tracks = tracks
        .map_points(|index| reference.continuous_index_to_physical_point(index))
        .context("mapping tracks to physical millimetres")?;
    let mut lengths = physical_tracks
        .streamlines()
        .iter()
        .map(|streamline| streamline.geometry().arc_length())
        .collect::<Vec<_>>();
    lengths.sort_by(f64::total_cmp);
    let median_length_mm = lengths[lengths.len() / 2];

    let top_edge = connectome
        .edges()
        .filter(|edge| edge.source != edge.target)
        .max_by(|left, right| left.weight.total_cmp(&right.weight))
        .context("the human connectome contains no inter-region edge")?;
    let metrics = HumanMetrics {
        fitted_voxels,
        seeds: seeds.len(),
        streamlines: tracks.streamlines_generated(),
        assigned_streamlines: connectome.accounting().assigned,
        median_length_mm,
        region_count: connectome.region_count(),
        edge_count: connectome.edge_count(),
        density: connectome.density(),
        top_source: atlas.name(top_edge.source),
        top_target: atlas.name(top_edge.target),
        top_weight: top_edge.weight,
    };

    let [depth, rows, columns] = reference.shape();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "slice index is a bounded fraction of a small image depth"
    )]
    let slice = (depth as f64 * SLICE_FRACTION) as usize;
    let plane = rows * columns;
    let figure = render::render(
        &SlicePanel {
            fa: &anisotropy[slice * plane..(slice + 1) * plane],
            pev: &volume.maps().principal_eigenvector()[slice * plane..(slice + 1) * plane],
            channels,
            rows,
            columns,
            slice,
            depth,
            peak: peak_anisotropy,
        },
        &tracks,
        &connectome,
        &metrics,
    )?;
    let figure_path = figure_path();
    std::fs::write(&figure_path, figure)
        .with_context(|| format!("writing {}", figure_path.display()))?;

    let matrix_path = matrix_path();
    std::fs::write(&matrix_path, connectome.to_json()?)
        .with_context(|| format!("writing {}", matrix_path.display()))?;

    println!(
        "wrote {} and {}: {} streamlines, {} assigned to regions, {} regions, {} edges, median length {:.1} mm",
        figure_path.display(),
        matrix_path.display(),
        metrics.streamlines,
        metrics.assigned_streamlines,
        metrics.region_count,
        metrics.edge_count,
        metrics.median_length_mm
    );
    Ok(())
}

fn read_scheme(bvals: &Path, bvecs: &Path) -> Result<GradientScheme> {
    read_fsl_scheme(
        &std::fs::read_to_string(bvals)?,
        &std::fs::read_to_string(bvecs)?,
    )
    .context("building the gradient scheme from FSL sidecars")
}

fn choose_seeds(
    anisotropy: &[f64],
    labels: &[u32],
    shape: [usize; 3],
    limit: usize,
) -> Vec<Point<3>> {
    let qualifying = anisotropy
        .iter()
        .zip(labels)
        .enumerate()
        .filter(|(_, (fa, label))| **fa >= SEED_FA_FLOOR && matches!(label, 1 | 2))
        .map(|(voxel, _)| voxel)
        .collect::<Vec<_>>();
    let stride = if limit == 0 || qualifying.len() <= limit {
        1
    } else {
        qualifying.len().div_ceil(limit)
    };
    let [_, rows, columns] = shape;
    let plane = rows * columns;
    qualifying
        .into_iter()
        .step_by(stride)
        .map(|voxel| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "voxel indices are far below f64 exact-integer range"
            )]
            Point::new([
                (voxel / plane) as f64,
                ((voxel % plane) / columns) as f64,
                (voxel % columns) as f64,
            ])
        })
        .collect()
}

fn index_to_label_point(index: &Point<3>) -> Point<3> {
    let [z, y, x] = index.to_array();
    Point::new([x, y, z])
}

fn fsl_direction_to_image_index(direction: Vector<3>) -> Vector<3> {
    let [column, row, depth] = direction.to_array();
    Vector::new([depth, row, column])
}

fn validate_connectome_accounting(matrix: &ritk_connectome::ConnectivityMatrix) -> Result<()> {
    // Every streamline that reached a region contributes exactly one unit of
    // weight, to an inter-region edge or to a region's own diagonal, so the
    // weights must sum to the two buckets that recorded such a streamline. The
    // check ties the matrix back to the accounting rather than trusting either
    // alone.
    let recorded_weight = matrix.edges().map(|edge| edge.weight).sum::<f64>();
    let accounting = matrix.accounting();
    #[expect(
        clippy::cast_precision_loss,
        reason = "the bounded example has far fewer than 2^53 streamlines"
    )]
    let placed = (accounting.assigned + accounting.intra_region) as f64;
    anyhow::ensure!(
        recorded_weight == placed,
        "connectome accounting differs: edge weights sum to {recorded_weight},          streamlines placed in a region are {placed}"
    );
    anyhow::ensure!(
        accounting.assigned + accounting.intra_region + accounting.unassigned == accounting.total,
        "connectome accounting does not partition the tractogram: {accounting:?}"
    );
    for source in matrix.region_labels() {
        for target in matrix.region_labels() {
            anyhow::ensure!(
                matrix.weight(*source, *target) == matrix.weight(*target, *source),
                "connectome is not symmetric at labels {source} and {target}"
            );
        }
    }
    Ok(())
}

fn figure_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/book/figures/brain_tractography.svg")
}

fn matrix_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/book/figures/brain_connectome.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_selection_uses_white_matter_and_image_order() {
        let anisotropy = [0.30, 0.40, 0.50, 0.10, 0.60, 0.70, 0.80, 0.90];
        let labels = [1, 3, 2, 1, 0, 2, 4, 1];
        let seeds = choose_seeds(&anisotropy, &labels, [2, 2, 2], 0);
        assert_eq!(seeds.len(), 4);
        assert_eq!(seeds[0].to_array(), [0.0, 0.0, 0.0]);
        assert_eq!(seeds[1].to_array(), [0.0, 1.0, 0.0]);
        assert_eq!(seeds[2].to_array(), [1.0, 0.0, 1.0]);
        assert_eq!(seeds[3].to_array(), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn image_indices_map_to_label_axes() {
        assert_eq!(
            index_to_label_point(&Point::new([7.0, 11.0, 13.0])).to_array(),
            [13.0, 11.0, 7.0]
        );
    }

    #[test]
    fn fsl_directions_map_to_internal_image_axes() {
        assert_eq!(
            fsl_direction_to_image_index(Vector::new([2.0, 3.0, 5.0])).to_array(),
            [5.0, 3.0, 2.0]
        );
    }
}
