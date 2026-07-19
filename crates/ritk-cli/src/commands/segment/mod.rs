//! `ritk segment` — image segmentation command.
//!
//! Applies one of the following segmentation algorithms to a 3-D medical image:
//!
//! | Method               | Algorithm                                      |
//! |----------------------|------------------------------------------------|
//! | `otsu`               | Single-threshold Otsu (maximises between-class variance) |
//! | `multi-otsu`         | Multi-class Otsu (K−1 thresholds, K classes)   |
//! | `connected-threshold`| BFS flood-fill region growing from a seed voxel|
//! | `li`                 | Li minimum cross-entropy threshold             |
//! | `yen`                | Yen maximum correlation criterion threshold    |
//! | `kapur`              | Kapur maximum entropy threshold                |
//! | `triangle`           | Triangle (geometric) threshold                 |
//! | `watershed`          | Watershed flooding segmentation                |
//! | `kmeans`             | K-Means intensity clustering                   |
//! | `distance-transform` | Euclidean distance transform of binary mask     |
//! | `fill-holes`          | 6-connected binary hole fill                    |
//! | `morphological-gradient` | Morphological gradient (dilation - erosion)  |
//! | `shape-detection`        | Shape Detection level set (Malladi, Sethian & Vemuri 1995) |
//! | `threshold-level-set`    | Threshold Level Set (Whitaker 1998)            |
//! | `laplacian-level-set`    | Laplacian Level Set                             |

use anyhow::Result;
use tracing::info;

mod args;
mod clustering;
mod helpers;
mod level_set;
mod region_growing;
mod threshold;
mod watershed;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use super::Backend;
pub use args::{SegmentArgs, SegmentMethod};
#[cfg(test)]
pub(crate) use helpers::{count_foreground, parse_seed};

/// Execute the `segment` subcommand.
///
/// Dispatches to the appropriate segmentation algorithm based on `args.method`,
/// writes the output mask / label image, and prints a one-line summary.
///
/// # Errors
/// Returns an error when:
/// - The input image cannot be read.
/// - A required argument for the chosen method is missing or malformed.
/// - The output image cannot be written.
pub fn run(args: SegmentArgs) -> Result<()> {
    info!(
        "segment: starting input={} output={} method={}",
        args.input.display(),
        args.output.display(),
        args.method
    );
    match args.method {
        SegmentMethod::Otsu => threshold::run_otsu(&args),
        SegmentMethod::MultiOtsu => threshold::run_multi_otsu(&args),
        SegmentMethod::ConnectedThreshold => region_growing::run_connected_threshold(&args),
        SegmentMethod::Li => threshold::run_li(&args),
        SegmentMethod::Yen => threshold::run_yen(&args),
        SegmentMethod::Kapur => threshold::run_kapur(&args),
        SegmentMethod::Triangle => threshold::run_triangle(&args),
        SegmentMethod::Watershed => watershed::run_watershed(&args),
        SegmentMethod::Kmeans => clustering::run_kmeans(&args),
        SegmentMethod::DistanceTransform => clustering::run_distance_transform(&args),
        SegmentMethod::FillHoles => clustering::run_fill_holes(&args),
        SegmentMethod::MorphologicalGradient => clustering::run_morphological_gradient(&args),
        SegmentMethod::ConfidenceConnected => region_growing::run_confidence_connected(&args),
        SegmentMethod::NeighborhoodConnected => region_growing::run_neighborhood_connected(&args),
        SegmentMethod::ShapeDetection => level_set::run_shape_detection(&args),
        SegmentMethod::ThresholdLevelSet => level_set::run_threshold_level_set(&args),
        SegmentMethod::LaplacianLevelSet => level_set::run_laplacian_level_set(&args),
        SegmentMethod::Skeletonization => clustering::run_skeletonization(&args),
        SegmentMethod::ConnectedComponents => clustering::run_connected_components(&args),
        SegmentMethod::ChanVese => level_set::run_chan_vese(&args),
        SegmentMethod::GeodesicActiveContour => level_set::run_geodesic_active_contour(&args),
        SegmentMethod::Binary => threshold::run_binary(&args),
        SegmentMethod::MarkerWatershed => watershed::run_marker_watershed(&args),
    }
}
