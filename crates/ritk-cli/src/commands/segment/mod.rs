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

use anyhow::{anyhow, Result};
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

pub use args::SegmentArgs;
#[cfg(test)]
pub(crate) use helpers::{count_foreground, parse_seed};
#[cfg(test)]
pub(crate) use super::Backend;

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
/// - An unknown method name is supplied.
pub fn run(args: SegmentArgs) -> Result<()> {
    info!(
        "segment: starting input={} output={} method={}",
        args.input.display(),
        args.output.display(),
        args.method
    );
    match args.method.as_str() {
        "otsu" => threshold::run_otsu(&args),
        "multi-otsu" => threshold::run_multi_otsu(&args),
        "connected-threshold" => region_growing::run_connected_threshold(&args),
        "li" => threshold::run_li(&args),
        "yen" => threshold::run_yen(&args),
        "kapur" => threshold::run_kapur(&args),
        "triangle" => threshold::run_triangle(&args),
        "watershed" => watershed::run_watershed(&args),
        "kmeans" => clustering::run_kmeans(&args),
        "distance-transform" => clustering::run_distance_transform(&args),
        "fill-holes" => clustering::run_fill_holes(&args),
        "morphological-gradient" => clustering::run_morphological_gradient(&args),
        "confidence-connected" => region_growing::run_confidence_connected(&args),
        "neighborhood-connected" => region_growing::run_neighborhood_connected(&args),
        "shape-detection" => level_set::run_shape_detection(&args),
        "threshold-level-set" => level_set::run_threshold_level_set(&args),
        "laplacian-level-set" => level_set::run_laplacian_level_set(&args),
        "skeletonization" => clustering::run_skeletonization(&args),
        "connected-components" => clustering::run_connected_components(&args),
        "chan-vese" => level_set::run_chan_vese(&args),
        "geodesic-active-contour" => level_set::run_geodesic_active_contour(&args),
        "binary" => threshold::run_binary(&args),
        "marker-watershed" => watershed::run_marker_watershed(&args),
        other => Err(anyhow!(
            "Unknown segmentation method '{}'. \
        Supported methods: otsu, multi-otsu, connected-threshold, \
        li, yen, kapur, triangle, watershed, kmeans, distance-transform, \
        fill-holes, morphological-gradient, confidence-connected, \
        neighborhood-connected, shape-detection, threshold-level-set, \
        laplacian-level-set, skeletonization, connected-components, chan-vese, \
        geodesic-active-contour, binary, marker-watershed.",
            other
        )),
    }
}
