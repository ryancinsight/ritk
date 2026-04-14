//! Segmentation algorithms.
//!
//! Provides intensity thresholding, morphological operations, connected-component
//! labeling, region-growing segmentation, clustering-based segmentation, and
//! watershed segmentation for medical images.
//!
//! # Module Structure
//! - [`threshold`]: Otsu, Multi-Otsu, Li, Yen, Kapur, and Triangle intensity thresholding.
//! - [`morphology`]: Binary morphological operations (erosion, dilation, opening, closing).
//! - [`labeling`]: Connected-component labeling with statistics.
//! - [`region_growing`]: Connected-threshold flood-fill region growing.
//! - [`clustering`]: Clustering-based segmentation (K-Means).
//! - [`watershed`]: Watershed segmentation (Meyer flooding algorithm).

pub mod clustering;
pub mod labeling;
pub mod morphology;
pub mod region_growing;
pub mod threshold;
pub mod watershed;

pub use clustering::{kmeans_segment, KMeansSegmentation};
pub use labeling::{connected_components, ConnectedComponentsFilter, LabelStatistics};
pub use morphology::{
    BinaryClosing, BinaryDilation, BinaryErosion, BinaryOpening, MorphologicalOperation,
};
pub use region_growing::{connected_threshold, ConnectedThresholdFilter};
pub use threshold::{
    kapur_threshold, li_threshold, multi_otsu_threshold, otsu_threshold, triangle_threshold,
    yen_threshold, KapurThreshold, LiThreshold, MultiOtsuThreshold, OtsuThreshold,
    TriangleThreshold, YenThreshold,
};
pub use watershed::WatershedSegmentation;
