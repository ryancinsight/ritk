//! Segmentation algorithms.
//!
//! Provides intensity thresholding, morphological operations, connected-component
//! labeling, region-growing segmentation, clustering-based segmentation,
//! watershed segmentation, level set segmentation, distance transforms,
//! and morphological skeletonization for medical images.
//!
//! # Module Structure
//! - [`threshold`]: Otsu, Multi-Otsu, Li, Yen, Kapur, and Triangle intensity thresholding.
//! - [`morphology`]: Binary morphological operations (erosion, dilation, opening, closing).
//! - [`labeling`]: Connected-component labeling with statistics.
//! - [`region_growing`]: Connected-threshold, confidence-connected, and neighborhood-connected region growing.
//! - [`clustering`]: Clustering-based segmentation (K-Means).
//! - [`watershed`]: Watershed segmentation (Meyer flooding algorithm).
//! - [`level_set`]: Level set methods (Chan-Vese, Geodesic Active Contour).
//! - [`distance_transform`]: Euclidean distance transform (Meijster et al. 2000).

pub mod clustering;
pub mod distance_transform;
pub mod labeling;
pub mod level_set;
pub mod morphology;
pub mod region_growing;
pub mod threshold;
pub mod watershed;

pub use clustering::{kmeans_segment, KMeansSegmentation};
pub use distance_transform::{distance_transform, distance_transform_squared, DistanceTransform};
pub use labeling::{connected_components, ConnectedComponentsFilter, LabelStatistics};
pub use level_set::{
    ChanVeseSegmentation, GeodesicActiveContourSegmentation, ShapeDetectionSegmentation,
    ThresholdLevelSet,
};
pub use morphology::{
    BinaryClosing, BinaryDilation, BinaryErosion, BinaryFillHoles, BinaryOpening,
    MorphologicalGradient, MorphologicalOperation, Skeletonization,
};
pub use region_growing::{
    confidence_connected, connected_threshold, neighborhood_connected, ConfidenceConnectedFilter,
    ConnectedThresholdFilter, NeighborhoodConnectedFilter,
};
pub use threshold::{
    kapur_threshold, li_threshold, multi_otsu_threshold, otsu_threshold, triangle_threshold,
    yen_threshold, KapurThreshold, LiThreshold, MultiOtsuThreshold, OtsuThreshold,
    TriangleThreshold, YenThreshold,
};
pub use watershed::WatershedSegmentation;
