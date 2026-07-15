//! Image segmentation algorithms for medical images.
//!
//! Provides intensity thresholding, morphological operations, connected-component
//! labeling, region-growing segmentation, clustering-based segmentation,
//! watershed segmentation, level set segmentation, and ensemble methods.
//!
//! # Module Structure
//! - [`threshold`]: Otsu, Multi-Otsu, Li, Yen, Kapur, and Triangle intensity thresholding.
//! - [`morphology`]: Binary morphological operations (erosion, dilation, opening, closing).
//! - [`labeling`]: Connected-component labeling with statistics.
//! - [`region_growing`]: Connected-threshold flood-fill region growing.
//! - [`clustering`]: Clustering-based segmentation (K-Means, SLIC superpixels).
//! - [`watershed`]: Watershed segmentation (Meyer flooding algorithm).
//! - [`level_set`]: Level set methods (Chan-Vese, Geodesic Active Contour).
//! - [`ensemble`]: Ensemble methods (STAPLE EM algorithm).

pub mod clustering;
pub mod ensemble;
pub mod labeling;
pub mod level_set;
pub mod morphology;
mod native_output;
mod native_support;
pub mod region_growing;
pub mod threshold;
pub mod watershed;

pub use clustering::{
    kmeans_segment, ConnectivityEnforcement, InitializationPerturbation, ItkSlicConfig,
    ItkSlicFilter, KMeansSegmentation, SlicConfig, SlicSuperpixelFilter,
};
pub use ensemble::{
    multi_label_staple, staple, MultiLabelStapleResult, StapleConvergence, StapleResult,
};
pub use labeling::{
    connected_components, label_set_morph, merge_label_maps, relabel_consecutive,
    scalar_connected_components, vector_connected_components_image, ConnectedComponentsFilter,
    Connectivity, LabelSetMorphOp, LabelStatistics, MergeLabelError, MergeLabelMethod,
    RelabelComponentFilter, RelabelStatistics, ThresholdMaximumConnectedComponentsFilter,
};
pub use level_set::{
    ChanVeseSegmentation, GeodesicActiveContourSegmentation, LaplacianLevelSet,
    ShapeDetectionSegmentation, ThresholdLevelSet,
};
pub use morphology::{
    BinaryClosing, BinaryDilation, BinaryErosion, BinaryFillHoles, BinaryOpening,
    MorphologicalGradient, MorphologicalOperation, Skeletonization,
};
pub use region_growing::{
    connected_threshold, growcut, growcut_slice, ConfidenceConnectedFilter,
    ConnectedThresholdFilter, GrowCutFilter, IsolatedConnectedConfig, IsolatedConnectedFilter,
    IsolatedConnectedOutput, IsolationThreshold, NeighborhoodConnectedFilter,
    VectorConfidenceConnectedConfig, VectorConfidenceConnectedFilter,
};
pub use threshold::{
    binary_threshold, kapur_threshold, li_threshold, multi_otsu_threshold, otsu_threshold,
    triangle_threshold, yen_threshold, AutoThreshold, BinaryThreshold, KapurThreshold, LiThreshold,
    MultiOtsuThreshold, OtsuThreshold, TriangleThreshold, YenThreshold,
};
pub use watershed::{
    FloodConnectivity, IsolatedWatershed, IsolatedWatershedConfig, MarkerControlledWatershed,
    MorphologicalWatershed, TobogganFilter, WatershedLinePolicy, WatershedSegmentation,
};
