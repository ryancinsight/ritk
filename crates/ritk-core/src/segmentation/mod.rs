//! Segmentation algorithms.
//!
//! Provides intensity thresholding, morphological operations, connected-component
//! labeling, and region-growing segmentation for medical images.
//!
//! # Module Structure
//! - [`threshold`]: Otsu and Multi-Otsu intensity thresholding.
//! - [`morphology`]: Binary morphological operations (erosion, dilation, opening, closing).
//! - [`labeling`]: Connected-component labeling with statistics.
//! - [`region_growing`]: Connected-threshold flood-fill region growing.

pub mod labeling;
pub mod morphology;
pub mod region_growing;
pub mod threshold;

pub use labeling::{connected_components, ConnectedComponentsFilter, LabelStatistics};
pub use morphology::{
    BinaryClosing, BinaryDilation, BinaryErosion, BinaryOpening, MorphologicalOperation,
};
pub use region_growing::{connected_threshold, ConnectedThresholdFilter};
pub use threshold::{multi_otsu_threshold, otsu_threshold, MultiOtsuThreshold, OtsuThreshold};
