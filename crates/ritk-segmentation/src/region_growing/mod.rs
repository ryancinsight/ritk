//! Region growing segmentation algorithms for 3-D images.
//!
//! Provides connected-threshold, confidence-connected, and neighborhood-connected
//! region growing algorithms for medical image segmentation.
//!
//! # Module Structure
//! - [`connected_threshold()`]: Fixed-intensity-bounds flood-fill region growing.
//! - [`confidence_connected`]: Adaptive statistics-based region growing.
//! - [`neighborhood_connected`]: Neighborhood-admissibility-predicate region growing.

#[doc(hidden)]
pub mod confidence_connected;
pub mod connected_threshold;
pub mod growcut;
mod intensity;
pub mod isolated_connected;
#[doc(hidden)]
pub mod neighborhood_connected;
pub mod vector_confidence_connected;

pub use confidence_connected::{confidence_connected, ConfidenceConnectedFilter};
pub use connected_threshold::{connected_threshold, ConnectedThresholdFilter};
pub use growcut::{growcut, growcut_slice, GrowCutFilter};
pub use isolated_connected::{
    IsolatedConnectedConfig, IsolatedConnectedFilter, IsolatedConnectedOutput, IsolationThreshold,
};
pub use neighborhood_connected::{neighborhood_connected, NeighborhoodConnectedFilter};
pub use vector_confidence_connected::{
    VectorConfidenceConnectedConfig, VectorConfidenceConnectedFilter,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "tests_native.rs"]
mod tests_native;
