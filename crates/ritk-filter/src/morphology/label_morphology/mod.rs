//! Label morphology operations for 3-D label volumes.
//!
//! Provides label-aware dilation, erosion, opening, closing, and geodesic
//! morphological reconstruction.  All operations work on `f32` label volumes
//! where `0.0` encodes background and positive values encode label IDs.

pub mod label_ops;
pub mod reconstruction;

pub use label_ops::{LabelClosing, LabelDilation, LabelErosion, LabelOpening};
pub use reconstruction::{MorphologicalReconstruction, ReconstructionMode};

#[cfg(test)]
mod tests;
