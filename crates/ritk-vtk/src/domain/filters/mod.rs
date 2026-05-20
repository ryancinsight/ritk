//! Concrete VTK geometry and attribute filters.
//!
//! Each filter implements `VtkFilter` from `crate::domain::vtk_pipeline` and
//! is isolated in its own module to enforce the 500-line structural limit.

pub mod normals;
pub mod smooth;
pub mod threshold;

pub use normals::ComputeNormalsFilter;
pub use smooth::SmoothFilter;
pub use threshold::ThresholdFilter;
