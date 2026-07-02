use burn::tensor::backend::Backend;
use ritk_image::Image;
use std::path::Path;

/// Coeus-typed I/O contract (ADR 0002 parallel family).
#[cfg(feature = "coeus")]
pub mod coeus;
#[cfg(feature = "coeus")]
pub use coeus::{CoeusImageReader, CoeusImageWriter};

// VTK domain types are authoritative in ritk-vtk.
// Keep module shims so existing `crate::domain::vtk_data_object::*` paths resolve.
pub mod vtk_data_object;
pub mod vtk_pipeline;
pub mod vtk_scene;
pub use vtk_data_object::{
    AttributeArray, VtkCellType, VtkDataObject, VtkImageData, VtkPolyData, VtkStructuredGrid,
    VtkUnstructuredGrid,
};
pub use vtk_pipeline::{VtkFilter, VtkPipeline, VtkSink, VtkSource};
pub use vtk_scene::{RenderProperties, VtkActor, VtkScene};

/// High-level trait for abstracting image reading.
pub trait ImageReader<B: Backend, const D: usize> {
    /// Read an image natively from a path returning bounded topological structures.
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, D>>;
}

/// High-level trait for abstracting image writing.
pub trait ImageWriter<B: Backend, const D: usize> {
    /// Write a constrained topology image onto disk avoiding approximations.
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, D>) -> std::io::Result<()>;
}
