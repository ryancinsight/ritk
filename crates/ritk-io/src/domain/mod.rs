use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

pub mod vtk_data_object;
pub mod vtk_pipeline;

pub use vtk_data_object::{AttributeArray, VtkDataObject, VtkPolyData, VtkStructuredGrid, VtkUnstructuredGrid};
pub use vtk_pipeline::{VtkFilter, VtkPipeline, VtkSink, VtkSource};

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

pub mod vtk_scene;
pub use vtk_scene::{RenderProperties, VtkActor, VtkScene};
