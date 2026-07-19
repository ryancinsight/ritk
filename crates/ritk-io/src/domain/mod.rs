use std::path::Path;

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
///
/// Generic over the image container `I` (the tensor substrate is a variation
/// dimension and never appears in trait or implementor names): implementors
/// read the Burn `ritk_image::Image<f32, B, D>` today and the Atlas
/// `ritk_image::Image<T, B, D>` on the migration path, monomorphized
/// per container â€” one contract, zero-cost, no parallel branded trait family.
pub trait ImageReader<I> {
    /// Read an image natively from a path returning bounded topological structures.
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<I>;
}

/// High-level trait for abstracting image writing.
///
/// Generic over the image container `I` â€” see [`ImageReader`].
pub trait ImageWriter<I> {
    /// Write a constrained topology image onto disk avoiding approximations.
    fn write<P: AsRef<Path>>(&self, path: P, image: &I) -> std::io::Result<()>;
}

/// Map a format crate's `anyhow` error onto the contract's `std::io::Error`
/// (one mapping shared by every implementor).
pub(crate) fn to_io_err(e: anyhow::Error) -> std::io::Error {
    std::io::Error::other(e.to_string())
}
