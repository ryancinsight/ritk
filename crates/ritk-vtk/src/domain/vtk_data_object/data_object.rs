use super::{VtkImageData, VtkPolyData, VtkStructuredGrid, VtkUnstructuredGrid};

/// Top-level VTK data object discriminating between supported dataset types.
#[derive(Debug, Clone)]
pub enum VtkDataObject {
    /// Polygonal mesh (DATASET POLYDATA).
    PolyData(VtkPolyData),
    /// Structured grid dataset (DATASET STRUCTURED_GRID).
    StructuredGrid(VtkStructuredGrid),
    /// Unstructured grid dataset (DATASET UNSTRUCTURED_GRID).
    UnstructuredGrid(VtkUnstructuredGrid),
    /// Regular Cartesian image data (.vti).
    ImageData(VtkImageData) }
