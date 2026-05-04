//! `ritk-vtk` — VTK-native data model and I/O for the RITK toolkit.
//!
//! Provides the authoritative VTK data model (domain types) and all VTK-format
//! I/O free functions.  Designed as a pure VTK-domain library with no
//! dependency on `ritk-io` domain traits (no orphan-rule violations).

pub mod domain;
pub mod io;

pub use domain::{
    AttributeArray, RenderProperties, VtkActor, VtkCellType, VtkDataObject, VtkFilter,
    VtkImageData, VtkPipeline, VtkPolyData, VtkScene, VtkSink, VtkSource, VtkStructuredGrid,
    VtkUnstructuredGrid,
};

pub use io::{
    read_vtk, write_vtk,
    read_vtk_polydata, write_vtk_polydata,
    read_vtp_polydata, write_vtp_polydata,
    read_vti_binary_appended, read_vti_binary_appended_bytes, read_vti_image_data,
    write_vti_binary_appended_bytes, write_vti_binary_appended_to_file, write_vti_image_data,
    write_vti_str,
    read_vtk_structured_grid, write_vtk_structured_grid,
    read_vtk_unstructured_grid, write_vtk_unstructured_grid,
    read_vtu_unstructured_grid, write_vtu_str, write_vtu_unstructured_grid,
    VtkReader, VtkWriter,
};