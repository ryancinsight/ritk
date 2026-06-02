//! `ritk-vtk` — VTK-native data model and I/O for the RITK toolkit.
//!
//! Provides the authoritative VTK data model (domain types) and all VTK-format
//! I/O free functions. Designed as a pure VTK-domain library with no
//! dependency on `ritk-io` domain traits (no orphan-rule violations).

pub mod domain;
pub mod io;

pub use domain::{
    AttributeArray, Block, ColormapPreset, ComputeNormalsFilter, EventHandlers, EventId, LeafIter,
    Modifiable, ModifiedTime, Observable, ObserverCallback, ObserverTag, PolygonMode,
    RenderProperties, ScalarVisibility, SmoothFilter, SurfaceMapper, ThresholdFilter, Visibility,
    VtkActor, VtkCellType, VtkDataObject, VtkFilter, VtkImageData, VtkLookupTable, VtkMapper,
    VtkMultiBlockDataSet, VtkPipeline, VtkPolyData, VtkScene, VtkSink, VtkSource,
    VtkStructuredGrid, VtkUnstructuredGrid,
};

pub use io::{
    read_obj_indexed,
    read_obj_mesh,
    read_ply_indexed,
    read_ply_mesh,
    // Gaia-native indexed-mesh I/O (welded, watertight IndexedMesh<f64>)
    read_stl_indexed,
    read_stl_mesh,
    read_vti_binary_appended,
    read_vti_binary_appended_bytes,
    read_vti_image_data,
    read_vtk,
    read_vtk_polydata,
    read_vtk_structured_grid,
    read_vtk_unstructured_grid,
    read_vtp_polydata,
    read_vtu_unstructured_grid,
    write_gltf,
    write_indexed_glb,
    write_indexed_obj,
    write_indexed_ply,
    write_indexed_stl_ascii,
    write_indexed_stl_binary,
    write_obj_mesh,
    write_ply_ascii,
    write_ply_binary_le,
    write_stl_ascii,
    write_stl_binary,
    write_vti_binary_appended_bytes,
    write_vti_binary_appended_to_file,
    write_vti_image_data,
    write_vti_str,
    write_vtk,
    write_vtk_polydata,
    write_vtk_structured_grid,
    write_vtk_unstructured_grid,
    write_vtp_polydata,
    write_vtu_str,
    write_vtu_unstructured_grid,
    VtkReader,
    VtkWriter,
};

pub use domain::mesh_bridge::{indexed_mesh_to_poly, poly_to_indexed_mesh};
