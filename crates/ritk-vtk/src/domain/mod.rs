//! VTK data model: types, pipeline traits, scene graph, observer system,
//! modification-time tracking, smart mapper, multi-block datasets, and
//! concrete geometry filters.

pub mod mesh_bridge;
pub mod vtk_data_object;
pub mod vtk_pipeline;
pub mod vtk_scene;
pub mod mtime;
pub mod observer;
pub mod mapper;
pub mod multi_block;
pub mod filters;

pub use vtk_data_object::{
    AttributeArray, VtkCellType, VtkDataObject, VtkImageData, VtkPolyData, VtkStructuredGrid,
    VtkUnstructuredGrid,
};
pub use vtk_pipeline::{VtkFilter, VtkPipeline, VtkSink, VtkSource};
pub use vtk_scene::{RenderProperties, Visibility, VtkActor, VtkScene};
pub use mtime::{ModifiedTime, Modifiable};
pub use observer::{EventHandlers, EventId, Observable, ObserverCallback, ObserverTag};
pub use mapper::{ColormapPreset, PolygonMode, ScalarVisibility, SurfaceMapper, VtkLookupTable, VtkMapper};
pub use multi_block::{Block, LeafIter, VtkMultiBlockDataSet};
pub use filters::{ComputeNormalsFilter, SmoothFilter, ThresholdFilter};
