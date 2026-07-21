//! VTK data model: types, pipeline traits, scene graph, observer system,
//! modification-time tracking, smart mapper, multi-block datasets, and
//! concrete geometry filters.

pub mod filters;
pub mod mapper;
pub mod mesh_bridge;
pub mod mtime;
pub mod multi_block;
pub mod observer;
pub mod vtk_data_object;
pub mod vtk_pipeline;
pub mod vtk_scene;

pub use filters::{ComputeNormalsFilter, SmoothFilter, ThresholdFilter};
pub use iris::color::NamedColorMap;
pub use mapper::{PolygonMode, ScalarVisibility, SurfaceMapper, VtkLookupTable, VtkMapper};
pub use mtime::{Modifiable, ModifiedTime};
pub use multi_block::{Block, LeafIter, VtkMultiBlockDataSet};
pub use observer::{EventHandlers, EventId, Observable, ObserverCallback, ObserverTag};
pub use vtk_data_object::{
    AttributeArray, VtkCellType, VtkDataObject, VtkImageData, VtkPolyData, VtkStructuredGrid,
    VtkUnstructuredGrid,
};
pub use vtk_pipeline::{VtkFilter, VtkPipeline, VtkSink, VtkSource};
pub use vtk_scene::{RenderProperties, Visibility, VtkActor, VtkScene};
