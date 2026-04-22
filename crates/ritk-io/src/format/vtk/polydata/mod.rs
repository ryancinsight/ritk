//! VTK legacy POLYDATA format reader and writer.

pub mod reader;
pub mod writer;

pub use reader::read_vtk_polydata;
pub use writer::write_vtk_polydata;
