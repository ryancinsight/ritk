//! VTK I/O module — free functions, VtkReader/VtkWriter wrappers, and sub-modules.

pub mod polydata;
pub use polydata::{read_vtk_polydata, write_vtk_polydata};
pub mod polydata_xml;
pub use polydata_xml::{read_vtp_polydata, write_vtp_polydata};
pub mod image_xml;
pub use image_xml::{
    read_vti_binary_appended, read_vti_binary_appended_bytes, read_vti_image_data,
    write_vti_binary_appended_bytes, write_vti_binary_appended_to_file, write_vti_image_data,
    write_vti_str,
};
pub mod struct_grid;
pub mod unstruct_grid;
pub use struct_grid::{read_vtk_structured_grid, write_vtk_structured_grid};
pub use unstruct_grid::{read_vtk_unstructured_grid, write_vtk_unstructured_grid};
pub mod unstructured_xml;
pub use unstructured_xml::{
    read_vtu_unstructured_grid, write_vtu_str, write_vtu_unstructured_grid,
};

pub mod obj;
pub use obj::{read_obj_mesh, write_obj_mesh};
pub mod stl;
pub use stl::{read_stl_mesh, write_stl_ascii, write_stl_binary};
pub mod ply;
pub use ply::{read_ply_mesh, write_ply_ascii, write_ply_binary_le};
pub mod gltf;
pub use gltf::write_gltf;

pub mod mesh_indexed;
pub use mesh_indexed::{
    read_obj_indexed, read_ply_indexed, read_stl_indexed, write_indexed_glb, write_indexed_obj,
    write_indexed_ply, write_indexed_stl_ascii, write_indexed_stl_binary,
};

pub(crate) mod legacy_write_attribute;
pub(crate) mod read_helpers;
pub(crate) mod xml_helpers;
pub mod xml_write_attr;

pub mod reader;
pub mod writer;

pub use reader::read_vtk;
pub use writer::write_vtk;

use burn::tensor::backend::Backend;
use ritk_image::Image;
use std::path::Path;

/// Simple wrapper for reading VTK legacy structured-points images.
///
/// Does not implement `ritk_io::ImageReader`; that wrapper lives in `ritk-io`
/// to avoid orphan-rule violations.
pub struct VtkReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> VtkReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// Read a VTK legacy structured-points file at `path`.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<Image<B, 3>> {
        read_vtk(path, &self.device)
    }
}

/// Simple wrapper for writing VTK legacy structured-points images.
///
/// Does not implement `ritk_io::ImageWriter`; that wrapper lives in `ritk-io`
/// to avoid orphan-rule violations.
pub struct VtkWriter<B: Backend> {
    _marker: std::marker::PhantomData<fn() -> B>,
}

impl<B: Backend> Default for VtkWriter<B> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> VtkWriter<B> {
    /// Write a VTK legacy structured-points file to `path`.
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> anyhow::Result<()> {
        write_vtk(path, image)
    }
}
