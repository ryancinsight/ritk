//! VTK I/O module â€” free functions, VtkReader/VtkWriter wrappers, and sub-modules.

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

pub use reader::{read_vtk, read_vtk_flat};
pub use writer::{encode_vtk_flat, write_vtk};

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use std::path::Path;

/// Simple wrapper for reading VTK legacy structured-points images.
///
/// Does not implement `ritk_io::ImageReader`; that wrapper lives in `ritk-io`
/// to avoid orphan-rule violations.
pub struct VtkReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> VtkReader<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Read a VTK legacy structured-points file at `path`.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<Image<f32, B, 3>> {
        read_vtk(path, &self.backend)
    }
}

/// Simple wrapper for writing VTK legacy structured-points images.
///
/// Does not implement `ritk_io::ImageWriter`; that wrapper lives in `ritk-io`
/// to avoid orphan-rule violations.
pub struct VtkWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> VtkWriter<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Write a VTK legacy structured-points file to `path`.
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> anyhow::Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        write_vtk(path, image, &self.backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    #[test]
    fn native_scalar_round_trip_preserves_values_and_spatial_metadata() {
        let backend = SequentialBackend;
        let shape = [2, 2, 3];
        let values: Vec<f32> = (0..shape.iter().product())
            .map(|index| index as f32 * 0.25 - 1.0)
            .collect();
        let origin = Point::new([1.0, -2.0, 3.5]);
        let spacing = Spacing::new([0.5, 0.75, 1.25]);
        let image = Image::from_flat_on(
            values.clone(),
            shape,
            origin,
            spacing,
            Direction::identity(),
            &backend,
        )
        .expect("native image");
        let directory = tempdir().expect("temporary directory");
        let path = directory.path().join("roundtrip.vtk");

        VtkWriter::new(backend)
            .write(&path, &image)
            .expect("write VTK");
        let loaded = VtkReader::new(backend).read(&path).expect("read VTK");

        assert_eq!(loaded.shape(), shape);
        assert_eq!(loaded.data_slice().expect("contiguous image"), values);
        assert_eq!(*loaded.origin(), origin);
        assert_eq!(*loaded.spacing(), spacing);
        assert_eq!(*loaded.direction(), Direction::identity());
    }
}
