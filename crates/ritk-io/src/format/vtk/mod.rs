//! VTK format facade for `ritk-io`.
//!
//! `ritk-vtk` owns the VTK data model and all VTK parsers/writers. This module
//! keeps only the `ritk_io::ImageReader` / `ImageWriter` trait adapters, which
//! are generic over `B: Backend` and compile through static dispatch.

pub use ritk_vtk::{
    read_obj_mesh, read_ply_mesh, read_stl_mesh, read_vti_binary_appended,
    read_vti_binary_appended_bytes, read_vti_image_data, read_vtk, read_vtk_polydata,
    read_vtk_structured_grid, read_vtk_unstructured_grid, read_vtp_polydata,
    read_vtu_unstructured_grid, write_gltf, write_obj_mesh, write_ply_ascii, write_ply_binary_le,
    write_stl_ascii, write_stl_binary, write_vti_binary_appended_bytes,
    write_vti_binary_appended_to_file, write_vti_image_data, write_vti_str, write_vtk,
    write_vtk_polydata, write_vtk_structured_grid, write_vtk_unstructured_grid, write_vtp_polydata,
    write_vtu_str, write_vtu_unstructured_grid,
};

pub mod image_xml {
    pub use ritk_vtk::io::image_xml::{
        read_vti_binary_appended, read_vti_binary_appended_bytes, read_vti_image_data,
        write_vti_binary_appended_bytes, write_vti_binary_appended_to_file, write_vti_image_data,
        write_vti_str,
    };
}

pub mod polydata {
    pub use ritk_vtk::io::polydata::{read_vtk_polydata, write_vtk_polydata};
}

pub mod polydata_xml {
    pub use ritk_vtk::io::polydata_xml::{read_vtp_polydata, write_vtp_polydata};
}

pub mod struct_grid {
    pub use ritk_vtk::io::struct_grid::{read_vtk_structured_grid, write_vtk_structured_grid};
}

pub mod unstruct_grid {
    pub use ritk_vtk::io::unstruct_grid::{
        read_vtk_unstructured_grid, write_vtk_unstructured_grid,
    };
}

pub mod unstructured_xml {
    pub use ritk_vtk::io::unstructured_xml::{
        read_vtu_unstructured_grid, write_vtu_str, write_vtu_unstructured_grid,
    };
}

pub mod mesh_writer;
pub use mesh_writer::{mesh_to_vtk_string, write_mesh_as_vtk};

use crate::domain::{ImageReader, ImageWriter};
use ritk_core::image::Image;
use ritk_image::tensor::backend::Backend;
use std::path::Path;

/// DIP boundary implementing `ImageReader` for VTK legacy structured points.
pub struct VtkReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> VtkReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> ImageReader<Image<B, 3>> for VtkReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_vtk(path, &self.device).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// DIP boundary implementing `ImageWriter` for VTK legacy structured points.
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

impl<B: Backend> ImageWriter<Image<B, 3>> for VtkWriter<B> {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_vtk(path, image).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    #[test]
    fn dip_reader_writer_roundtrip_uses_authoritative_vtk_backend() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("facade.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let tensor = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(values.clone(), Shape::new([2, 2, 3])),
            &device,
        );
        let image = Image::new(
            tensor,
            Point::new([1.0, 2.0, 3.0]),
            Spacing::new([0.5, 0.75, 1.25]),
            Direction::identity(),
        );

        VtkWriter::<TestBackend>::default()
            .write(&path, &image)
            .unwrap();
        let loaded = VtkReader::<TestBackend>::new(device).read(&path).unwrap();

        assert_eq!(loaded.shape(), [2, 2, 3]);
        assert_eq!(*loaded.spacing(), Spacing::new([0.5, 0.75, 1.25]));
        assert_eq!(*loaded.origin(), Point::new([1.0, 2.0, 3.0]));
        loaded.with_data_slice(|loaded_vals| {
            assert_eq!(loaded_vals, values.as_slice());
        });
    }
}
