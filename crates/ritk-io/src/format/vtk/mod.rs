//! VTK format facade for `ritk-io`.
//!
//! `ritk-vtk` owns the VTK parsers and encoders. This module only binds those
//! operations to the native `ImageReader` and `ImageWriter` contracts.

pub use ritk_vtk::{
    read_obj_mesh, read_ply_mesh, read_stl_mesh, read_vti_binary_appended,
    read_vti_binary_appended_bytes, read_vti_image_data, read_vtk_polydata,
    read_vtk_structured_grid, read_vtk_unstructured_grid, read_vtp_polydata,
    read_vtu_unstructured_grid, write_gltf, write_obj_mesh, write_ply_ascii, write_ply_binary_le,
    write_stl_ascii, write_stl_binary, write_vti_binary_appended_bytes,
    write_vti_binary_appended_to_file, write_vti_image_data, write_vti_str, write_vtk_polydata,
    write_vtk_structured_grid, write_vtk_unstructured_grid, write_vtp_polydata, write_vtu_str,
    write_vtu_unstructured_grid,
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

/// Native VTK image reader and writer contracts.
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::Image;
    use std::path::Path;

    /// Backend-bound VTK reader.
    pub struct VtkReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> VtkReader<B> {
        /// Creates a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for VtkReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_vtk::read_vtk(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound VTK writer.
    pub struct VtkWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> VtkWriter<B> {
        /// Creates a writer that extracts host data through `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for VtkWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_vtk::write_vtk(path, image, &self.backend).map_err(to_io_err)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use coeus_core::SequentialBackend;
        use ritk_spatial::{Direction, Point, Spacing};
        use tempfile::tempdir;

        #[test]
        fn native_contract_round_trips_vtk_values_and_geometry() {
            let image = Image::from_flat_on(
                (0..12).map(|value| value as f32 * 0.5 - 2.0).collect(),
                [2, 2, 3],
                Point::new([1.0, 2.0, 3.0]),
                Spacing::new([0.5, 0.75, 1.25]),
                Direction::identity(),
                &SequentialBackend,
            )
            .expect("native VTK fixture");
            let directory = tempdir().expect("temporary directory");
            let path = directory.path().join("roundtrip.vtk");

            ImageWriter::write(&VtkWriter::new(SequentialBackend), &path, &image)
                .expect("VTK write");
            let loaded =
                ImageReader::read(&VtkReader::new(SequentialBackend), &path).expect("VTK read");

            assert_eq!(loaded.shape(), [2, 2, 3]);
            assert_eq!(loaded.origin(), image.origin());
            assert_eq!(loaded.spacing(), image.spacing());
            assert_eq!(
                loaded.data_slice().expect("contiguous data"),
                image.data_slice().expect("contiguous fixture")
            );
        }
    }
}
