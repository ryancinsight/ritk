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
use anyhow::{Context, Result};
use coeus_core::MoiraiBackend;
use ritk_core::image::Image;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::backend::Backend;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::BufWriter;
use std::path::Path;

/// Read a VTK legacy structured-points file into a native Coeus-backed image.
///
/// Native counterpart of [`read_vtk`]: both route through the substrate-free
/// [`ritk_vtk::read_vtk_flat`] decode core, so the returned voxel values and
/// geometry are byte-identical to the burn-backed reader — they differ only in
/// the image carrier. Returns `Image<f32, MoiraiBackend, 3>` with tensor shape
/// `[nz, ny, nx]`.
///
/// ## Convention
///
/// `read_vtk_flat` returns `dims` in VTK header **[nx, ny, nz]** order; the
/// native image is built with RITK tensor **[nz, ny, nx]** order (Z slowest, X
/// fastest), matching the burn reader. `origin` / `spacing` are VTK **[X, Y, Z]**
/// order and transfer directly.
///
/// # Errors
///
/// Returns an error under the same conditions as [`ritk_vtk::read_vtk_flat`], or
/// when the flat data cannot be laid out as a native image.
pub fn read_vtk_native<P: AsRef<Path>>(path: P) -> Result<NativeImage<f32, MoiraiBackend, 3>> {
    let (data, [nx, ny, nz], origin, spacing) = ritk_vtk::read_vtk_flat(path)?;
    NativeImage::<f32, MoiraiBackend, 3>::from_flat(
        data,
        [nz, ny, nx],
        Point::new(origin),
        Spacing::new(spacing),
        Direction::identity(),
    )
}

/// Write a native Coeus-backed image to a VTK legacy structured-points file.
///
/// Native counterpart of [`write_vtk`]: both delegate the byte-level encode to
/// the substrate-free [`ritk_vtk::encode_vtk_flat`], so given identical voxel
/// data and geometry the output file is byte-identical to the burn-backed
/// writer. Emits BINARY encoding with big-endian `f32` scalars (VTK legacy
/// version 3.0).
///
/// # Errors
///
/// Returns an error when the image data is not host-contiguous, or when the
/// file cannot be created or written.
pub fn write_vtk_native<P: AsRef<Path>>(
    image: &NativeImage<f32, MoiraiBackend, 3>,
    path: P,
) -> Result<()> {
    let path = path.as_ref();
    let dims = image.shape(); // [nz, ny, nx]
    let origin = image.origin(); // [X, Y, Z] order
    let spacing = image.spacing(); // [X, Y, Z] order
    let origin_arr = [origin[0], origin[1], origin[2]];
    let spacing_arr = [spacing[0], spacing[1], spacing[2]];
    let slice = image.data_slice()?;

    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create VTK file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    ritk_vtk::encode_vtk_flat(&mut writer, slice, dims, origin_arr, spacing_arr)
}

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

/// Differential parity between the native (Coeus `MoiraiBackend`) VTK path and
/// the authoritative burn path. The burn reader/writer is used purely as a
/// byte-level oracle; both routes share the substrate-free flat core, so
/// agreement verifies the native carrier construction, not the decode itself.
#[cfg(test)]
mod native_parity_tests {
    use super::{read_vtk, read_vtk_native, write_vtk, write_vtk_native};
    use burn_ndarray::NdArray;
    use coeus_core::MoiraiBackend;
    use ritk_core::image::Image;
    use ritk_image::native::Image as NativeImage;
    use ritk_image::tensor::backend::Backend;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    type BurnBackend = NdArray<f32>;

    /// Tensor shape `[nz, ny, nx] = [2, 2, 3]`, negative and fractional values to
    /// exercise the full big-endian f32 encoding (sign bit, mantissa).
    fn fixture() -> (Vec<f32>, [usize; 3], [f64; 3], [f64; 3]) {
        let values: Vec<f32> = (0..12).map(|v| v as f32 * 1.5 - 3.25).collect();
        (values, [2, 2, 3], [1.0, 2.0, 3.0], [0.5, 0.75, 1.25])
    }

    #[test]
    fn native_read_decodes_identically_to_burn_read() {
        let (values, shape, origin, spacing) = fixture();
        let dir = tempdir().unwrap();
        let path = dir.path().join("parity_read.vtk");

        // Produce a VTK fixture on disk via the authoritative burn writer.
        let device: <BurnBackend as Backend>::Device = Default::default();
        let tensor = Tensor::<BurnBackend, 3>::from_data(
            TensorData::new(values.clone(), Shape::new(shape)),
            &device,
        );
        let burn_img = Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        );
        write_vtk(&path, &burn_img).unwrap();

        // Oracle: burn reader.
        let burn_loaded = read_vtk::<BurnBackend, _>(&path, &device).unwrap();
        // Under test: native reader.
        let native_loaded = read_vtk_native(&path).unwrap();

        assert_eq!(native_loaded.shape(), burn_loaded.shape());
        assert_eq!(native_loaded.origin(), burn_loaded.origin());
        assert_eq!(native_loaded.spacing(), burn_loaded.spacing());

        let native_data = native_loaded.data_slice().unwrap();
        let burn_data = burn_loaded.try_data_vec().unwrap();
        // Byte-identical: f32 bit patterns must match exactly (no reinterpretation).
        assert_eq!(native_data, burn_data.as_slice());
    }

    #[test]
    fn native_write_emits_identical_bytes_to_burn_write() {
        let (values, shape, origin, spacing) = fixture();
        let dir = tempdir().unwrap();
        let burn_path = dir.path().join("parity_write_burn.vtk");
        let native_path = dir.path().join("parity_write_native.vtk");

        // Burn carrier + writer (oracle).
        let device: <BurnBackend as Backend>::Device = Default::default();
        let tensor = Tensor::<BurnBackend, 3>::from_data(
            TensorData::new(values.clone(), Shape::new(shape)),
            &device,
        );
        let burn_img = Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        );
        write_vtk(&burn_path, &burn_img).unwrap();

        // Native carrier + writer (under test).
        let native_img = NativeImage::<f32, MoiraiBackend, 3>::from_flat(
            values,
            shape,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        )
        .unwrap();
        write_vtk_native(&native_img, &native_path).unwrap();

        let burn_bytes = std::fs::read(&burn_path).unwrap();
        let native_bytes = std::fs::read(&native_path).unwrap();
        assert_eq!(native_bytes, burn_bytes);
    }
}
