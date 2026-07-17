use crate::domain::ImageWriter;
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_image::tensor::backend::Backend;

use ritk_image::tensor::{Shape, Tensor, TensorData};
use std::path::Path;

fn native_to_legacy<B: Backend>(
    native: ritk_image::native::Image<f32, SequentialBackend, 3>,
    device: &B::Device,
) -> Image<B, 3> {
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(
            native.data_cow_on(&SequentialBackend).into_owned(),
            Shape::new(native.shape()),
        ),
        device,
    );
    Image::new(
        tensor,
        *native.origin(),
        *native.spacing(),
        *native.direction(),
    )
}

fn legacy_metadata_to_native<B: Backend>(
    image: &Image<B, 3>,
    values: Vec<f32>,
) -> Result<ritk_image::native::Image<f32, SequentialBackend, 3>> {
    ritk_image::native::Image::from_flat_on(
        values,
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        &SequentialBackend,
    )
}

/// Reads NRRD through the native provider and converts at this legacy boundary.
pub fn read_nrrd<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    ritk_nrrd::read_nrrd(path, &SequentialBackend).map(|native| native_to_legacy(native, device))
}

/// Writes a legacy image through the native NRRD provider.
pub fn write_nrrd<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let native = legacy_metadata_to_native(image, image.try_data_vec()?)?;
    ritk_nrrd::write_nrrd(path, &native, &SequentialBackend)
}

/// Writes caller-provided voxels with legacy image metadata through the native provider.
pub fn write_nrrd_with_data<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<B, 3>,
    values: &[f32],
) -> Result<()> {
    let native = legacy_metadata_to_native(image, values.to_vec())?;
    ritk_nrrd::write_nrrd_with_data(path, &native, values)
}

/// Stateless legacy NRRD reader.
pub struct NrrdReader;

impl NrrdReader {
    /// Reads NRRD into the legacy image substrate.
    pub fn read<B: Backend, P: AsRef<Path>>(
        &self,
        path: P,
        device: &B::Device,
    ) -> Result<Image<B, 3>> {
        read_nrrd(path, device)
    }
}

/// Stateless legacy NRRD writer.
pub struct NrrdWriter;

impl<B: Backend> ImageWriter<Image<B, 3>> for NrrdWriter {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_nrrd(path, image).map_err(|error| std::io::Error::other(error.to_string()))
    }
}

/// Native-substrate implementors of [`crate::domain::ImageReader`].
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound native reader.
    pub struct NrrdReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> NrrdReader<B> {
        /// Creates a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for NrrdReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_nrrd::read_nrrd(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound native writer.
    pub struct NrrdWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> NrrdWriter<B> {
        /// Creates a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for NrrdWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_nrrd::write_nrrd(path, image, &self.backend).map_err(to_io_err)
        }
    }
}
