use crate::domain::ImageWriter;
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;

use ritk_image::tensor::Tensor;
use std::path::Path;

/// Reads MGH/MGZ through the native provider and converts at this legacy boundary.
pub fn read_mgh<B: Backend, P: AsRef<Path>>(path: P, device: &B) -> Result<Image<f32, B, 3>> {
    let native = ritk_mgh::read_mgh(path, &SequentialBackend)?;
    let values = native.data_cow_on(&SequentialBackend);
    let tensor = Tensor::<f32, B>::from_slice_on(native.shape(), values.as_ref(), device);
    Image::new(
        tensor,
        *native.origin(),
        *native.spacing(),
        *native.direction(),
    )
}

/// Writes a legacy image through the native MGH provider.
pub fn write_mgh<B: Backend, P: AsRef<Path>>(image: &Image<f32, B, 3>, path: P) -> Result<()> {
    let backend = SequentialBackend;
    let native = ritk_image::Image::from_flat_on(
        image.try_data_vec()?,
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        &backend,
    )?;
    ritk_mgh::write_mgh(&native, path, &backend)
}

/// Stateless legacy MGH/MGZ reader.
pub struct MghReader;

impl MghReader {
    /// Reads MGH/MGZ into the legacy image substrate.
    pub fn read<B: Backend, P: AsRef<Path>>(path: P, device: &B) -> Result<Image<f32, B, 3>> {
        read_mgh(path, device)
    }
}

/// Stateless legacy MGH/MGZ writer.
pub struct MghWriter;

impl<B: Backend> ImageWriter<Image<f32, B, 3>> for MghWriter {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
        write_mgh(image, path).map_err(|error| std::io::Error::other(error.to_string()))
    }
}

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Burn types during coexistence and
/// folds away when the Burn path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::MghReader`]).
    pub struct MghReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MghReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for MghReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_mgh::read_mgh(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Burn writer).
    pub struct MghWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MghWriter<B> {
        /// Create a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for MghWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_mgh::write_mgh(image, path, &self.backend).map_err(to_io_err)
        }
    }
}
