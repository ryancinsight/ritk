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

/// Reads MetaImage through the native provider and converts at this legacy boundary.
pub fn read_metaimage<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    ritk_metaimage::read_metaimage(path, &SequentialBackend)
        .map(|native| native_to_legacy(native, device))
}

/// Writes a legacy image through the native MetaImage provider.
pub fn write_metaimage<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let native = legacy_metadata_to_native(image, image.try_data_vec()?)?;
    ritk_metaimage::write_metaimage(path, &native, &SequentialBackend)
}

/// Writes caller-provided voxels with legacy image metadata through the native provider.
pub fn write_metaimage_with_data<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<B, 3>,
    values: &[f32],
) -> Result<()> {
    let native = legacy_metadata_to_native(image, values.to_vec())?;
    ritk_metaimage::write_metaimage_with_data(path, &native, values)
}

/// Stateless legacy MetaImage reader.
pub struct MetaImageReader;

impl MetaImageReader {
    /// Reads MetaImage into the legacy image substrate.
    pub fn read<B: Backend, P: AsRef<Path>>(
        &self,
        path: P,
        device: &B::Device,
    ) -> Result<Image<B, 3>> {
        read_metaimage(path, device)
    }
}

/// Stateless legacy MetaImage writer.
pub struct MetaImageWriter;

impl<B: Backend> ImageWriter<Image<B, 3>> for MetaImageWriter {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_metaimage(path, image).map_err(|error| std::io::Error::other(error.to_string()))
    }
}

/// Native-substrate implementors of [`crate::domain::ImageReader`].
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound native reader.
    pub struct MetaImageReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MetaImageReader<B> {
        /// Creates a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for MetaImageReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_metaimage::read_metaimage(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound native writer.
    pub struct MetaImageWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MetaImageWriter<B> {
        /// Creates a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for MetaImageWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_metaimage::write_metaimage(path, image, &self.backend).map_err(to_io_err)
        }
    }
}
