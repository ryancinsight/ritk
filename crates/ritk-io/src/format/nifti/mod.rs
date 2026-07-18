pub use ritk_nifti::{read_nifti_labels, write_nifti_labels};

use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;

use ritk_image::tensor::Tensor;
use std::path::Path;

fn native_to_legacy<B: Backend>(
    native: ritk_image::native::Image<f32, SequentialBackend, 3>,
    device: &B,
) -> Image<f32, B, 3> {
    let values = native.data_cow_on(&SequentialBackend);
    let tensor = Tensor::<f32, B>::from_slice_on(native.shape(), values.as_ref(), device);
    Image::new(
        tensor,
        *native.origin(),
        *native.spacing(),
        *native.direction(),
    )
}

/// Reads NIfTI through the native provider and converts at this legacy boundary.
pub fn read_nifti<B: Backend, P: AsRef<Path>>(path: P, device: &B) -> Result<Image<f32, B, 3>> {
    ritk_nifti::read_nifti(path, &SequentialBackend).map(|native| native_to_legacy(native, device))
}

/// Reads in-memory NIfTI through the native provider and converts at this boundary.
pub fn read_nifti_from_bytes<B: Backend>(bytes: &[u8], device: &B) -> Result<Image<f32, B, 3>> {
    ritk_nifti::read_nifti_from_bytes(bytes, &SequentialBackend)
        .map(|native| native_to_legacy(native, device))
}

/// Reads an in-memory NIfTI payload directly into a Coeus-backed image.
pub fn read_nifti_from_bytes_native<B: coeus_core::ComputeBackend>(
    bytes: &[u8],
    backend: &B,
) -> Result<ritk_image::native::Image<f32, B, 3>> {
    ritk_nifti::read_nifti_from_bytes(bytes, backend)
}

/// Writes a legacy image through the native NIfTI provider.
pub fn write_nifti<B: Backend, P: AsRef<Path>>(path: P, image: &Image<f32, B, 3>) -> Result<()> {
    let backend = SequentialBackend;
    let native = ritk_image::native::Image::from_flat_on(
        image.try_data_vec()?,
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        &backend,
    )?;
    ritk_nifti::write_nifti(path, &native, &backend)
}

/// Device-bound legacy NIfTI reader.
pub struct NiftiReader<B: Backend> {
    device: B,
}

impl<B: Backend> NiftiReader<B> {
    /// Creates a reader for `device`.
    pub fn new(device: B) -> Self {
        Self { device }
    }

    /// Reads NIfTI into the legacy image substrate.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
        read_nifti(path, &self.device).map_err(|error| std::io::Error::other(error.to_string()))
    }
}

/// Stateless legacy NIfTI writer.
pub struct NiftiWriter;

impl<B: Backend> crate::domain::ImageWriter<Image<f32, B, 3>> for NiftiWriter {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
        write_nifti(path, image).map_err(|error| std::io::Error::other(error.to_string()))
    }
}

/// Native-substrate NIfTI reader/writer contracts.
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound NIfTI reader.
    pub struct NiftiReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> NiftiReader<B> {
        /// Creates a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for NiftiReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_nifti::read_nifti(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound NIfTI writer.
    pub struct NiftiWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> NiftiWriter<B> {
        /// Creates a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for NiftiWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_nifti::write_nifti(path, image, &self.backend).map_err(to_io_err)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use coeus_core::SequentialBackend;
        use ritk_spatial::{Direction, Point, Spacing};
        use tempfile::tempdir;

        #[test]
        fn native_contract_round_trips_nifti() {
            let dims = [2usize, 3, 4];
            let voxels: Vec<f32> = (0..dims.iter().product())
                .map(|index| index as f32 * 0.25 - 2.0)
                .collect();
            let origin = Point::new([-11.0, 7.5, 3.25]);
            let spacing = Spacing::new([2.0, 1.5, 0.75]);
            let image = Image::from_flat_on(
                voxels.clone(),
                dims,
                origin,
                spacing,
                Direction::identity(),
                &SequentialBackend,
            )
            .expect("native image");

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("contract.nii");
            ImageWriter::write(&NiftiWriter::new(SequentialBackend), &path, &image)
                .expect("contract write");
            let loaded = ImageReader::read(&NiftiReader::new(SequentialBackend), &path)
                .expect("contract read");

            assert_eq!(loaded.shape(), dims);
            assert_eq!(loaded.data_slice().expect("contiguous"), voxels);
        }
    }
}
