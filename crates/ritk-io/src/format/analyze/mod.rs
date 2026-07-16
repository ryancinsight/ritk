//! Analyze 7.5 format dispatch.

use anyhow::{Context, Result};
use coeus_core::SequentialBackend;
use ritk_core::image::Image as BurnImage;
use ritk_image::tensor::backend::Backend;
use coeus_tensor::Tensor;
use ritk_image::tensor::{Shape, TensorData};
use std::marker::PhantomData;
use std::path::Path;

/// Legacy Burn bridge for callers that have not migrated to the native image
/// contract. The Analyze leaf crate is native-only; this bridge performs the
/// remaining tensor construction at the `ritk-io` consumer boundary.
pub fn read_analyze<B: Backend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<BurnImage<f32, B, 3>> {
    let backend = SequentialBackend;
    let native = ritk_analyze::read_analyze(path, &backend)?;
    let shape = native.shape();
    let origin = *native.origin();
    let spacing = *native.spacing();
    let direction = *native.direction();
    let data = native.data_vec_on(&backend);
    let tensor = Tensor::<B, 3>::from_data((data, (shape)), device);
    Ok(BurnImage::new(tensor, origin, spacing, direction))
}

/// Legacy Burn bridge for callers that have not migrated to the native image
/// contract. The Analyze leaf crate owns serialization and receives a native
/// image built from the caller's Burn image data.
pub fn write_analyze<B: Backend, P: AsRef<Path>>(path: P, image: &BurnImage<f32, B, 3>) -> Result<()> {
    let backend = SequentialBackend;
    let values = image
        .try_data_vec()
        .context("Analyze writer requires f32 image data")?;
    let native = ritk_image::native::Image::from_flat_on(
        values,
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        &backend,
    )?;
    ritk_analyze::write_analyze(path, &native, &backend)
}

/// Read-side wrapper type implementing the Burn image reader contract.
pub struct AnalyzeReader<B: Backend> {
    device: B::Device,
    _phantom: PhantomData<fn() -> B>,
}

impl<B: Backend> AnalyzeReader<B> {
    /// Construct a reader bound to `device`.
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            _phantom: PhantomData,
        }
    }

    /// Read an Analyze image into a Burn-backed image.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> Result<BurnImage<f32, B, 3>> {
        read_analyze(path, &self.device)
    }
}

/// Write-side wrapper type implementing the Burn image writer contract.
pub struct AnalyzeWriter<B: Backend> {
    _phantom: PhantomData<fn() -> B>,
}

impl<B: Backend> AnalyzeWriter<B> {
    /// Construct a writer.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Write a Burn-backed Analyze image.
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &BurnImage<f32, B, 3>) -> Result<()> {
        write_analyze(path, image)
    }
}

impl<B: Backend> Default for AnalyzeWriter<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> crate::domain::ImageReader<BurnImage<f32, B, 3>> for AnalyzeReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<BurnImage<f32, B, 3>> {
        read_analyze(path, &self.device).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

impl<B: Backend> crate::domain::ImageWriter<BurnImage<f32, B, 3>> for AnalyzeWriter<B> {
    fn write<P: AsRef<Path>>(&self, path: P, image: &BurnImage<f32, B, 3>) -> std::io::Result<()> {
        write_analyze(path, image).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the module
/// itself disambiguates from the remaining Burn bridge during coexistence and
/// folds away when the bridge is deleted.
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::AnalyzeReader`]).
    pub struct AnalyzeReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> AnalyzeReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for AnalyzeReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_analyze::read_analyze(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Burn [`super::AnalyzeWriter`]).
    pub struct AnalyzeWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> AnalyzeWriter<B> {
        /// Create a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for AnalyzeWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_analyze::write_analyze(path, image, &self.backend).map_err(to_io_err)
        }
    }
}
