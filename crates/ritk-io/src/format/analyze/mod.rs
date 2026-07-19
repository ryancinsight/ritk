//! Analyze 7.5 format dispatch.

use crate::domain::{to_io_err, ImageReader, ImageWriter};
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use std::path::Path;

/// Backend-bound Analyze reader.
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

/// Backend-bound Analyze writer.
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

/// Read an Analyze image on `backend`.
pub fn read_analyze<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> anyhow::Result<Image<f32, B, 3>> {
    ritk_analyze::read_analyze(path, backend)
}

/// Write an Analyze image using its backend.
pub fn write_analyze<B, P>(path: P, image: &Image<f32, B, 3>) -> anyhow::Result<()>
where
    B: ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    ritk_analyze::write_analyze(path, image, &B::default())
}
