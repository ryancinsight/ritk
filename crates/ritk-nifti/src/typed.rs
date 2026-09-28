//! Typed reader/writer handles that bind a compute backend to the NIfTI codec.

use crate::reader::read_nifti;
use crate::writer::write_nifti;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use std::path::Path;

/// DIP boundary executing strict spatial metadata preservation over standard NIfTI datasets.
pub struct NiftiReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> NiftiReader<B> {
    /// Create a reader that decodes through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Read `path` as a 3-D `f32` image.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
        read_nifti(path, &self.backend).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard NIfTI datasets.
pub struct NiftiWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> NiftiWriter<B> {
    /// Create a writer that encodes through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Write `image` to `path` as NIfTI-1.
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        write_nifti(path, image, &self.backend).map_err(|e| std::io::Error::other(e.to_string()))
    }
}
