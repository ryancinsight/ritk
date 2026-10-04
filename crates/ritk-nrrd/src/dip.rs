//! Dependency-inversion boundary over the compute-image NRRD entry points.
//!
//! These adapters execute strict spatial metadata preservation over standard
//! NRRD datasets through the shared [`ComputeBackend`] tensor path.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use std::path::Path;

use crate::{read_nrrd, write_nrrd};

/// DIP boundary executing strict spatial metadata preservation over standard NRRD datasets.
pub struct NrrdDipReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> NrrdDipReader<B> {
    /// Attach the reader to a compute backend.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Read a NRRD file as a compute-ready single-precision image.
    pub fn read<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<Image<f32, B, 3>> {
        read_nrrd(path, &self.backend)
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard NRRD datasets.
pub struct NrrdDipWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> NrrdDipWriter<B> {
    /// Attach the writer to a compute backend.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Write a single-precision image as a NRRD file.
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> anyhow::Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        write_nrrd(path, image, &self.backend)
    }
}
