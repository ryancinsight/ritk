//! Typed reader/writer handles that bind a compute backend to the NIfTI codec.

use crate::reader::read_nifti;
use crate::writer::write_nifti;
use coeus_core::ComputeBackend;
use ritk_codecs::into_io_error;
use ritk_codecs::sample::{Conversion, Sample};
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

    /// Read `path` as a 3-D image of physical values in `T`, converting the
    /// stored samples under `conversion`.
    ///
    /// # Errors
    ///
    /// Returns the error of [`read_nifti`] as an I/O error whose source chain
    /// carries every cause and whose kind is the root I/O failure's, or
    /// [`std::io::ErrorKind::Other`] when no I/O call failed.
    pub fn read<T: Sample, C: Conversion, P: AsRef<Path>>(
        &self,
        path: P,
        conversion: C,
    ) -> std::io::Result<Image<T, B, 3>> {
        read_nifti(path, &self.backend, conversion).map_err(into_io_error)
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

    /// Write `image` to `path` as NIfTI-1 samples of `T`.
    ///
    /// # Errors
    ///
    /// Returns the error of [`write_nifti`] as an I/O error whose source
    /// chain carries every cause and whose kind is the root I/O failure's, or
    /// [`std::io::ErrorKind::Other`] when no I/O call failed.
    pub fn write<T: Sample, P: AsRef<Path>>(
        &self,
        path: P,
        image: &Image<T, B, 3>,
    ) -> std::io::Result<()> {
        write_nifti(path, image, &self.backend).map_err(into_io_error)
    }
}
