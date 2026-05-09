//! Re-export NIfTI functionality from ritk-nifti.
//!
//! This module provides backward-compatible access to NIfTI readers and writers
//! that are now implemented in the dedicated ritk-nifti crate.

pub use ritk_nifti::{
    read_nifti,
    read_nifti_from_bytes,
    read_nifti_labels,
    write_nifti,
    write_nifti_labels,
};

use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

/// DIP boundary executing strict spatial metadata preservation over standard NIfTI datasets.
pub struct NiftiReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> NiftiReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
    
    pub fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_nifti(path, &self.device)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard NIfTI datasets.
pub struct NiftiWriter<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for NiftiWriter<B> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> NiftiWriter<B> {
    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_nifti(path, image)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

