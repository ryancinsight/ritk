pub mod reader;
pub mod writer;

pub use reader::read_nifti;
pub use writer::write_nifti;

use crate::domain::{ImageReader, ImageWriter};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

/// DIP boundary executing strict `ImageReader` invariants over standard NifTI datasets.
pub struct NiftiReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> NiftiReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> ImageReader<B, 3> for NiftiReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_nifti(path, &self.device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

/// DIP boundary executing strict `ImageWriter` invariants over standard NifTI datasets.
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

impl<B: Backend> ImageWriter<B, 3> for NiftiWriter<B> {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_nifti(path, image)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

#[cfg(test)]
mod tests;
