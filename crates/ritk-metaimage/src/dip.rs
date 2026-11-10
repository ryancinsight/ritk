use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_codecs::ImageReadBudget;
use ritk_image::Image;
use std::path::Path;

use crate::{read_metaimage, write_metaimage, MetaImageError};

/// DIP boundary executing strict spatial metadata preservation over standard MetaImage datasets.
pub struct MetaImageDipReader<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MetaImageDipReader<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn read<P: AsRef<Path>>(
        &self,
        path: P,
        budget: ImageReadBudget,
    ) -> Result<Image<f32, B, 3>, MetaImageError> {
        read_metaimage(path, &self.backend, budget)
    }
}

/// DIP boundary executing strict spatial metadata preservation over standard MetaImage datasets.
pub struct MetaImageDipWriter<B: ComputeBackend> {
    backend: B,
}

impl<B: ComputeBackend> MetaImageDipWriter<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> anyhow::Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        write_metaimage(path, image, &self.backend)
    }
}
