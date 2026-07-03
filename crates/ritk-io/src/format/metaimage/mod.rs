pub use ritk_metaimage::{
    read_metaimage, write_metaimage, write_metaimage_with_data, MetaImageReader, MetaImageWriter,
};

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Burn types during coexistence and
/// folds away when the Burn path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::MetaImageReader`]).
    pub struct MetaImageReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MetaImageReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for MetaImageReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_metaimage::native::read_metaimage(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Burn writer).
    pub struct MetaImageWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MetaImageWriter<B> {
        /// Create a writer that extracts host data via `backend`.
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
            ritk_metaimage::native::write_metaimage(path, image, &self.backend).map_err(to_io_err)
        }
    }
}
