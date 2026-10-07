//! MRtrix `.mif` image dispatch.
//!
//! The codec itself is owned by [`ritk_mif`]; this module is the route that
//! lets the unified [`crate::domain::ImageReader`]/[`crate::domain::ImageWriter`]
//! contract, and therefore [`crate::ImageFormat::Mif`] and the native dispatch,
//! reach it without a consumer depending on `ritk-mif` directly.

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Coeus types during coexistence and
/// folds away when the Coeus path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Coeus `MifReader`).
    pub struct MifReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MifReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for MifReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_mif::read_mif(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Coeus writer).
    pub struct MifWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MifWriter<B> {
        /// Create a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for MifWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_mif::write_mif(path, image, &self.backend).map_err(to_io_err)
        }
    }
}
