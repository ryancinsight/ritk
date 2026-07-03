//! Analyze 7.5 format re-export shim.
//!
//! All functionality migrated to ritk-analyze crate.

pub use ritk_analyze::{read_analyze, write_analyze, AnalyzeReader, AnalyzeWriter};

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
            ritk_analyze::native::read_analyze(path, &self.backend).map_err(to_io_err)
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
            ritk_analyze::native::write_analyze(path, image, &self.backend).map_err(to_io_err)
        }
    }
}
