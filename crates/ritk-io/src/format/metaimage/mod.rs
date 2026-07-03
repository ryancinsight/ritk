pub use ritk_metaimage::{
    read_metaimage, write_metaimage, write_metaimage_with_data, MetaImageReader, MetaImageWriter,
};

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Burn types during coexistence and
/// folds away when the Burn path is deleted (ADR 0002).
#[cfg(feature = "coeus")]
pub mod native {
    use crate::domain::{to_io_err, ImageReader};
    use coeus_core::ComputeBackend;
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
            ritk_metaimage::read_metaimage_coeus(path, &self.backend).map_err(to_io_err)
        }
    }
}
