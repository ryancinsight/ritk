//! MGH/MGZ format re-export shim.
//!
//! All functionality migrated to ritk-mgh crate.

pub use ritk_mgh::{read_mgh, write_mgh, MghReader, MghWriter};

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Burn types during coexistence and
/// folds away when the Burn path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader};
    use coeus_core::ComputeBackend;
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::MghReader`]).
    pub struct MghReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MghReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for MghReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_mgh::native::read_mgh(path, &self.backend).map_err(to_io_err)
        }
    }
}
