//! MGH/MGZ format re-export shim.
//!
//! All functionality migrated to ritk-mgh crate.

pub use ritk_mgh::{read_mgh, write_mgh, MghReader, MghWriter};

#[cfg(feature = "coeus")]
pub use coeus::{CoeusMghReader};

/// Coeus-typed reader implementors for the [`crate::domain::coeus`] contract
/// (ADR 0002 cutover step 2 — format coverage).
#[cfg(feature = "coeus")]
mod coeus {
    use crate::domain::coeus::{to_io_err, CoeusImageReader};
    use coeus_core::ComputeBackend;
    use ritk_image::coeus::Image;
    use std::path::Path;

    /// Backend-bound Coeus reader (counterpart of [`super::MghReader`]).
    pub struct CoeusMghReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> CoeusMghReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> CoeusImageReader<f32, B, 3> for CoeusMghReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_mgh::read_mgh_coeus(path, &self.backend).map_err(to_io_err)
        }
    }
}
