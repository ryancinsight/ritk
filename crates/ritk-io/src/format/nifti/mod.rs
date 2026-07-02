//! Re-export NIfTI functionality from ritk-nifti.
//!
//! This module provides backward-compatible access to NIfTI readers and writers
//! that are now implemented in the dedicated ritk-nifti crate.

pub use ritk_nifti::{
    read_nifti, read_nifti_from_bytes, read_nifti_labels, write_nifti, write_nifti_labels,
    NiftiReader, NiftiWriter,
};

#[cfg(feature = "coeus")]
pub use coeus::{CoeusNiftiReader, CoeusNiftiWriter};

/// Coeus-typed NIfTI reader/writer — the first implementors of the
/// [`crate::domain::coeus`] contract (ADR 0002 cutover step 2).
#[cfg(feature = "coeus")]
mod coeus {
    use crate::domain::coeus::{to_io_err, CoeusImageReader, CoeusImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::coeus::Image;
    use std::path::Path;

    /// Backend-bound Coeus NIfTI reader (counterpart of [`super::NiftiReader`]).
    pub struct CoeusNiftiReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> CoeusNiftiReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> CoeusImageReader<f32, B, 3> for CoeusNiftiReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_nifti::read_nifti_coeus(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Coeus NIfTI writer (counterpart of [`super::NiftiWriter`]).
    pub struct CoeusNiftiWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> CoeusNiftiWriter<B> {
        /// Create a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> CoeusImageWriter<f32, B, 3> for CoeusNiftiWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_nifti::write_nifti_coeus(path, image, &self.backend).map_err(to_io_err)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use coeus_core::SequentialBackend;
        use ritk_spatial::{Direction, Point, Spacing};
        use tempfile::tempdir;

        /// Trait-dispatched round trip: write through `CoeusImageWriter`, read
        /// back through `CoeusImageReader`, exact voxel + metadata parity —
        /// the Coeus I/O contract is usable end-to-end, not just nominal.
        #[test]
        fn coeus_contract_round_trips_nifti() {
            let dims = [2usize, 3, 4];
            let n = dims[0] * dims[1] * dims[2];
            let voxels: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 2.0).collect();
            let origin = Point::new([-11.0, 7.5, 3.25]);
            let spacing = Spacing::new([2.0, 1.5, 0.75]);
            let image = Image::from_flat_on(
                voxels.clone(),
                dims,
                origin,
                spacing,
                Direction::identity(),
                &SequentialBackend,
            )
            .expect("coeus image");

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("contract.nii");

            let writer = CoeusNiftiWriter::new(SequentialBackend);
            CoeusImageWriter::write(&writer, &path, &image).expect("contract write");

            let reader = CoeusNiftiReader::new(SequentialBackend);
            let loaded = CoeusImageReader::read(&reader, &path).expect("contract read");

            assert_eq!(loaded.shape(), dims);
            assert_eq!(
                loaded.data_slice().expect("contiguous"),
                voxels.as_slice(),
                "voxels must round-trip exactly through the trait contract"
            );
            let sp = loaded.spacing();
            let og = loaded.origin();
            for k in 0..3 {
                assert!((sp[k] - spacing[k]).abs() < 1e-5, "spacing[{k}] round-trip");
                assert!((og[k] - origin[k]).abs() < 1e-4, "origin[{k}] round-trip");
            }
        }
    }
}
