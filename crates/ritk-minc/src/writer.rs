//! MINC2 writer: HDF5-based 3-D volumetric image export.
//!
//! # HDF5 Structure Written
//!
//! ```text
//! / (root)
//!   Attributes: ident, minc_version
//!   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ minc-2.0/ (group)
//!       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ dimensions/ (group)
//!       Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ xspace (group, attrs: start, step, length, direction_cosines)
//!       Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ yspace (group, same attrs)
//!       Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ zspace (group, same attrs)
//!       Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ image/ (group)
//!           Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ 0/ (group)
//!               Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ image (dataset: f32 voxel data, contiguous layout)
//!                   Attributes: dimorder, valid_range, signtype, complete
//! ```
//!
//! # Data Type
//!
//! Voxel data is always written as little-endian IEEE 754 `f32`,
//! consistent with the RITK tensor representation.
//!
//! # direction_cosines
//!
//! Each dimension group carries a single `direction_cosines` attribute
//! encoded as a 1-D HDF5 float array of 3 `f64` values. This is the
//! format the MINC2 reader's `parse_dimension_attrs` expects.

use crate::hdf5_binary::write_minc2_hdf5;
use anyhow::{bail, Result};
use std::path::Path;

// Ã¢â€â‚¬Ã¢â€â‚¬ Public API Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

/// Write a 3-D `Image` as a MINC2 (.mnc) HDF5 file.
///
/// # Arguments
///
/// - `image`: the 3-D image to write.
/// - `path`: output file path (`.mnc` or `.mnc2` extension recommended).
///
/// # Errors
///
/// Returns `Err` when the file cannot be created, tensor data extraction
/// fails, or an I/O error occurs during HDF5 writing.
pub fn write_minc<B, P>(
    image: &ritk_image::native::Image<f32, B, 3>,
    path: P,
    backend: &B,
) -> Result<()>
where
    B: coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let [nz, ny, nx] = image.shape();
    let total_voxels = nz * ny * nx;

    if total_voxels == 0 {
        bail!("Cannot write empty image (zero voxels)");
    }

    let f32_values = image.data_cow_on(backend);

    if f32_values.len() != total_voxels {
        bail!(
            "Tensor data length {} does not match shape {:?} ({} voxels)",
            f32_values.len(),
            [nz, ny, nx],
            total_voxels
        );
    }

    let origin = image.origin();
    let spacing = image.spacing();
    let direction = image.direction();

    let mut raw_bytes: Vec<u8> = Vec::with_capacity(total_voxels * 4);
    for &v in f32_values.iter() {
        raw_bytes.extend_from_slice(&v.to_le_bytes());
    }

    write_minc2_hdf5(
        path.as_ref(),
        &raw_bytes,
        [nz, ny, nx],
        [origin[0], origin[1], origin[2]],
        [spacing[0], spacing[1], spacing[2]],
        direction,
    )?;

    Ok(())
}

/// Typed writer wrapping `write_minc` for API consistency.
pub struct MincWriter<B: coeus_core::ComputeBackend> {
    backend: B }

impl<B: coeus_core::ComputeBackend> MincWriter<B> {
    /// Construct a writer that extracts image data through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Write a 3-D image as a MINC2 file.
    pub fn write<P: AsRef<Path>>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        path: P,
    ) -> Result<()>
    where
        B: Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        write_minc(image, path, &self.backend)
    }
}

#[cfg(test)]
#[path = "tests_writer.rs"]
mod tests;
