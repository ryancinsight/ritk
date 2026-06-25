//! MINC2 writer: HDF5-based 3-D volumetric image export.
//!
//! # HDF5 Structure Written
//!
//! ```text
//! / (root)
//!   Attributes: ident, minc_version
//!   â””â”€â”€ minc-2.0/ (group)
//!       â”œâ”€â”€ dimensions/ (group)
//!       â”‚   â”œâ”€â”€ xspace (group, attrs: start, step, length, direction_cosines)
//!       â”‚   â”œâ”€â”€ yspace (group, same attrs)
//!       â”‚   â””â”€â”€ zspace (group, same attrs)
//!       â””â”€â”€ image/ (group)
//!           â””â”€â”€ 0/ (group)
//!               â””â”€â”€ image (dataset: f32 voxel data, contiguous layout)
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
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

// â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
pub fn write_minc<B: Backend, P: AsRef<Path>>(image: &Image<B, 3>, path: P) -> Result<()> {
    let path = path.as_ref();

    let shape = image.data().shape();
    let dims = shape.dims;
    if dims.len() != 3 {
        bail!(
            "MINC2 writer requires a 3-D image, got {} dimensions",
            dims.len()
        );
    }

    let nz = dims[0];
    let ny = dims[1];
    let nx = dims[2];
    let total_voxels = nz * ny * nx;

    if total_voxels == 0 {
        bail!("Cannot write empty image (zero voxels)");
    }

    let tensor_data = image.data().to_data();
    let f32_values: Vec<f32> = tensor_data
        .to_vec()
        .map_err(|e| anyhow::anyhow!("Failed to extract f32 data from tensor: {:?}", e))?;

    if f32_values.len() != total_voxels {
        bail!(
            "Tensor data length {} does not match shape {:?} ({} voxels)",
            f32_values.len(),
            dims,
            total_voxels
        );
    }

    let origin = image.origin();
    let spacing = image.spacing();
    let direction = image.direction();

    let mut raw_bytes: Vec<u8> = Vec::with_capacity(total_voxels * 4);
    for &v in &f32_values {
        raw_bytes.extend_from_slice(&v.to_le_bytes());
    }

    write_minc2_hdf5(
        path,
        &raw_bytes,
        [nz, ny, nx],
        [origin[0], origin[1], origin[2]],
        [spacing[0], spacing[1], spacing[2]],
        direction,
    )?;

    Ok(())
}

/// Typed writer wrapping `write_minc` for API consistency.
pub struct MincWriter;

impl MincWriter {
    /// Write a 3-D image as a MINC2 file.
    pub fn write<B: Backend, P: AsRef<Path>>(image: &Image<B, 3>, path: P) -> Result<()> {
        write_minc(image, path)
    }
}

#[cfg(test)]
#[path = "tests_writer.rs"]
mod tests;
