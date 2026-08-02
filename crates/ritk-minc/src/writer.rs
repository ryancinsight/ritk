//! MINC2 writer: HDF5-based 3-D volumetric image export.
//!
//! # HDF5 Structure Written
//!
//! ```text
//! / (root)
//!   └── minc-2.0/ (group)
//!       ├── dimensions/ (group)
//!       │   ├── xspace (group, attrs: start, step, length, direction_cosines)
//!       │   ├── yspace (group, same attrs)
//!       │   └── zspace (group, same attrs)
//!       └── image/ (group)
//!           └── 0/ (group)
//!               └── image (dataset: f32 voxel data, contiguous layout)
//!                   Contiguous little-endian f32 voxels
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
use anyhow::{bail, Context, Result};
use std::path::Path;

// ── Public API ────────────────────────────────────────────────────────────

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
pub fn write_minc<B, P>(image: &ritk_image::Image<f32, B, 3>, path: P, backend: &B) -> Result<()>
where
    B: coeus_core::ComputeBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    P: AsRef<Path>,
{
    let shape = image.shape();
    let origin = image.origin();
    let spacing = image.spacing();
    let direction = image.direction();
    let total_voxels = validate_geometry(shape, origin, spacing, direction)?;
    let f32_values = image.data_cow_on(backend);

    if f32_values.len() != total_voxels {
        bail!(
            "Tensor data length {} does not match shape {:?} ({total_voxels} voxels)",
            f32_values.len(),
            shape
        );
    }

    write_minc2_hdf5(
        path.as_ref(),
        &f32_values,
        shape,
        [origin[0], origin[1], origin[2]],
        [spacing[0], spacing[1], spacing[2]],
        direction,
    )?;

    Ok(())
}

fn validate_geometry(
    shape: [usize; 3],
    origin: &ritk_spatial::Point<3>,
    spacing: &ritk_spatial::Spacing<3>,
    direction: &ritk_spatial::Direction<3>,
) -> Result<usize> {
    let total_voxels = shape.into_iter().try_fold(1_usize, |product, extent| {
        product
            .checked_mul(extent)
            .context("MINC2 voxel count overflows usize")
    })?;
    if total_voxels == 0 {
        bail!("Cannot write empty MINC2 image (zero voxels)");
    }
    for (axis, extent) in shape.into_iter().enumerate() {
        i32::try_from(extent).with_context(|| {
            format!("MINC2 axis {axis} length {extent} exceeds the i32 length attribute")
        })?;
    }
    total_voxels
        .checked_mul(size_of::<f32>())
        .context("MINC2 voxel byte count overflows usize")?;

    for (axis, coordinate) in origin.as_slice().iter().copied().enumerate() {
        if !coordinate.is_finite() {
            bail!("MINC2 origin axis {axis} is not finite: {coordinate}");
        }
    }
    for axis in 0..3 {
        let step = spacing[axis];
        if !step.is_finite() || step <= 0.0 {
            bail!("MINC2 spacing axis {axis} must be finite and positive, got {step}");
        }
    }
    if direction.iter().any(|value| !value.is_finite()) {
        bail!("MINC2 direction matrix contains a non-finite value");
    }
    if !direction.is_orthogonal() {
        bail!("MINC2 direction matrix must contain orthonormal axis vectors");
    }

    Ok(total_voxels)
}

/// Typed writer wrapping `write_minc` for API consistency.
pub struct MincWriter<B: coeus_core::ComputeBackend> {
    backend: B,
}

impl<B: coeus_core::ComputeBackend> MincWriter<B> {
    /// Construct a writer that extracts image data through `backend`.
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    /// Write a 3-D image as a MINC2 file.
    pub fn write<P: AsRef<Path>>(&self, image: &ritk_image::Image<f32, B, 3>, path: P) -> Result<()>
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
