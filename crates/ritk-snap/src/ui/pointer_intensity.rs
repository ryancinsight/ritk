//! Pointer-voxel intensity lookup SSOT.
//!
//! This module provides a pure, testable function for reading the voxel intensity
//! value at a given 3D image coordinate, with automatic boundary clamping and
//! out-of-bounds safety.
//!
//! # Intensity semantics
//!
//! The intensity returned is the raw normalized voxel value stored in the loaded
//! volume (typically HU for CT, or relative intensity for MR). Boundary voxels
//! return their actual value; out-of-bounds accesses return `0.0`.

use crate::LoadedVolume;

/// Retrieve the voxel intensity value at the given 3D image coordinate.
///
/// # Parameters
///
/// - `vol` — loaded 3D medical image volume.
/// - `voxel` — 3D voxel index as `[d, r, c]` (depth/z, row/y, column/x) in pixels.
///
/// # Returns
///
/// The normalized intensity value (HU or relative) at the voxel, or `0.0` if the
/// coordinate lies outside the volume bounds.
///
/// # Boundary behaviour
///
/// Voxel coordinates are clamped to the valid index range `[0, shape[i])` before
/// lookup, so any negative or over-limit coordinate returns `0.0`.
///
/// # Formula
///
/// ```text
/// d_clamped = clamp(d, 0, shape[0] − 1)
/// r_clamped = clamp(r, 0, shape[1] − 1)
/// c_clamped = clamp(c, 0, shape[2] − 1)
/// idx = d_clamped × (shape[1] × shape[2]) + r_clamped × shape[2] + c_clamped
/// intensity = pixels[idx]  if idx < pixels.len(), else 0.0
/// ```
pub fn intensity_at_voxel(vol: &LoadedVolume, voxel: [usize; 3]) -> f32 {
    let [d, r, c] = voxel;
    let [depth, height, width] = vol.shape;

    // Out-of-bounds: return 0.0
    if d >= depth || r >= height || c >= width {
        return 0.0;
    }

    // Linear index into row-major pixel buffer: idx = d × (h × w) + r × w + c
    let idx = d * (height * width) + r * width + c;

    // Safe access with bounds check
    if idx < vol.data.len() {
        vol.data[idx]
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "tests_pointer_intensity.rs"]
mod tests;
