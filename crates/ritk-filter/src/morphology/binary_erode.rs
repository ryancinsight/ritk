//! Binary erosion filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Binary erosion with a flat rectangular structuring element B whose
//! `[z, y, x]` half-widths are `r_i`:
//!
//!   (E_B f)(x) = fg  iff  ∀ b ∈ B: f(x + b) = fg
//!             = bg  otherwise
//!
//! where B = { b ∈ ℤ³ : |b_i| ≤ r_i for i ∈ {0, 1, 2} }.
//!
//! # Boundary Handling
//!
//! Out-of-bounds neighbours are treated as background (`bg`).  This causes
//! erosion to remove the foreground layer at the image border — consistent
//! with `itk::BinaryErodeImageFilter` when `BoundaryToForeground = false`
//! (the ITK default).
//!
//! # ITK Parity
//!
//! Matches `itk::BinaryErodeImageFilter` with:
//! - `SetForegroundValue(foreground_value)` (default 1.0)
//! - `SetBackgroundValue(0.0)`
//! - `SetBoundaryToForeground(false)` (default)
//! - Flat rectangular structuring element with the configured axis radii.
//!
//! # Complexity
//!
//! O(N · Π_i(2r_i + 1)) where N is the total voxel count.
//!
//! # References
//!
//! - Haralick, R.M., Sternberg, S.R., & Zhuang, X. (1987). Image analysis
//!   using mathematical morphology. *IEEE TPAMI*, 9(4), 532–550.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use super::types::ForegroundValue;
use moirai;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Binary erosion filter for 3-D images.
///
/// Shrinks foreground regions by the configured `[z, y, x]` voxel radii. Each
/// voxel is foreground in the output iff every voxel in its rectangular
/// neighbourhood is foreground in the input.
///
/// Out-of-bounds neighbours are treated as background, so foreground regions
/// touching the image border are eroded to background (ITK default behaviour).
#[derive(Debug, Clone)]
pub struct BinaryErodeFilter {
    /// Structuring-element half-width in `[z, y, x]` voxels.
    radii: [usize; 3],
    /// Voxel value treated as foreground. Default: 1.0.
    foreground_value: ForegroundValue,
}

impl BinaryErodeFilter {
    /// Create a binary erosion filter with `radius` and default `foreground_value = 1.0`.
    pub fn new(radius: usize) -> Self {
        Self {
            radii: [radius; 3],
            foreground_value: ForegroundValue::ONE,
        }
    }

    /// Set independent `[z, y, x]` voxel radii.
    ///
    /// This represents a rectangular physical neighbourhood when each radius
    /// is derived from one physical distance divided by that axis's spacing.
    #[must_use]
    pub fn with_axis_radii(mut self, radii: [usize; 3]) -> Self {
        self.radii = radii;
        self
    }

    /// Set the foreground value (ITK `SetForegroundValue`).
    pub fn with_foreground(mut self, v: impl Into<ForegroundValue>) -> Self {
        self.foreground_value = v.into();
        self
    }

    /// Apply binary erosion to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    /// Output voxels are `foreground_value` (foreground) or `0.0` (background).
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let result = erode_binary_3d_with_radii(&vals, dims, self.radii, self.foreground_value);

        Ok(rebuild(result, dims, image))
    }
    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;

        let result = erode_binary_3d_with_radii(&vals, dims, self.radii, self.foreground_value);

        crate::native_support::rebuild_image(result, dims, image, backend)
    }
}

impl Default for BinaryErodeFilter {
    fn default() -> Self {
        Self::new(1)
    }
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Binary erosion on a flat Z×Y×X volume.
///
/// # Invariants
///
/// - Output length = `nz × ny × nx`.
/// - `Output[i]` ∈ {foreground_value, 0.0}.
/// - `Output[i]` = foreground_value iff all (2r+1)³ neighbours (clamped-background) = fg.
pub(crate) fn erode_binary_3d(
    data: &[f32],
    dims: [usize; 3],
    radius: usize,
    fg: ForegroundValue,
) -> Vec<f32> {
    erode_binary_3d_with_radii(data, dims, [radius; 3], fg)
}

/// Binary erosion with independent `[z, y, x]` voxel radii.
pub(crate) fn erode_binary_3d_with_radii(
    data: &[f32],
    dims: [usize; 3],
    radii: [usize; 3],
    fg: ForegroundValue,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let fg: f32 = fg.into();
    let n = nz * ny * nx;

    moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |flat| {
        let iz = flat / (ny * nx);
        let iy = (flat / nx) % ny;
        let ix = flat % nx;
        let [radius_z, radius_y, radius_x] = radii.map(|radius| radius as isize);
        let all_fg = (-radius_z..=radius_z)
            .flat_map(|dz| {
                (-radius_y..=radius_y)
                    .flat_map(move |dy| (-radius_x..=radius_x).map(move |dx| (dz, dy, dx)))
            })
            .all(|(dz, dy, dx)| {
                let zz = iz as isize + dz;
                let yy = iy as isize + dy;
                let xx = ix as isize + dx;
                if zz < 0
                    || yy < 0
                    || xx < 0
                    || zz >= nz as isize
                    || yy >= ny as isize
                    || xx >= nx as isize
                {
                    return false; // OOB treated as background
                }
                data[zz as usize * ny * nx + yy as usize * nx + xx as usize] == fg
            });
        if all_fg {
            fg
        } else {
            0.0_f32
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_binary_erode.rs"]
mod tests_binary_erode;
