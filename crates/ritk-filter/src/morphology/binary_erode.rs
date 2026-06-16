//! Binary erosion filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Binary erosion with a flat cubic structuring element B of half-width r:
//!
//!   (E_B f)(x) = fg  iff  ∀ b ∈ B: f(x + b) = fg
//!             = bg  otherwise
//!
//! where B = { b ∈ ℤ³ : |b_i| ≤ r  for i ∈ {0, 1, 2} }.
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
//! - Flat ball structuring element of radius r.
//!
//! # Complexity
//!
//! O(N · (2r + 1)³) where N is the total voxel count.
//!
//! # References
//!
//! - Haralick, R.M., Sternberg, S.R., & Zhuang, X. (1987). Image analysis
//!   using mathematical morphology. *IEEE TPAMI*, 9(4), 532–550.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use super::types::ForegroundValue;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Binary erosion filter for 3-D images.
///
/// Shrinks foreground regions by eroding their boundaries.  Each voxel is
/// foreground in the output iff every voxel in its `(2r+1)³` cubic
/// neighbourhood is foreground in the input.
///
/// Out-of-bounds neighbours are treated as background, so foreground regions
/// touching the image border are eroded to background (ITK default behaviour).
#[derive(Debug, Clone)]
pub struct BinaryErodeFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
    /// Voxel value treated as foreground. Default: 1.0.
    foreground_value: ForegroundValue,
}

impl BinaryErodeFilter {
    /// Create a binary erosion filter with `radius` and default `foreground_value = 1.0`.
    pub fn new(radius: usize) -> Self {
        Self {
            radius,
            foreground_value: ForegroundValue::ONE,
        }
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
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let result = erode_binary_3d(&vals, dims, self.radius, self.foreground_value);

        let device = image.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(
            t,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
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
/// - Output[i] ∈ {foreground_value, 0.0}.
/// - Output[i] = foreground_value iff all (2r+1)³ neighbours (clamped-background) = fg.
pub(crate) fn erode_binary_3d(
    data: &[f32],
    dims: [usize; 3],
    radius: usize,
    fg: ForegroundValue,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let fg: f32 = fg.into();
    let n = nz * ny * nx;
    let mut output = vec![0.0_f32; n];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut all_fg = true;
                'outer: for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let zz = iz as isize + dz;
                            let yy = iy as isize + dy;
                            let xx = ix as isize + dx;
                            // Out-of-bounds → background
                            if zz < 0
                                || yy < 0
                                || xx < 0
                                || zz >= nz as isize
                                || yy >= ny as isize
                                || xx >= nx as isize
                            {
                                all_fg = false;
                                break 'outer;
                            }
                            let idx = zz as usize * ny * nx + yy as usize * nx + xx as usize;
                            if data[idx] != fg {
                                all_fg = false;
                                break 'outer;
                            }
                        }
                    }
                }
                if all_fg {
                    output[iz * ny * nx + iy * nx + ix] = fg;
                }
            }
        }
    }
    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_binary_erode.rs"]
mod tests_binary_erode;
