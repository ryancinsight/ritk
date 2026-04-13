//! Binary morphological closing: dilation followed by erosion.
//!
//! # Algorithm
//! Closing fills small holes (background regions) in a binary mask.
//!
//! Given a structuring element S (hypercube of side 2r+1):
//!   closing(I) = erode(dilate(I, S), S)
//!
//! # Complexity
//! O(n · (2r+1)^D) where n = total voxels, r = radius, D = image dimension.

use super::MorphologicalOperation;
use crate::image::Image;
use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

/// Binary morphological closing filter.
///
/// Applies dilation followed by erosion using a hypercube structuring element
/// of half-width `radius`. Fills small holes (background voids) without
/// significantly changing the overall foreground extent.
pub struct BinaryClosing {
    /// Half-width of the structuring element in voxels. Default 1.
    pub radius: usize,
}

impl BinaryClosing {
    /// Create a `BinaryClosing` filter with the given structuring element radius.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
}

impl Default for BinaryClosing {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

impl<B: Backend, const D: usize> MorphologicalOperation<B, D> for BinaryClosing {
    /// Apply closing (dilate then erode) to `mask`.
    ///
    /// # Arguments
    /// * `mask` – Binary mask image (0.0 = background, 1.0 = foreground).
    ///
    /// # Returns
    /// A new `Image<B, D>` with holes filled, preserving spatial metadata.
    fn apply(&self, mask: &Image<B, D>) -> Image<B, D> {
        let dilated = apply_morphological_op(mask, self.radius, false);
        apply_morphological_op(&dilated, self.radius, true)
    }
}

// ── Shared implementation ─────────────────────────────────────────────────────

/// Apply a binary morphological operation (erosion or dilation) in D dimensions.
///
/// `is_erosion = true`  → erosion  (output = 1 iff ALL neighbours are 1)
/// `is_erosion = false` → dilation (output = 1 iff ANY neighbour is 1)
pub(super) fn apply_morphological_op<B: Backend, const D: usize>(
    mask: &Image<B, D>,
    radius: usize,
    is_erosion: bool,
) -> Image<B, D> {
    let shape: [usize; D] = mask.shape();
    let total: usize = shape.iter().product();

    // Compute row-major strides.
    let mut strides = [0usize; D];
    let mut s = 1usize;
    for i in (0..D).rev() {
        strides[i] = s;
        s = s.saturating_mul(shape[i]);
    }

    let input_data = mask.data().clone().into_data();
    let input_vals = input_data
        .as_slice::<f32>()
        .expect("mask tensor must hold f32 values");

    let mut output = vec![0.0f32; total];
    let r = radius as isize;

    for flat_idx in 0..total {
        // Decompose flat_idx into D-dimensional coordinates.
        let mut coords = [0isize; D];
        let mut rem = flat_idx;
        for i in 0..D {
            coords[i] = (rem / strides[i]) as isize;
            rem %= strides[i];
        }

        let result = scan_neighborhood::<D>(input_vals, &coords, &shape, &strides, r, is_erosion);
        output[flat_idx] = if result { 1.0 } else { 0.0 };
    }

    let device = mask.data().device();
    let tensor = Tensor::<B, D>::from_data(TensorData::new(output, Shape::new(shape)), &device);

    Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
}

/// Scan the hypercube neighbourhood of `center` and return the erosion/dilation
/// result.
///
/// Iterates all offsets in `[−r, r]^D` using a D-dimensional counter.
/// Out-of-bounds neighbours are skipped (treated as absent).
fn scan_neighborhood<const D: usize>(
    data: &[f32],
    center: &[isize; D],
    shape: &[usize; D],
    strides: &[usize; D],
    r: isize,
    is_erosion: bool,
) -> bool {
    // D-dimensional counter, initialised to (−r, −r, …, −r).
    let mut offsets = [-r; D];

    loop {
        // Compute neighbour flat index (skip if out of bounds).
        let mut in_bounds = true;
        let mut flat = 0usize;
        for i in 0..D {
            let c = center[i] + offsets[i];
            if c < 0 || c >= shape[i] as isize {
                in_bounds = false;
                break;
            }
            flat += c as usize * strides[i];
        }

        if in_bounds {
            let is_foreground = data[flat] >= 0.5;
            if is_erosion && !is_foreground {
                return false; // Found a background voxel → erosion output = 0.
            }
            if !is_erosion && is_foreground {
                return true; // Found a foreground voxel → dilation output = 1.
            }
        }

        // Increment the D-dimensional counter (odometer style).
        let mut carry = true;
        for i in (0..D).rev() {
            if carry {
                offsets[i] += 1;
                if offsets[i] > r {
                    offsets[i] = -r;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break; // All combinations exhausted.
        }
    }

    // Exhausted all neighbours:
    // Erosion → all foreground (would have returned false on first background).
    // Dilation → all background (would have returned true on first foreground).
    is_erosion
}
