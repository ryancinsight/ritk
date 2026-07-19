//! Binary morphological closing: dilation followed by erosion.
//!
//! # Algorithm
//! Closing fills small holes (background regions) in a binary mask.
//!
//! Given a structuring element S (hypercube of side 2r+1):
//!   closing(I) = erode(dilate(I, S), S)
//!
//! # Complexity
//! O(n Â· (2r+1)^D) where n = total voxels, r = radius, D = image dimension.

use super::MorphologicalOperation;
use ritk_image::tensor::{Backend, Tensor};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

/// Discriminates erosion from dilation in the shared morphological scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MorphOp {
    Erosion,
    Dilation,
}

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
    /// * `mask` â€“ Binary mask image (0.0 = background, 1.0 = foreground).
    ///
    /// # Returns
    /// A new `Image<f32, B, D>` with holes filled, preserving spatial metadata.
    ///
    /// Uses ITK's default "safe border": the mask is padded with `radius`
    /// background voxels on every face before the dilateâ†’erode, then cropped
    /// back. Without this, the trailing erosion treats out-of-bounds neighbours
    /// as foreground, leaving spurious foreground within `radius` of the volume
    /// border (closing is *not* border-invariant otherwise). This reproduces
    /// SimpleITK's `BinaryMorphologicalClosing` (`SafeBorder = true`).
    fn apply(&self, mask: &Image<f32, B, D>) -> Image<f32, B, D> {
        if self.radius == 0 {
            return Image::new(
                mask.data().clone(),
                *mask.origin(),
                *mask.spacing(),
                *mask.direction(),
            )
            .expect("invariant: segmentation output tensor preserves the image rank");
        }
        let padded = pad_background(mask, self.radius);
        let dilated = apply_morphological_op(&padded, self.radius, MorphOp::Dilation);
        let eroded = apply_morphological_op(&dilated, self.radius, MorphOp::Erosion);
        crop_border(&eroded, self.radius, mask)
    }
}

/// Row-major strides for a shape.
fn strides_of<const D: usize>(shape: &[usize; D]) -> [usize; D] {
    let mut strides = [0usize; D];
    let mut s = 1usize;
    for i in (0..D).rev() {
        strides[i] = s;
        s = s.saturating_mul(shape[i]);
    }
    strides
}

/// Pad `mask` with `r` background (0.0) voxels on every face. Spatial metadata is
/// carried through unchanged â€” the padded image is a transient processing buffer
/// that `crop_border` reverses.
fn pad_background<B: Backend, const D: usize>(
    mask: &Image<f32, B, D>,
    r: usize,
) -> Image<f32, B, D> {
    let shape: [usize; D] = mask.shape();
    let (vals, _) = extract_vec_infallible(mask);
    let mut new_shape = shape;
    for v in new_shape.iter_mut() {
        *v += 2 * r;
    }
    let in_strides = strides_of(&shape);
    let out_strides = strides_of(&new_shape);
    let in_total: usize = shape.iter().product();
    let mut out = vec![0.0f32; new_shape.iter().product()];
    for (flat, &val) in vals.iter().enumerate().take(in_total) {
        let mut rem = flat;
        let mut out_flat = 0usize;
        for i in 0..D {
            let c = rem / in_strides[i];
            rem %= in_strides[i];
            out_flat += (c + r) * out_strides[i];
        }
        out[out_flat] = val;
    }
    let device = B::default();
    let tensor = Tensor::<f32, B>::from_slice_on(new_shape, &out, &device);
    Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
        .expect("invariant: segmentation output tensor preserves the image rank")
}

/// Crop `r` voxels from every face of `padded`, restoring `original`'s shape and
/// spatial metadata.
fn crop_border<B: Backend, const D: usize>(
    padded: &Image<f32, B, D>,
    r: usize,
    original: &Image<f32, B, D>,
) -> Image<f32, B, D> {
    let p_shape: [usize; D] = padded.shape();
    let o_shape: [usize; D] = original.shape();
    let (pvals, _) = extract_vec_infallible(padded);
    let p_strides = strides_of(&p_shape);
    let o_strides = strides_of(&o_shape);
    let o_total: usize = o_shape.iter().product();
    let mut out = vec![0.0f32; o_total];
    for (flat, slot) in out.iter_mut().enumerate().take(o_total) {
        let mut rem = flat;
        let mut p_flat = 0usize;
        for i in 0..D {
            let c = rem / o_strides[i];
            rem %= o_strides[i];
            p_flat += (c + r) * p_strides[i];
        }
        *slot = pvals[p_flat];
    }
    let device = B::default();
    let tensor = Tensor::<f32, B>::from_slice_on(o_shape, &out, &device);
    Image::new(
        tensor,
        *original.origin(),
        *original.spacing(),
        *original.direction(),
    )
    .expect("invariant: segmentation output tensor preserves the image rank")
}

// â”€â”€ Shared implementation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Apply a binary morphological operation (erosion or dilation) in D dimensions.
///
/// `op = MorphOp::Erosion`  â†’ erosion  (output = 1 iff ALL neighbours are 1)
/// `op = MorphOp::Dilation` â†’ dilation (output = 1 iff ANY neighbour is 1)
pub(super) fn apply_morphological_op<B: Backend, const D: usize>(
    mask: &Image<f32, B, D>,
    radius: usize,
    op: MorphOp,
) -> Image<f32, B, D> {
    let shape: [usize; D] = mask.shape();
    let total: usize = shape.iter().product();
    // Compute row-major strides.
    let mut strides = [0usize; D];
    let mut s = 1usize;
    for i in (0..D).rev() {
        strides[i] = s;
        s = s.saturating_mul(shape[i]);
    }
    let (input_vals, _shape) = extract_vec_infallible(mask);

    let mut output = vec![0.0f32; total];
    let r = radius as isize;

    for (flat_idx, out) in output.iter_mut().enumerate().take(total) {
        // Decompose flat_idx into D-dimensional coordinates.
        let mut coords = [0isize; D];
        let mut rem = flat_idx;
        for i in 0..D {
            coords[i] = (rem / strides[i]) as isize;
            rem %= strides[i];
        }

        let result = scan_neighborhood::<D>(&input_vals, &coords, &shape, &strides, r, op);
        *out = if result { 1.0 } else { 0.0 };
    }

    let device = B::default();
    let tensor = Tensor::<f32, B>::from_slice_on(shape, &output, &device);

    Image::new(tensor, *mask.origin(), *mask.spacing(), *mask.direction())
        .expect("invariant: segmentation output tensor preserves the image rank")
}

/// Scan the hypercube neighbourhood of `center` and return the erosion/dilation
/// result.
///
/// Iterates all offsets in `[âˆ’r, r]^D` using a D-dimensional counter.
/// Out-of-bounds neighbours are skipped (treated as absent).
fn scan_neighborhood<const D: usize>(
    data: &[f32],
    center: &[isize; D],
    shape: &[usize; D],
    strides: &[usize; D],
    r: isize,
    op: MorphOp,
) -> bool {
    // D-dimensional counter, initialised to (âˆ’r, âˆ’r, â€¦, âˆ’r).
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
            let is_foreground = data[flat] >= super::FOREGROUND_THRESHOLD;
            if op == MorphOp::Erosion && !is_foreground {
                return false; // Found a background voxel â†’ erosion output = 0.
            }
            if op == MorphOp::Dilation && is_foreground {
                return true; // Found a foreground voxel â†’ dilation output = 1.
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
    // Erosion â†’ all foreground (would have returned false on first background).
    // Dilation â†’ all background (would have returned true on first foreground).
    matches!(op, MorphOp::Erosion)
}
