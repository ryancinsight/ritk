//! Permute-axes (transpose) filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Let `order = [a, b, c]` where `{a, b, c}` is a permutation of `{0, 1, 2}`.
//! Axis 0 = Z (slowest), axis 1 = Y, axis 2 = X (fastest) in RITK ZYX order.
//!
//! Output shape: `[in_shape[a], in_shape[b], in_shape[c]]`
//!
//! For output voxel `(i0, i1, i2)`:
//!
//! Let in-index `in_idx` be constructed by: `in_idx[order[j]] = ij` for j=0,1,2,
//! then `out[i0][i1][i2] = in[in_idx[0]][in_idx[1]][in_idx[2]]`.
//!
//! ## Direction Update
//!
//! The direction matrix columns are permuted to maintain physical consistency:
//! `new_dir.col(j) = old_dir.col(order[j])`
//!
//! The spacing is likewise permuted:
//! `new_spacing[j] = old_spacing[order[j]]`
//!
//! ## Invariants
//!
//! - `order` must be a permutation of `{0, 1, 2}`; validated at runtime.
//! - Identity permutation `[0, 1, 2]` is a no-op.
//! - Origin is unchanged (origin = physical position of voxel \[0,0,0\], unchanged).
//!
//! # ITK Parity
//!
//! `itk::PermuteAxesImageFilter` with `SetOrder({a, b, c})`.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_spatial::{Direction, Spacing};
use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_metadata};

/// Rearrange the axes of a 3-D image according to a permutation.
///
/// `order[i]` specifies which input axis feeds output axis `i`.
#[derive(Debug, Clone)]
pub struct PermuteAxesImageFilter {
    /// Permutation of `{0, 1, 2}`.  `order[i]` = input axis for output axis i.
    pub order: [usize; 3],
}

impl PermuteAxesImageFilter {
    pub fn new(order: [usize; 3]) -> Self {
        Self { order }
    }

    /// Apply the permutation to a 3-D image.
    ///
    /// Returns `Err` if `order` is not a valid permutation of `{0, 1, 2}`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let order = self.order;

        // Validate: order must be a permutation of {0, 1, 2}
        let mut seen = [false; 3];
        for &ax in &order {
            if ax > 2 {
                anyhow::bail!(
                    "PermuteAxesImageFilter: axis index {} out of range [0,2]",
                    ax
                );
            }
            if seen[ax] {
                anyhow::bail!("PermuteAxesImageFilter: duplicate axis {} in order", ax);
            }
            seen[ax] = true;
        }

        let (vals_vec, in_shape) = extract_vec_infallible(image);
        let vals = &vals_vec;
        let out_shape = [in_shape[order[0]], in_shape[order[1]], in_shape[order[2]]];

        let [_inz, iny, inx] = in_shape;
        let [oz, oy, ox] = out_shape;

        let in_idx = |iz: usize, iy: usize, ix: usize| iz * iny * inx + iy * inx + ix;

        let mut out = vec![0.0f32; oz * oy * ox];

        for i0 in 0..oz {
            for i1 in 0..oy {
                for i2 in 0..ox {
                    // Build in-index: in_idx[order[j]] = ij
                    let mut in_coords = [0usize; 3];
                    in_coords[order[0]] = i0;
                    in_coords[order[1]] = i1;
                    in_coords[order[2]] = i2;
                    let src = in_idx(in_coords[0], in_coords[1], in_coords[2]);
                    let dst = i0 * oy * ox + i1 * ox + i2;
                    out[dst] = vals[src];
                }
            }
        }

        // Permute spacing: new_spacing[j] = old_spacing[order[j]]
        let old_spacing = image.spacing();
        let new_spacing = Spacing::new([
            old_spacing[order[0]],
            old_spacing[order[1]],
            old_spacing[order[2]],
        ]);

        // Permute direction columns: new_dir.col(j) = old_dir.col(order[j])
        let old_dir = image.direction();
        let mut new_dir = Direction::zeros();
        for j in 0..3 {
            for row in 0..3 {
                new_dir[(row, j)] = old_dir[(row, order[j])];
            }
        }

        Ok(rebuild_with_metadata(
            out,
            out_shape,
            *image.origin(),
            new_spacing,
            new_dir,
            image,
        ))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_permute_axes.rs"]
mod tests_permute_axes;
