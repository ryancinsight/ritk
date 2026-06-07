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

use crate::filter::ops::{extract_vec_infallible, rebuild_with_metadata};
use crate::image::Image;
use crate::spatial::{Direction, Spacing};
use burn::tensor::backend::Backend;

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
        let mut new_dir_mat = nalgebra::Matrix3::zeros();
        for j in 0..3 {
            for row in 0..3 {
                new_dir_mat[(row, j)] = old_dir.0[(row, order[j])];
            }
        }
        let new_dir = Direction(new_dir_mat);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::ops::extract_vec_infallible;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        use burn::tensor::{Shape, Tensor, TensorData};
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 2.0, 3.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        let (v, _) = extract_vec_infallible(img);
        v
    }

    #[test]
    fn permute_axes_identity_order_is_noop() {
        let vals: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let img = make_image(vals.clone(), [1, 2, 3]);
        let out = PermuteAxesImageFilter::new([0, 1, 2]).apply(&img).unwrap();
        assert_eq!(out.shape(), [1, 2, 3]);
        let v = voxels(&out);
        for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
            assert_eq!(a, b, "voxel {}: identity mismatch", i);
        }
    }

    #[test]
    fn permute_axes_transpose_zx_swaps_shape_and_voxels() {
        // Input shape [2, 1, 3], order [2, 1, 0] → output shape [3, 1, 2]
        // vals layout (ZYX): val at (iz,iy,ix) = iz*3 + iy*3 + ix + 1
        let vals: Vec<f32> = (1..=6).map(|x| x as f32).collect(); // 2×1×3
        let img = make_image(vals, [2, 1, 3]);
        let out = PermuteAxesImageFilter::new([2, 1, 0]).apply(&img).unwrap();
        assert_eq!(out.shape(), [3, 1, 2], "output shape after ZX transpose");
        let v = voxels(&out);
        // out[i0][i1][i2] = in[in_idx]; in_idx[order[j]] = ij
        // order=[2,1,0]: in_coords[2]=i0, in_coords[1]=i1, in_coords[0]=i2
        // out[0][0][0] = in[0][0][0] = 1
        // out[0][0][1] = in[1][0][0] = 4
        // out[1][0][0] = in[0][0][1] = 2
        assert_eq!(v[0], 1.0, "out[0][0][0]");
        assert_eq!(v[1], 4.0, "out[0][0][1]");
        assert_eq!(v[2], 2.0, "out[1][0][0]");
    }

    #[test]
    fn permute_axes_spacing_is_permuted() {
        // spacing = [1, 2, 3]; order = [2, 0, 1] → new spacing = [3, 1, 2]
        let img = make_image(vec![0.0; 6], [1, 2, 3]);
        let out = PermuteAxesImageFilter::new([2, 0, 1]).apply(&img).unwrap();
        let s = out.spacing();
        assert!(
            (s[0] - 3.0).abs() < 1e-9,
            "spacing[0] expected 3, got {}",
            s[0]
        );
        assert!(
            (s[1] - 1.0).abs() < 1e-9,
            "spacing[1] expected 1, got {}",
            s[1]
        );
        assert!(
            (s[2] - 2.0).abs() < 1e-9,
            "spacing[2] expected 2, got {}",
            s[2]
        );
    }

    #[test]
    fn permute_axes_invalid_order_returns_error() {
        let img = make_image(vec![1.0; 8], [2, 2, 2]);
        // Duplicate axis
        let r = PermuteAxesImageFilter::new([0, 0, 1]).apply(&img);
        assert!(r.is_err(), "duplicate axis should return Err");
        // Out-of-range axis
        let r2 = PermuteAxesImageFilter::new([0, 1, 3]).apply(&img);
        assert!(r2.is_err(), "out-of-range axis should return Err");
    }
}
