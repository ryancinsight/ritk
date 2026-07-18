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

use ritk_image::tensor::Backend;
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
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals_vec, in_shape) = extract_vec_infallible(image);
        let (out, out_shape, new_spacing, new_dir) =
            self.permute(&vals_vec, in_shape, image.spacing(), image.direction())?;

        Ok(rebuild_with_metadata(
            out,
            out_shape,
            *image.origin(),
            new_spacing,
            new_dir,
            image,
        ))
    }

    /// Apply the axis permutation to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (values, shape, spacing, direction) = self.permute(
            image.data_slice()?,
            image.shape(),
            image.spacing(),
            image.direction(),
        )?;
        ritk_image::native::Image::from_flat_on(
            values,
            shape,
            *image.origin(),
            spacing,
            direction,
            backend,
        )
    }

    fn permute(
        &self,
        values: &[f32],
        in_shape: [usize; 3],
        old_spacing: &Spacing<3>,
        old_direction: &Direction<3>,
    ) -> anyhow::Result<(Vec<f32>, [usize; 3], Spacing<3>, Direction<3>)> {
        self.validate_order()?;
        let [iny, inx] = [in_shape[1], in_shape[2]];
        let out_shape = [
            in_shape[self.order[0]],
            in_shape[self.order[1]],
            in_shape[self.order[2]],
        ];
        let [oz, oy, ox] = out_shape;
        let mut output = vec![0.0; oz * oy * ox];
        for i0 in 0..oz {
            for i1 in 0..oy {
                for i2 in 0..ox {
                    let mut input = [0; 3];
                    input[self.order[0]] = i0;
                    input[self.order[1]] = i1;
                    input[self.order[2]] = i2;
                    output[i0 * oy * ox + i1 * ox + i2] =
                        values[input[0] * iny * inx + input[1] * inx + input[2]];
                }
            }
        }
        let spacing = Spacing::new(std::array::from_fn(|axis| old_spacing[self.order[axis]]));
        let mut direction = Direction::zeros();
        for column in 0..3 {
            for row in 0..3 {
                direction[(row, column)] = old_direction[(row, self.order[column])];
            }
        }
        Ok((output, out_shape, spacing, direction))
    }

    fn validate_order(&self) -> anyhow::Result<()> {
        let mut seen = [false; 3];
        for axis in self.order {
            if axis > 2 {
                anyhow::bail!("PermuteAxesImageFilter: axis index {axis} out of range [0,2]");
            }
            if seen[axis] {
                anyhow::bail!("PermuteAxesImageFilter: duplicate axis {axis} in order");
            }
            seen[axis] = true;
        }
        Ok(())
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_permute_axes.rs"]
mod tests_permute_axes;
