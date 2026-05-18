//! Paste filter — copy a source image into a region of a destination image.
//!
//! # Mathematical Specification
//!
//! Given destination image `D` with shape `[Dz, Dy, Dx]` and source image `S`
//! with shape `[Sz, Sy, Sx]`, and a destination start index `dest_start = [dz, dy, dx]`:
//!
//! `out = copy(D)`
//! `out[dz + iz][dy + iy][dx + ix] = S[iz][iy][ix]`
//!
//! for `iz ∈ [0, Sz)`, `iy ∈ [0, Sy)`, `ix ∈ [0, Sx)`.
//!
//! ## Invariants
//!
//! - `dest_start[k] + S_shape[k] ≤ D_shape[k]` for k ∈ {z, y, x}; validated.
//! - Destination voxels outside the paste region are unchanged.
//! - Spatial metadata of the *destination* image is preserved in the output.
//! - Source and destination must have the same f32 voxel type.
//!
//! # ITK Parity
//!
//! `itk::PasteImageFilter` with `SetDestinationIndex(idx)` and
//! `SetSourceRegion(region)` spanning the full source.

use crate::filter::ops::{extract_vec_infallible, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Paste a source image into a destination image at a given index.
///
/// The output adopts the spatial metadata (origin, spacing, direction) of the
/// *destination* image.
#[derive(Debug, Clone)]
pub struct PasteImageFilter {
    /// Starting voxel index in the destination: `[z, y, x]`.
    pub dest_start: [usize; 3],
}

impl PasteImageFilter {
    pub fn new(dest_start: [usize; 3]) -> Self {
        Self { dest_start }
    }

    /// Apply the paste: returns a copy of `dest` with `source` written at
    /// `dest_start`.
    ///
    /// Returns `Err` if the source region would exceed the destination bounds.
    pub fn apply<B: Backend>(
        &self,
        dest: &Image<B, 3>,
        source: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let [dz, dy, dx] = dest.shape();
        let [sz, sy, sx] = source.shape();
        let [sdz, sdy, sdx] = self.dest_start;

        if sdz + sz > dz {
            anyhow::bail!(
                "PasteImageFilter: source Z extent [{}..{}) exceeds dest depth {}",
                sdz,
                sdz + sz,
                dz
            );
        }
        if sdy + sy > dy {
            anyhow::bail!(
                "PasteImageFilter: source Y extent [{}..{}) exceeds dest height {}",
                sdy,
                sdy + sy,
                dy
            );
        }
        if sdx + sx > dx {
            anyhow::bail!(
                "PasteImageFilter: source X extent [{}..{}) exceeds dest width {}",
                sdx,
                sdx + sx,
                dx
            );
        }

        let (dest_vec, dims) = extract_vec_infallible(dest);
        let mut out = dest_vec;

        let (src_vals_vec, _) = extract_vec_infallible(source);
        let src_vals = &src_vals_vec;

        for iz in 0..sz {
            for iy in 0..sy {
                for ix in 0..sx {
                    let src_idx = iz * sy * sx + iy * sx + ix;
                    let dst_idx = (sdz + iz) * dy * dx + (sdy + iy) * dx + (sdx + ix);
                    out[dst_idx] = src_vals[src_idx];
                }
            }
        }

        Ok(rebuild(out, dims, dest))
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
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        let (v, _) = extract_vec_infallible(img);
        v
    }

    #[test]
    fn paste_writes_source_into_destination() {
        // dest: 3×3×3 all zeros; source: 1×1×1 = [99]; paste at [1,1,1]
        let dest = make_image(vec![0.0f32; 27], [3, 3, 3]);
        let src = make_image(vec![99.0f32], [1, 1, 1]);
        let out = PasteImageFilter::new([1, 1, 1]).apply(&dest, &src).unwrap();
        let v = voxels(&out);
        let pasted_idx = 1 * 9 + 1 * 3 + 1;
        assert_eq!(v[pasted_idx], 99.0, "pasted voxel value");
        // All other voxels remain 0
        for (i, &x) in v.iter().enumerate() {
            if i != pasted_idx {
                assert_eq!(x, 0.0, "voxel {} should be 0, got {}", i, x);
            }
        }
    }

    #[test]
    fn paste_at_origin_replaces_corner() {
        let dest = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let src = make_image(vec![5.0, 6.0, 7.0, 8.0], [1, 2, 2]);
        let out = PasteImageFilter::new([0, 0, 0]).apply(&dest, &src).unwrap();
        let v = voxels(&out);
        assert_eq!(v[0], 5.0);
        assert_eq!(v[1], 6.0);
        assert_eq!(v[2], 7.0);
        assert_eq!(v[3], 8.0);
        // Second z-slice unchanged
        assert_eq!(v[4], 0.0);
        assert_eq!(v[5], 0.0);
        assert_eq!(v[6], 0.0);
        assert_eq!(v[7], 0.0);
    }

    #[test]
    fn paste_preserves_dest_spatial_metadata() {
        let dest = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let src = make_image(vec![1.0f32], [1, 1, 1]);
        let out = PasteImageFilter::new([0, 0, 0]).apply(&dest, &src).unwrap();
        assert_eq!(out.shape(), dest.shape());
        assert_eq!(out.origin(), dest.origin());
        assert_eq!(out.spacing(), dest.spacing());
    }

    #[test]
    fn paste_out_of_bounds_returns_error() {
        let dest = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let src = make_image(vec![1.0f32; 4], [1, 2, 2]);
        // dest_start=[1,0,0]: Z extent [1..2) OK, but source is [1,2,2] → Z [1..2) OK
        // Increase to [1,1,0] → Y extent [1..3) exceeds height 2 → error
        let r = PasteImageFilter::new([1, 1, 0]).apply(&dest, &src);
        assert!(r.is_err(), "out-of-bounds paste must return Err");
    }

    #[test]
    fn paste_full_source_into_full_dest_replaces_all() {
        let dest = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let src_vals: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let src = make_image(src_vals.clone(), [2, 2, 2]);
        let out = PasteImageFilter::new([0, 0, 0]).apply(&dest, &src).unwrap();
        let v = voxels(&out);
        for (i, (&a, &b)) in v.iter().zip(src_vals.iter()).enumerate() {
            assert_eq!(a, b, "voxel {}: expected {} got {}", i, b, a);
        }
    }
}
