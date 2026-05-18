//! Zero-crossing detection filter.
//!
//! # Mathematical Specification
//!
//! A zero crossing exists at voxel x if:
//!
//! 1. `I(x) == 0.0`, **or**
//! 2. There exists a 6-connected neighbour y such that `I(x) * I(y) < 0`
//!    (i.e. x and y have strictly opposite signs).
//!
//! This matches the ITK `ZeroCrossingImageFilter` definition:
//! a voxel is a zero crossing if it equals exactly 0 or if at least one
//! 6-connected neighbour has an opposite sign.
//!
//! ## Invariants
//!
//! - `foreground_value` is emitted at zero-crossing voxels.
//! - `background_value` is emitted at all other voxels.
//! - Boundary voxels (edges of the volume) can only cross with in-bounds neighbours;
//!   out-of-bounds neighbours are ignored (do not trigger a crossing).
//! - Spatial metadata (shape, origin, spacing, direction) is preserved exactly.
//!
//! # ITK Parity
//!
//! `itk::ZeroCrossingImageFilter` with `SetForegroundValue` / `SetBackgroundValue`.
//! Typical use: detect zero crossings of a Laplacian-of-Gaussian edge image.

use crate::filter::ops::{extract_vec_infallible, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Detect zero crossings in a 3-D image.
///
/// Output voxels equal `foreground_value` where a sign change or exact zero
/// occurs in the 6-connected neighbourhood, and `background_value` elsewhere.
#[derive(Debug, Clone)]
pub struct ZeroCrossingImageFilter {
    /// Value assigned to zero-crossing voxels (default 1.0).
    pub foreground_value: f32,
    /// Value assigned to non-crossing voxels (default 0.0).
    pub background_value: f32,
}

impl Default for ZeroCrossingImageFilter {
    fn default() -> Self {
        Self {
            foreground_value: 1.0,
            background_value: 0.0,
        }
    }
}

impl ZeroCrossingImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_foreground(mut self, v: f32) -> Self {
        self.foreground_value = v;
        self
    }

    pub fn with_background(mut self, v: f32) -> Self {
        self.background_value = v;
        self
    }

    /// Apply to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let vals = &vals_vec;
        let [nz, ny, nx] = dims;

        let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;

        let fg = self.foreground_value;
        let bg = self.background_value;

        let mut out = vec![bg; nz * ny * nx];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = vals[idx(iz, iy, ix)];
                    // Case 1: exact zero
                    if v == 0.0 {
                        out[idx(iz, iy, ix)] = fg;
                        continue;
                    }
                    // Case 2: opposite-sign 6-connected neighbour
                    let neighbours: &[(isize, isize, isize)] = &[
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ];
                    let crosses = neighbours.iter().any(|&(dz, dy, dx)| {
                        let nz_ = iz as isize + dz;
                        let ny_ = iy as isize + dy;
                        let nx_ = ix as isize + dx;
                        if nz_ < 0
                            || ny_ < 0
                            || nx_ < 0
                            || nz_ >= nz as isize
                            || ny_ >= ny as isize
                            || nx_ >= nx as isize
                        {
                            return false;
                        }
                        let nv = vals[idx(nz_ as usize, ny_ as usize, nx_ as usize)];
                        v * nv < 0.0
                    });
                    if crosses {
                        out[idx(iz, iy, ix)] = fg;
                    }
                }
            }
        }

        Ok(rebuild(out, dims, image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        img.data_vec()
    }

    #[test]
    fn zero_crossing_detects_sign_change_across_boundary() {
        // 1×1×3: [-1, +1, +1] — crossing between ix=0 and ix=1
        let img = make_image(vec![-1.0, 1.0, 1.0], [1, 1, 3]);
        let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
        let v = voxels(&out);
        // ix=0: neighbour ix=1 is +1, -1 * +1 < 0 → crossing
        assert_eq!(v[0], 1.0, "ix=0 should be zero-crossing");
        // ix=1: neighbour ix=0 is -1, +1 * -1 < 0 → crossing
        assert_eq!(v[1], 1.0, "ix=1 should be zero-crossing");
        // ix=2: only neighbour ix=1 is +1, +1 * +1 > 0 → not crossing
        assert_eq!(v[2], 0.0, "ix=2 should be background");
    }

    #[test]
    fn zero_crossing_exact_zero_is_foreground() {
        // 1×1×3: [1, 0, 1] — middle voxel is exactly 0 → crossing
        let img = make_image(vec![1.0, 0.0, 1.0], [1, 1, 3]);
        let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
        let v = voxels(&out);
        assert_eq!(v[1], 1.0, "exact-zero voxel must be foreground");
        assert_eq!(
            v[0], 0.0,
            "positive voxel with only positive neighbours is background"
        );
        assert_eq!(
            v[2], 0.0,
            "positive voxel with only positive neighbours is background"
        );
    }

    #[test]
    fn zero_crossing_uniform_positive_no_crossings() {
        let img = make_image(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2, 2, 2]);
        let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
        let v = voxels(&out);
        for (i, &x) in v.iter().enumerate() {
            assert_eq!(x, 0.0, "voxel {} should be background, got {}", i, x);
        }
    }

    #[test]
    fn zero_crossing_custom_foreground_background_values() {
        // 1×1×2: [-1, +1] — both voxels are crossings
        let img = make_image(vec![-1.0, 1.0], [1, 1, 2]);
        let out = ZeroCrossingImageFilter::new()
            .with_foreground(255.0)
            .with_background(-1.0)
            .apply(&img)
            .unwrap();
        let v = voxels(&out);
        assert_eq!(v[0], 255.0);
        assert_eq!(v[1], 255.0);
    }

    #[test]
    fn zero_crossing_preserves_spatial_metadata() {
        let img = make_image(vec![-1.0, 1.0, -1.0, 1.0], [1, 2, 2]);
        let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
        assert_eq!(out.shape(), img.shape());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
    }

    #[test]
    fn zero_crossing_boundary_voxel_no_oob_crossing() {
        // 1×1×2: [1, 2] — both positive, no sign change
        // Boundary voxels only consider in-bounds neighbours
        let img = make_image(vec![1.0, 2.0], [1, 1, 2]);
        let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
        let v = voxels(&out);
        assert_eq!(
            v[0], 0.0,
            "boundary voxel with no sign-change neighbour is background"
        );
        assert_eq!(
            v[1], 0.0,
            "boundary voxel with no sign-change neighbour is background"
        );
    }
}
