//! Mean (box) smoothing filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! For a 3-D image `I` with voxel spacing `(sz, sy, sx)`, the mean filter at
//! voxel `(iz, iy, ix)` computes the arithmetic mean over the cubic
//! neighbourhood of half-width `radius`:
//!
//! ```text
//! M(iz, iy, ix) = (1 / |N|) · Σ_{(kz,ky,kx)∈N(iz,iy,ix)} I(kz, ky, kx)
//! ```
//!
//! where `N(p)` is the set of voxels within `radius` steps in each axis
//! direction, clamped to the image bounds (replicate padding).
//!
//! # Neighbourhood cardinality
//!
//! ```text
//! |N| = (min(iz, r)−max(iz−r, 0) + ... = (wz)(wy)(wx)
//! ```
//! where `wz = min(iz+r, Nz-1) − max(iz−r, 0) + 1`, etc. This accounts for
//! boundary voxels that have fewer neighbours.
//!
//! # ITK parity
//!
//! Corresponds to `itk::MeanImageFilter<InputImageType, OutputImageType>`.
//! ITK default radius = 1 (3×3×3 kernel). `radius = 0` is the identity.
//!
//! # Complexity
//!
//! O(N · (2r+1)³) — a separable integral-image approach would be O(N) per
//! radius, but (2r+1)³ ≤ 125 for default `r=1`, so the direct approach
//! matches expected workload. Parallelised over Z-slices with Rayon.
//!
//! # Reference
//!
//! - Gonzalez, R.C. & Woods, R.E. (2008). *Digital Image Processing*, 3rd ed.
//!   §3.5.1 Smoothing Linear Filters.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use rayon::prelude::*;
use std::sync::Arc;

/// Mean (box) smoothing filter.
///
/// Replaces each voxel with the arithmetic mean of its
/// `(2·radius+1)³` cubic neighbourhood.
/// `radius = 0` is the identity transform.
#[derive(Debug, Clone)]
pub struct MeanImageFilter {
    /// Half-width of the cubic neighbourhood in voxels. Default 1.
    pub radius: usize,
}

impl MeanImageFilter {
    /// Construct with the given neighbourhood radius (in voxels).
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
}

impl Default for MeanImageFilter {
    fn default() -> Self {
        Self::new(1)
    }
}

impl MeanImageFilter {
    /// Apply the mean filter to a 3-D image.
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let [nz, ny, nx] = image.shape();
        let device = image.data().device();

        let vals: Arc<Vec<f32>> = Arc::new(
            image
                .data()
                .clone()
                .into_data()
                .into_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("MeanImageFilter: {:?}", e))?,
        );

        let r = self.radius;

        // identity shortcut
        if r == 0 || nz == 0 || ny == 0 || nx == 0 {
            return Ok(Image::new(
                image.data().clone(),
                *image.origin(),
                *image.spacing(),
                *image.direction(),
            ));
        }

        let out: Vec<f32> = (0..nz)
            .into_par_iter()
            .flat_map(|iz| {
                let vals = Arc::clone(&vals);
                let z0 = iz.saturating_sub(r);
                let z1 = (iz + r).min(nz - 1);
                (0..ny)
                    .flat_map(move |iy| {
                        let vals = Arc::clone(&vals);
                        let y0 = iy.saturating_sub(r);
                        let y1 = (iy + r).min(ny - 1);
                        (0..nx).map(move |ix| {
                            let x0 = ix.saturating_sub(r);
                            let x1 = (ix + r).min(nx - 1);
                            let mut sum = 0.0f64;
                            let mut count = 0u64;
                            for kz in z0..=z1 {
                                for ky in y0..=y1 {
                                    for kx in x0..=x1 {
                                        sum += vals[kz * ny * nx + ky * nx + kx] as f64;
                                        count += 1;
                                    }
                                }
                            }
                            (sum / count as f64) as f32
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let shape = Shape::new([nz, ny, nx]);
        let data = TensorData::new(out, shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn::tensor::TensorData;
    use crate::spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    /// Constant image → mean = constant for any radius.
    #[test]
    fn constant_image_identity() {
        let img = make_image(vec![7.0f32; 27], [3, 3, 3]);
        let out = MeanImageFilter::new(1).apply(&img).unwrap();
        let v = voxels(&out);
        for &x in &v {
            assert!((x - 7.0).abs() < 1e-5, "expected 7.0 got {x}");
        }
    }

    /// radius=0 → exact identity.
    #[test]
    fn radius_zero_is_identity() {
        let data: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let img = make_image(data.clone(), [3, 3, 3]);
        let out = MeanImageFilter::new(0).apply(&img).unwrap();
        let v = voxels(&out);
        for (a, b) in v.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    /// Single-voxel image: radius > 0 → returns that voxel unchanged.
    #[test]
    fn single_voxel_returns_itself() {
        let img = make_image(vec![42.0], [1, 1, 1]);
        let out = MeanImageFilter::new(2).apply(&img).unwrap();
        assert!((voxels(&out)[0] - 42.0).abs() < 1e-5);
    }

    /// Step-edge smoothing: 3×3×3 volume, left half = 0, right half = 10.
    /// Center voxel mean over a 3×3×3 (no boundary cut) should be 5.0.
    #[test]
    fn step_edge_center_mean() {
        // 1×1×4 image: [0, 0, 10, 10]. With r=1, voxel at index 1:
        // neighbourhood = [0,0,10] → mean = 10/3 ≈ 3.333
        let img = make_image(vec![0.0, 0.0, 10.0, 10.0], [1, 1, 4]);
        let out = MeanImageFilter::new(1).apply(&img).unwrap();
        let v = voxels(&out);
        // voxel 1 (0-indexed): neighbourhood [0,0,10] → 10/3
        assert!((v[1] - 10.0 / 3.0).abs() < 1e-4, "v[1]={}", v[1]);
        // voxel 2: neighbourhood [0,10,10] → 20/3
        assert!((v[2] - 20.0 / 3.0).abs() < 1e-4, "v[2]={}", v[2]);
    }

    /// Spatial metadata is preserved.
    #[test]
    fn preserves_spatial_metadata() {
        use crate::spatial::Direction;
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vec![1.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let origin = Point::new([3.0_f64, 5.0, 7.0]);
        let spacing = Spacing::new([2.0_f64, 3.0, 4.0]);
        let dir = Direction::identity();
        let img = Image::new(tensor, origin, spacing, dir);
        let out = MeanImageFilter::new(1).apply(&img).unwrap();
        assert_eq!(*out.origin(), origin);
        assert_eq!(*out.spacing(), spacing);
    }

    /// Shape is unchanged after filtering.
    #[test]
    fn output_shape_matches_input() {
        let img = make_image(vec![1.0f32; 60], [3, 4, 5]);
        let out = MeanImageFilter::new(2).apply(&img).unwrap();
        assert_eq!(out.shape(), [3, 4, 5]);
    }
}
