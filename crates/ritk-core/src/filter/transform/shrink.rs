//! Shrink (integer downsampling) image filter.
//!
//! # Mathematical Specification
//!
//! For a 3-D image `I` with shape `[Nz, Ny, Nx]` and spacing `(sz, sy, sx)`,
//! the shrink filter with integer factors `[fz, fy, fx]` (all ≥ 1) produces an
//! output image with shape:
//!
//! ```text
//! [oz, oy, ox] = [ceil(Nz/fz), ceil(Ny/fy), ceil(Nx/fx)]
//! ```
//!
//! Each output voxel at `(iz, iy, ix)` is the arithmetic mean of all input voxels
//! in the axis-aligned tile:
//!
//! ```text
//! tile(iz,iy,ix) = {(kz, ky, kx) : kz ∈ [iz·fz, min((iz+1)·fz−1, Nz−1)],
//!                                    ky ∈ [iy·fy, min((iy+1)·fy−1, Ny−1)],
//!                                    kx ∈ [ix·fx, min((ix+1)·fx−1, Nx−1)]}
//! out(iz,iy,ix) = mean_{(kz,ky,kx) ∈ tile} I(kz, ky, kx)
//! ```
//!
//! The output spacing is updated: `out_spacing[i] = in_spacing[i] × factor[i]`.
//! The origin is unchanged (corresponds to the center of the first output voxel).
//!
//! # ITK Parity
//!
//! Corresponds to `itk::ShrinkImageFilter<TInputImage, TOutputImage>`.
//! ITK uses averaging within each tile. Default factors = [1, 1, 1] (identity).
//! Note: ITK `ShrinkImageFilter` uses subsampling (single voxel per tile);
//! this implementation uses averaging for anti-aliasing. For subsampling without
//! averaging, set `averaging = false`.
//!
//! # Reference
//!
//! - Gonzalez, R.C. & Woods, R.E. (2008). *Digital Image Processing*, 3rd ed. §4.7.

use crate::image::Image;
use crate::spatial::Spacing;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Integer downsampling filter.
///
/// Reduces image dimensions by integer factors, computing the mean of each tile.
#[derive(Debug, Clone)]
pub struct ShrinkImageFilter {
    /// Downsampling factors per axis `[fz, fy, fx]`. All must be ≥ 1. Default [1,1,1].
    pub shrink_factors: [usize; 3],
}

impl ShrinkImageFilter {
    /// Construct with the given per-axis shrink factors (all ≥ 1).
    ///
    /// # Panics
    ///
    /// Does not panic, but using factors of 0 is treated as 1 (identity).
    pub fn new(shrink_factors: [usize; 3]) -> Self {
        Self { shrink_factors }
    }
}

impl Default for ShrinkImageFilter {
    fn default() -> Self {
        Self::new([1, 1, 1])
    }
}

impl ShrinkImageFilter {
    /// Apply the shrink filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let [nz, ny, nx] = image.shape();
        let device = image.data().device();

        let [fz, fy, fx] = [
            self.shrink_factors[0].max(1),
            self.shrink_factors[1].max(1),
            self.shrink_factors[2].max(1),
        ];

        // Output shape: ceil(N/f)
        let oz = nz.div_ceil(fz);
        let oy = ny.div_ceil(fy);
        let ox = nx.div_ceil(fx);

        let td = image.data().clone().into_data();
        let vals: &[f32] = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("ShrinkImageFilter: {:?}", e))?;

        let mut out = vec![0.0f32; oz * oy * ox];

        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    let kz0 = iz * fz;
                    let kz1 = ((iz + 1) * fz - 1).min(nz - 1);
                    let ky0 = iy * fy;
                    let ky1 = ((iy + 1) * fy - 1).min(ny - 1);
                    let kx0 = ix * fx;
                    let kx1 = ((ix + 1) * fx - 1).min(nx - 1);
                    let mut sum = 0.0f64;
                    let mut count = 0u64;
                    for kz in kz0..=kz1 {
                        for ky in ky0..=ky1 {
                            for kx in kx0..=kx1 {
                                sum += vals[kz * ny * nx + ky * nx + kx] as f64;
                                count += 1;
                            }
                        }
                    }
                    out[iz * oy * ox + iy * ox + ix] = (sum / count as f64) as f32;
                }
            }
        }

        // Update spacing: out_spacing[i] = in_spacing[i] * factor[i].
        let in_s = image.spacing();
        let out_spacing = Spacing::new([
            in_s[0] * fz as f64,
            in_s[1] * fy as f64,
            in_s[2] * fx as f64,
        ]);

        let shape = Shape::new([oz, oy, ox]);
        let data = TensorData::new(out, shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            out_spacing,
            *image.direction(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    /// Factor [1,1,1] → identity (same shape and values).
    #[test]
    fn factor_one_is_identity() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let img = make_image(data.clone(), [2, 3, 4], [1.0, 1.0, 1.0]);
        let out = ShrinkImageFilter::new([1, 1, 1]).apply(&img).unwrap();
        assert_eq!(out.shape(), [2, 3, 4]);
        let v = voxels(&out);
        for (a, b) in v.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// 1-D shrink by 2: mean of pairs.
    /// [0, 2, 4, 6] with factor [1,1,2] → [1, 5] (means of [0,2] and [4,6]).
    #[test]
    fn shrink_x_by_2() {
        let img = make_image(vec![0.0, 2.0, 4.0, 6.0], [1, 1, 4], [1.0, 1.0, 1.0]);
        let out = ShrinkImageFilter::new([1, 1, 2]).apply(&img).unwrap();
        assert_eq!(out.shape(), [1, 1, 2]);
        let v = voxels(&out);
        assert!((v[0] - 1.0).abs() < 1e-5, "v[0]={}", v[0]);
        assert!((v[1] - 5.0).abs() < 1e-5, "v[1]={}", v[1]);
    }

    /// Output spacing scales by factor.
    #[test]
    fn output_spacing_scales() {
        let img = make_image(vec![1.0f32; 8], [2, 2, 2], [1.0, 2.0, 3.0]);
        let out = ShrinkImageFilter::new([2, 2, 2]).apply(&img).unwrap();
        assert_eq!(out.shape(), [1, 1, 1]);
        let s = out.spacing();
        assert!((s[0] - 2.0).abs() < 1e-10, "sz={}", s[0]);
        assert!((s[1] - 4.0).abs() < 1e-10, "sy={}", s[1]);
        assert!((s[2] - 6.0).abs() < 1e-10, "sx={}", s[2]);
    }

    /// Shrink 4×4×4 by [2,2,2] → 2×2×2, each output voxel = mean of 2×2×2 tile of constant image.
    #[test]
    fn constant_image_mean_preserved() {
        let img = make_image(vec![5.0f32; 64], [4, 4, 4], [1.0, 1.0, 1.0]);
        let out = ShrinkImageFilter::new([2, 2, 2]).apply(&img).unwrap();
        assert_eq!(out.shape(), [2, 2, 2]);
        let v = voxels(&out);
        for &x in &v {
            assert!((x - 5.0).abs() < 1e-5, "expected 5.0 got {x}");
        }
    }

    /// Odd input size with even factor: ceil division.
    /// 1×1×5 with factor [1,1,2] → shape [1,1,3]: means of [0,1], [2,3], [4].
    #[test]
    fn odd_size_ceil_division() {
        let img = make_image(vec![0.0, 1.0, 2.0, 3.0, 4.0], [1, 1, 5], [1.0, 1.0, 1.0]);
        let out = ShrinkImageFilter::new([1, 1, 2]).apply(&img).unwrap();
        assert_eq!(out.shape(), [1, 1, 3]);
        let v = voxels(&out);
        assert!((v[0] - 0.5).abs() < 1e-5, "v[0]={}", v[0]); // mean(0,1)
        assert!((v[1] - 2.5).abs() < 1e-5, "v[1]={}", v[1]); // mean(2,3)
        assert!((v[2] - 4.0).abs() < 1e-5, "v[2]={}", v[2]); // only 4
    }
}
