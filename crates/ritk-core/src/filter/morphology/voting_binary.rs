//! Voting binary image filter — cellular-automata label voting.
//!
//! # Mathematical Specification
//!
//! A **voting binary filter** applies a single-step cellular automaton to a
//! binary image using a cubic neighbourhood of half-width `radius`:
//!
//! Let `N_fg(p)` = number of foreground neighbours of `p` in the neighbourhood.
//!
//! The output at `p` is:
//! ```text
//! I_out(p) = fg_value  if I(p) = bg_value  AND  N_fg(p) >= birth_threshold
//!          = bg_value  if I(p) = fg_value  AND  N_fg(p) <  survival_threshold
//!          = I(p)      otherwise
//! ```
//!
//! - `birth_threshold`: minimum foreground count for a background voxel to become foreground.
//! - `survival_threshold`: minimum foreground count for an existing foreground voxel to survive.
//!
//! # ITK Parity
//!
//! Corresponds to `itk::VotingBinaryImageFilter<TInputImage, TOutputImage>`.
//! ITK defaults: `Radius = 1`, `BirthThreshold = 1`, `SurvivalThreshold = 1`,
//! `ForegroundValue = 1`, `BackgroundValue = 0`.
//!
//! # Reference
//!
//! - Breu, H. et al. (1995). Linear time Euclidean distance algorithms.
//!   *IEEE Trans. PAMI* 17(5):529–533.

use super::types::ForegroundValue;
use crate::filter::ops::extract_vec;
use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Voting binary image filter.
///
/// Each voxel casts a "vote" by examining how many of its neighbours are foreground.
/// Background voxels may turn foreground (birth); foreground voxels may turn background
/// (death). A single iteration is applied.
#[derive(Debug, Clone)]
pub struct VotingBinaryImageFilter {
    /// Half-width of the cubic neighbourhood in voxels. Default 1.
    pub radius: usize,
    /// Minimum foreground neighbour count for a background voxel to become foreground.
    pub birth_threshold: usize,
    /// Minimum foreground neighbour count for a foreground voxel to survive.
    pub survival_threshold: usize,
    /// Foreground intensity value. Default 1.0.
    pub foreground_value: ForegroundValue,
    /// Background intensity value. Default 0.0.
    pub background_value: f32,
}

impl VotingBinaryImageFilter {
    /// Construct with explicit parameters.
    pub fn new(
        radius: usize,
        birth_threshold: usize,
        survival_threshold: usize,
        foreground_value: impl Into<ForegroundValue>,
        background_value: f32,
    ) -> Self {
        Self {
            radius,
            birth_threshold,
            survival_threshold,
            foreground_value: foreground_value.into(),
            background_value,
        }
    }
}

impl Default for VotingBinaryImageFilter {
    fn default() -> Self {
        Self::new(1, 1, 1, ForegroundValue::ONE, 0.0)
    }
}

impl VotingBinaryImageFilter {
    /// Apply a single voting step to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;
        let device = image.data().device();

        let r = self.radius;
        let birth = self.birth_threshold;
        let survival = self.survival_threshold;
        let fg = f32::from(self.foreground_value);
        let bg = self.background_value;

        let mut out = vec![bg; nz * ny * nx];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = vals[iz * ny * nx + iy * nx + ix];
                    let is_fg = (v - fg).abs() < 1e-5;

                    let z0 = iz.saturating_sub(r);
                    let z1 = (iz + r).min(nz - 1);
                    let y0 = iy.saturating_sub(r);
                    let y1 = (iy + r).min(ny - 1);
                    let x0 = ix.saturating_sub(r);
                    let x1 = (ix + r).min(nx - 1);
                    let mut fg_count = 0usize;
                    for kz in z0..=z1 {
                        for ky in y0..=y1 {
                            for kx in x0..=x1 {
                                if kz == iz && ky == iy && kx == ix {
                                    continue;
                                }
                                let kv = vals[kz * ny * nx + ky * nx + kx];
                                if (kv - fg).abs() < 1e-5 {
                                    fg_count += 1;
                                }
                            }
                        }
                    }

                    out[iz * ny * nx + iy * nx + ix] = if !is_fg {
                        if fg_count >= birth {
                            fg
                        } else {
                            bg
                        }
                    } else {
                        if fg_count >= survival {
                            fg
                        } else {
                            bg
                        }
                    };
                }
            }
        }

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
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

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
        img.data_slice().into_owned()
    }

    /// All-background image with birth_threshold=1 and high survival_threshold=0:
    /// no neighbours → fg_count=0 < birth_threshold → stays background.
    #[test]
    fn all_background_stays_background() {
        let img = make_image(vec![0.0f32; 27], [3, 3, 3]);
        let filter = VotingBinaryImageFilter::default(); // birth=1
        let out = filter.apply(&img).unwrap();
        // No foreground neighbours anywhere → all voxels stay background.
        assert!(voxels(&out).iter().all(|&v| v == 0.0));
    }

    /// All-foreground image: every voxel has neighbours ≥ survival_threshold=1 → stays foreground.
    #[test]
    fn all_foreground_survives() {
        let img = make_image(vec![1.0f32; 27], [3, 3, 3]);
        let filter = VotingBinaryImageFilter::default(); // survival=1
        let out = filter.apply(&img).unwrap();
        // Every fg voxel has at least 1 fg neighbour → survives.
        assert!(voxels(&out).iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    /// Single isolated foreground voxel with survival_threshold=1 dies (no fg neighbours).
    #[test]
    fn isolated_fg_dies_at_survival_threshold_1() {
        let mut data = vec![0.0f32; 27];
        data[13] = 1.0; // center voxel
        let img = make_image(data, [3, 3, 3]);
        let filter = VotingBinaryImageFilter::new(1, 1, 1, 1.0, 0.0);
        let out = filter.apply(&img).unwrap();
        // Center has fg_count=0 < survival=1 → dies.
        assert_eq!(voxels(&out)[13], 0.0);
    }

    /// Birth: background voxel adjacent to a foreground cluster is born
    /// when fg_count >= birth_threshold.
    #[test]
    fn birth_from_fg_neighbor() {
        // 1×1×3: [1, 1, 0]. With r=1, birth=1:
        // voxel 2 (bg): neighbours = [1,1] → fg_count=2 ≥ 1 → born.
        let img = make_image(vec![1.0, 1.0, 0.0], [1, 1, 3]);
        let filter = VotingBinaryImageFilter::new(1, 1, 1, 1.0, 0.0);
        let out = filter.apply(&img).unwrap();
        let v = voxels(&out);
        assert!(
            (v[2] - 1.0).abs() < 1e-5,
            "voxel 2 should be born, got {}",
            v[2]
        );
    }

    /// Spatial metadata preserved.
    #[test]
    fn preserves_metadata() {
        let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let out = VotingBinaryImageFilter::default().apply(&img).unwrap();
        assert_eq!(out.shape(), [2, 2, 2]);
        assert_eq!(*out.origin(), *img.origin());
    }
}
