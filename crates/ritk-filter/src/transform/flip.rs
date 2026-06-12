//! Axis-flip image filter.
//!
//! # Mathematical Specification
//!
//! For a 3-D image `I : ℤ³ → ℝ` with shape `[nz, ny, nx]` and flip policies
//! `flip = [fz, fy, fx]` where each `f*` is [`FlipPolicy::Keep`] or [`FlipPolicy::Flip`]:
//!
//! `out(iz, iy, ix) = I(iz', iy', ix')`
//!
//! where:
//! - `iz' = if fz { nz − 1 − iz } else { iz }`
//! - `iy' = if fy { ny − 1 − iy } else { iy }`
//! - `ix' = if fx { nx − 1 − ix } else { ix }`
//!
//! # Properties
//!
//! - Involutory: applying the same flip twice returns the original image.
//! - Preserves shape and all spatial metadata.
//! - `flip = [Keep, Keep, Keep]` is the identity transform.
//! - O(N) time and O(N) output space.
//!
//! # ITK / ImageJ Parity
//!
//! | Filter             | ITK class          | ImageJ (Image > Transform)  |
//! |--------------------|--------------------|-----------------------------|
//! | `FlipImageFilter`  | `FlipImageFilter`  | Flip Horizontally / Vertically |

use ritk_tensor_ops::{extract_vec_infallible, rebuild};
use ritk_image::Image;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

/// Per-axis flip policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum FlipPolicy {
    /// Keep the axis as-is.
    #[default]
    Keep,
    /// Flip this axis.
    Flip,
}

impl From<bool> for FlipPolicy {
    fn from(flip: bool) -> Self {
        if flip {
            FlipPolicy::Flip
        } else {
            FlipPolicy::Keep
        }
    }
}

impl From<FlipPolicy> for bool {
    fn from(policy: FlipPolicy) -> Self {
        matches!(policy, FlipPolicy::Flip)
    }
}

/// Flip a 3-D image along any combination of the Z, Y, and X axes.
///
/// # Example
///
/// ```rust,ignore
/// // Flip along the Z axis only
/// let out = FlipImageFilter::new([FlipPolicy::Flip, FlipPolicy::Keep, FlipPolicy::Keep])
///     .apply(&image)?;
/// ```
#[derive(Debug, Clone)]
pub struct FlipImageFilter {
    /// Which axes to flip: `[flip_z, flip_y, flip_x]`.
    pub axes: [FlipPolicy; 3],
}

impl FlipImageFilter {
    pub fn new(axes: [FlipPolicy; 3]) -> Self {
        Self { axes }
    }

    /// Construct from a boolean array (`true` = flip, `false` = keep).
    pub fn from_bools(axes: [bool; 3]) -> Self {
        Self {
            axes: axes.map(FlipPolicy::from),
        }
    }

    /// Convenience constructor: flip Z axis only.
    pub fn flip_z() -> Self {
        Self::new([FlipPolicy::Flip, FlipPolicy::Keep, FlipPolicy::Keep])
    }

    /// Convenience constructor: flip Y axis only.
    pub fn flip_y() -> Self {
        Self::new([FlipPolicy::Keep, FlipPolicy::Flip, FlipPolicy::Keep])
    }

    /// Convenience constructor: flip X axis only.
    pub fn flip_x() -> Self {
        Self::new([FlipPolicy::Keep, FlipPolicy::Keep, FlipPolicy::Flip])
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec_infallible(image);
        let vals = &vals_vec;

        let [nz, ny, nx] = dims;
        let [fz, fy, fx] = self.axes;
        let mut out = vec![0.0f32; nz * ny * nx];

        for iz in 0..nz {
            let iz_src = if matches!(fz, FlipPolicy::Flip) {
                nz - 1 - iz
            } else {
                iz
            };
            for iy in 0..ny {
                let iy_src = if matches!(fy, FlipPolicy::Flip) {
                    ny - 1 - iy
                } else {
                    iy
                };
                for ix in 0..nx {
                    let ix_src = if matches!(fx, FlipPolicy::Flip) {
                        nx - 1 - ix
                    } else {
                        ix
                    };
                    let dst = iz * ny * nx + iy * nx + ix;
                    let src = iz_src * ny * nx + iy_src * nx + ix_src;
                    out[dst] = vals[src];
                }
            }
        }

        Ok(rebuild(out, dims, image))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_tensor_ops::extract_vec_infallible;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
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

    /// No-flip is identity.
    #[test]
    fn flip_none_is_identity() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = make_image(vals.clone(), [2, 2, 2]);
        let out = FlipImageFilter::new([FlipPolicy::Keep; 3])
            .apply(&img)
            .unwrap();
        let v = voxels(&out);
        for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "[{}] expected {}, got {}", i, b, a);
        }
    }

    /// Flip X on a 1×1×4 array reverses the sequence.
    #[test]
    fn flip_x_reverses_x_axis() {
        let img = make_image(vec![1.0f32, 2.0, 3.0, 4.0], [1, 1, 4]);
        let out = FlipImageFilter::flip_x().apply(&img).unwrap();
        let v = voxels(&out);
        let expected = [4.0f32, 3.0, 2.0, 1.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "[{}] expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    /// Flip Z on a 4×1×1 array reverses the sequence.
    #[test]
    fn flip_z_reverses_z_axis() {
        let img = make_image(vec![10.0f32, 20.0, 30.0, 40.0], [4, 1, 1]);
        let out = FlipImageFilter::flip_z().apply(&img).unwrap();
        let v = voxels(&out);
        let expected = [40.0f32, 30.0, 20.0, 10.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "[{}] expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    /// Applying the same flip twice returns the original image (involutory property).
    #[test]
    fn flip_twice_returns_original() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let img = make_image(vals.clone(), [2, 2, 2]);
        let flip = FlipImageFilter::new([FlipPolicy::Flip, FlipPolicy::Flip, FlipPolicy::Keep]);
        let out1 = flip.apply(&img).unwrap();
        let out2 = flip.apply(&out1).unwrap();
        let v = voxels(&out2);
        for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "[{}] double-flip expected {}, got {}",
                i,
                b,
                a
            );
        }
    }

    /// Flip preserves shape and spatial metadata.
    #[test]
    fn flip_preserves_spatial_metadata() {
        let img = make_image(vec![1.0f32; 8], [2, 2, 2]);
        let out = FlipImageFilter::flip_y().apply(&img).unwrap();
        assert_eq!(out.shape(), img.shape());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
    }

    /// Flip all three axes on a 2×3×4 volume and verify specific voxel.
    #[test]
    fn flip_all_axes_2x3x4_correctness() {
        // dims [nz=2, ny=3, nx=4]; voxel (iz,iy,ix) has value iz*100 + iy*10 + ix
        let dims = [2usize, 3, 4];
        let mut vals = vec![0.0f32; 2 * 3 * 4];
        for iz in 0..2usize {
            for iy in 0..3usize {
                for ix in 0..4usize {
                    vals[iz * 12 + iy * 4 + ix] = (iz * 100 + iy * 10 + ix) as f32;
                }
            }
        }
        let img = make_image(vals, dims);
        let out = FlipImageFilter::new([FlipPolicy::Flip; 3])
            .apply(&img)
            .unwrap();
        let v = voxels(&out);
        // After flipping all axes: out(iz,iy,ix) = in(nz-1-iz, ny-1-iy, nx-1-ix)
        for iz in 0..2usize {
            for iy in 0..3usize {
                for ix in 0..4usize {
                    let got = v[iz * 12 + iy * 4 + ix];
                    let iz_s = 1 - iz;
                    let iy_s = 2 - iy;
                    let ix_s = 3 - ix;
                    let exp = (iz_s * 100 + iy_s * 10 + ix_s) as f32;
                    assert!(
                        (got - exp).abs() < 1e-4,
                        "({},{},{}): expected {}, got {}",
                        iz,
                        iy,
                        ix,
                        exp,
                        got
                    );
                }
            }
        }
    }
}
