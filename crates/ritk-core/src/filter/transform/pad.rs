//! Padding image filters: constant, mirror, and wrap pad.
//!
//! # Mathematical Specification
//!
//! All three filters extend an image `I` with shape `[Nz, Ny, Nx]` by `[lz,ly,lx]`
//! voxels on the lower boundary and `[uz,uy,ux]` voxels on the upper boundary,
//! producing an output of shape:
//!
//! ```text
//! [Nz+lz+uz, Ny+ly+uy, Nx+lx+ux]
//! ```
//!
//! The three strategies differ in how the padded voxels are filled:
//!
//! ## ConstantPadImageFilter
//! Padded voxels are filled with a fixed constant `c`.
//! - `out(iz,iy,ix) = I(iz−lz, iy−ly, ix−lx)` for interior voxels.
//! - `out(iz,iy,ix) = c` otherwise.
//!
//! ## MirrorPadImageFilter
//! Padded voxels are filled by reflecting the image about its boundaries.
//! - `reflect(k, N) = |2·N − 2 − (k mod 2(N−1))| mod (2(N−1))`
//!   (period-`2(N−1)` symmetric extension, matching ITK MirrorPadImageFilter).
//!   For `N=1`, reflected index = 0 always.
//!
//! ## WrapPadImageFilter
//! Padded voxels are filled by periodic extension:
//! - `wrap(k, N) = ((k mod N) + N) mod N`.
//!
//! # ITK Parity
//!
//! - `itk::ConstantPadImageFilter`: constant = 0 by default.
//! - `itk::MirrorPadImageFilter`: symmetric reflection.
//! - `itk::WrapPadImageFilter`: periodic extension.
//!
//! Origin is updated so the physical position of voxel `(0,0,0)` of the output
//! corresponds to the physical position of voxel `(−lz, −ly, −lx)` of the input.
//!
//! # Reference
//!
//! - ITK Software Guide, 2nd ed., §6.2 Padding.

use crate::image::Image;
use crate::spatial::{Point, Spacing};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Shared ───────────────────────────────────────────────────────────────────

fn updated_origin(origin: &Point<3>, spacing: &Spacing<3>, lower: [usize; 3]) -> Point<3> {
    // Output origin = input_origin - lower * spacing (in each axis).
    Point::new([
        origin[0] - lower[0] as f64 * spacing[0],
        origin[1] - lower[1] as f64 * spacing[1],
        origin[2] - lower[2] as f64 * spacing[2],
    ])
}

// ── ConstantPadImageFilter ────────────────────────────────────────────────────

/// Constant-value padding filter.
///
/// Extends the image by inserting `constant` values in the padded region.
#[derive(Debug, Clone)]
pub struct ConstantPadImageFilter {
    /// Number of voxels to add on the lower side per axis `[lz, ly, lx]`.
    pub pad_lower: [usize; 3],
    /// Number of voxels to add on the upper side per axis `[uz, uy, ux]`.
    pub pad_upper: [usize; 3],
    /// Constant fill value. Default 0.0.
    pub constant: f32,
}

impl ConstantPadImageFilter {
    /// Construct with explicit parameters.
    pub fn new(pad_lower: [usize; 3], pad_upper: [usize; 3], constant: f32) -> Self {
        Self {
            pad_lower,
            pad_upper,
            constant,
        }
    }
}

impl Default for ConstantPadImageFilter {
    fn default() -> Self {
        Self::new([1, 1, 1], [1, 1, 1], 0.0)
    }
}

impl ConstantPadImageFilter {
    /// Apply the constant pad filter.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let [nz, ny, nx] = image.shape();
        let device = image.data().device();
        let [lz, ly, lx] = self.pad_lower;
        let [uz, uy, ux] = self.pad_upper;
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let td = image.data().clone().into_data();
        let vals: &[f32] = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("ConstantPadImageFilter: {:?}", e))?;

        let c = self.constant;

        let mut out = vec![c; oz * oy * ox];

        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    if iz >= lz
                        && iz < lz + nz
                        && iy >= ly
                        && iy < ly + ny
                        && ix >= lx
                        && ix < lx + nx
                    {
                        let sz = iz - lz;
                        let sy = iy - ly;
                        let sx = ix - lx;
                        out[iz * oy * ox + iy * ox + ix] = vals[sz * ny * nx + sy * nx + sx];
                    }
                }
            }
        }

        let new_origin = updated_origin(image.origin(), image.spacing(), [lz, ly, lx]);
        let shape = Shape::new([oz, oy, ox]);
        let data = TensorData::new(out, shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(
            tensor,
            new_origin,
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── MirrorPadImageFilter ──────────────────────────────────────────────────────

/// Mirror (reflective) padding filter.
///
/// Extends the image by symmetric reflection about each boundary.
#[derive(Debug, Clone)]
pub struct MirrorPadImageFilter {
    /// Number of voxels to add on the lower side per axis.
    pub pad_lower: [usize; 3],
    /// Number of voxels to add on the upper side per axis.
    pub pad_upper: [usize; 3],
}

impl MirrorPadImageFilter {
    /// Construct with explicit parameters.
    pub fn new(pad_lower: [usize; 3], pad_upper: [usize; 3]) -> Self {
        Self {
            pad_lower,
            pad_upper,
        }
    }
}

impl Default for MirrorPadImageFilter {
    fn default() -> Self {
        Self::new([1, 1, 1], [1, 1, 1])
    }
}

/// Mirror-reflect index `i` into the range `[0, n)`.
///
/// Uses the symmetric extension formula: period = 2*(n-1), then fold.
/// For n=1 always returns 0.
#[inline]
fn mirror_index(i: i64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as i64 - 1);
    // Reduce to [0, period).
    let r = ((i % period) + period) % period;
    if r < n as i64 {
        r as usize
    } else {
        (period - r) as usize
    }
}

impl MirrorPadImageFilter {
    /// Apply the mirror pad filter.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let [nz, ny, nx] = image.shape();
        let device = image.data().device();
        let [lz, ly, lx] = self.pad_lower;
        let [uz, uy, ux] = self.pad_upper;
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let td = image.data().clone().into_data();
        let vals: &[f32] = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("MirrorPadImageFilter: {:?}", e))?;

        let mut out = vec![0.0f32; oz * oy * ox];

        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    let sz = mirror_index(iz as i64 - lz as i64, nz);
                    let sy = mirror_index(iy as i64 - ly as i64, ny);
                    let sx = mirror_index(ix as i64 - lx as i64, nx);
                    out[iz * oy * ox + iy * ox + ix] = vals[sz * ny * nx + sy * nx + sx];
                }
            }
        }

        let new_origin = updated_origin(image.origin(), image.spacing(), [lz, ly, lx]);
        let shape = Shape::new([oz, oy, ox]);
        let data = TensorData::new(out, shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(
            tensor,
            new_origin,
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── WrapPadImageFilter ────────────────────────────────────────────────────────

/// Wrap (periodic) padding filter.
///
/// Extends the image by periodic replication (tile the image in all directions).
#[derive(Debug, Clone)]
pub struct WrapPadImageFilter {
    /// Number of voxels to add on the lower side per axis.
    pub pad_lower: [usize; 3],
    /// Number of voxels to add on the upper side per axis.
    pub pad_upper: [usize; 3],
}

impl WrapPadImageFilter {
    /// Construct with explicit parameters.
    pub fn new(pad_lower: [usize; 3], pad_upper: [usize; 3]) -> Self {
        Self {
            pad_lower,
            pad_upper,
        }
    }
}

impl Default for WrapPadImageFilter {
    fn default() -> Self {
        Self::new([1, 1, 1], [1, 1, 1])
    }
}

#[inline]
fn wrap_index(i: i64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let n = n as i64;
    ((i % n) + n) as usize % n as usize
}

impl WrapPadImageFilter {
    /// Apply the wrap pad filter.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let [nz, ny, nx] = image.shape();
        let device = image.data().device();
        let [lz, ly, lx] = self.pad_lower;
        let [uz, uy, ux] = self.pad_upper;
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let td = image.data().clone().into_data();
        let vals: &[f32] = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("WrapPadImageFilter: {:?}", e))?;

        let mut out = vec![0.0f32; oz * oy * ox];

        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    let sz = wrap_index(iz as i64 - lz as i64, nz);
                    let sy = wrap_index(iy as i64 - ly as i64, ny);
                    let sx = wrap_index(ix as i64 - lx as i64, nx);
                    out[iz * oy * ox + iy * ox + ix] = vals[sz * ny * nx + sy * nx + sx];
                }
            }
        }

        let new_origin = updated_origin(image.origin(), image.spacing(), [lz, ly, lx]);
        let shape = Shape::new([oz, oy, ox]);
        let data = TensorData::new(out, shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(
            tensor,
            new_origin,
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::Direction;
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
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    // ── ConstantPadImageFilter tests ──────────────────────────────────────────

    /// Zero-padding: padded voxels filled with 0.
    #[test]
    fn constant_pad_zero() {
        // 1×1×2 image [3,7], pad by 1 on each side → 1×1×4 [0,3,7,0].
        let img = make_image(vec![3.0, 7.0], [1, 1, 2]);
        let out = ConstantPadImageFilter::new([0, 0, 1], [0, 0, 1], 0.0)
            .apply(&img)
            .unwrap();
        assert_eq!(out.shape(), [1, 1, 4]);
        let v = voxels(&out);
        assert_eq!(v, vec![0.0, 3.0, 7.0, 0.0]);
    }

    /// Custom constant pad value.
    #[test]
    fn constant_pad_custom_value() {
        let img = make_image(vec![5.0], [1, 1, 1]);
        let out = ConstantPadImageFilter::new([0, 0, 2], [0, 0, 2], -1.0)
            .apply(&img)
            .unwrap();
        assert_eq!(out.shape(), [1, 1, 5]);
        let v = voxels(&out);
        assert_eq!(v, vec![-1.0, -1.0, 5.0, -1.0, -1.0]);
    }

    /// Constant pad preserves spacing, updates origin.
    #[test]
    fn constant_pad_origin_updated() {
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(vec![1.0f32], Shape::new([1, 1, 1]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        // Origin at [0, 0, 10], spacing [1, 1, 2] — pad 1 voxel on lower X.
        let img2 = Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 10.0]),
            Spacing::new([1.0_f64, 1.0, 2.0]),
            Direction::identity(),
        );
        let out = ConstantPadImageFilter::new([0, 0, 1], [0, 0, 0], 0.0)
            .apply(&img2)
            .unwrap();
        // Origin x (axis 2) shifts by -1 * spacing[2] = -1 * 2.0 = -2.0 → new origin[2] = 10 - 2 = 8.
        let ox = out.origin()[2];
        assert!((ox - 8.0).abs() < 1e-10, "origin[2]={ox}");
        // Origin z (axis 0) unchanged (pad_lower[0] = 0).
        assert!((out.origin()[0]).abs() < 1e-10, "origin[0] should be 0");
    }

    // ── MirrorPadImageFilter tests ────────────────────────────────────────────

    /// Mirror pad: 1×1×3 = [1,2,3], pad 2 on each side → [3,2,1,2,3,2,1].
    #[test]
    fn mirror_pad_1d() {
        let img = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let out = MirrorPadImageFilter::new([0, 0, 2], [0, 0, 2])
            .apply(&img)
            .unwrap();
        assert_eq!(out.shape(), [1, 1, 7]);
        let v = voxels(&out);
        // Expected: period=4, mirror extension:
        // index -2 → mirror_index(-2,3): i=-2, period=4, r=((-2%4)+4)%4=2, r<3 → 2 → val=3
        // index -1 → r=3, 3>=3 → 4-3=1 → val=2
        // index 0..2 → original [1,2,3]
        // index 3 → r=3, 3>=3 → 4-3=1 → val=2
        // index 4 → r=0 → val=1
        assert!((v[0] - 3.0).abs() < 1e-5, "v[0]={}", v[0]);
        assert!((v[1] - 2.0).abs() < 1e-5, "v[1]={}", v[1]);
        assert!((v[2] - 1.0).abs() < 1e-5, "v[2]={}", v[2]);
        assert!((v[3] - 2.0).abs() < 1e-5, "v[3]={}", v[3]);
        assert!((v[4] - 3.0).abs() < 1e-5, "v[4]={}", v[4]);
        assert!((v[5] - 2.0).abs() < 1e-5, "v[5]={}", v[5]);
        assert!((v[6] - 1.0).abs() < 1e-5, "v[6]={}", v[6]);
    }

    /// Mirror index formula for n=1 always returns 0.
    #[test]
    fn mirror_index_n1() {
        for i in -5i64..=5 {
            assert_eq!(super::mirror_index(i, 1), 0);
        }
    }

    // ── WrapPadImageFilter tests ──────────────────────────────────────────────

    /// Wrap pad: 1×1×3 = [A,B,C], pad 2 on each side → [B,C,A,B,C,A,B].
    #[test]
    fn wrap_pad_1d() {
        let img = make_image(vec![10.0, 20.0, 30.0], [1, 1, 3]);
        let out = WrapPadImageFilter::new([0, 0, 2], [0, 0, 2])
            .apply(&img)
            .unwrap();
        assert_eq!(out.shape(), [1, 1, 7]);
        let v = voxels(&out);
        // index shifts: output i → input wrap(i-2, 3)
        // i=0: wrap(-2,3)=1 → 20
        // i=1: wrap(-1,3)=2 → 30
        // i=2: wrap(0,3)=0 → 10
        // i=3: wrap(1,3)=1 → 20
        // i=4: wrap(2,3)=2 → 30
        // i=5: wrap(3,3)=0 → 10
        // i=6: wrap(4,3)=1 → 20
        assert!((v[0] - 20.0).abs() < 1e-5, "v[0]={}", v[0]);
        assert!((v[2] - 10.0).abs() < 1e-5, "v[2]={}", v[2]);
        assert!((v[5] - 10.0).abs() < 1e-5, "v[5]={}", v[5]);
    }

    /// Output shape correct for wrap pad.
    #[test]
    fn wrap_pad_shape() {
        let img = make_image(vec![0.0f32; 24], [2, 3, 4]);
        let out = WrapPadImageFilter::new([1, 2, 3], [1, 2, 3])
            .apply(&img)
            .unwrap();
        assert_eq!(out.shape(), [4, 7, 10]);
    }
}
