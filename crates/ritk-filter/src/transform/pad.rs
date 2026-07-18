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
//! - `out(iz,iy,ix) = I(izâˆ’lz, iyâˆ’ly, ixâˆ’lx)` for interior voxels.
//! - `out(iz,iy,ix) = c` otherwise.
//!
//! ## MirrorPadImageFilter
//! Padded voxels are filled by reflecting the image about its boundaries.
//! - `reflect(k, N) = |2Â·N âˆ’ 2 âˆ’ (k mod 2(Nâˆ’1))| mod (2(Nâˆ’1))`
//!   (period-`2(Nâˆ’1)` symmetric extension, matching ITK MirrorPadImageFilter).
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
//! corresponds to the physical position of voxel `(âˆ’lz, âˆ’ly, âˆ’lx)` of the input.
//!
//! # Reference
//!
//! - ITK Software Guide, 2nd ed., Â§6.2 Padding.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_origin};

// â”€â”€ Padding newtype â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Per-axis padding extents for a 3-D volume: `[pad_z, pad_y, pad_x]`.
///
/// Each element is the number of voxels to pad along that axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Padding(pub [usize; 3]);

impl Padding {
    /// Construct from a per-axis array `[pad_z, pad_y, pad_x]`.
    pub fn new(pad: [usize; 3]) -> Self {
        Self(pad)
    }

    /// Zero padding on all axes.
    pub fn zero() -> Self {
        Self([0, 0, 0])
    }

    /// Borrow the inner array.
    pub fn as_array(&self) -> &[usize; 3] {
        &self.0
    }

    /// Sum of padding across all three axes.
    pub fn total(&self) -> usize {
        self.0[0] + self.0[1] + self.0[2]
    }
}

impl Default for Padding {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::ops::Index<usize> for Padding {
    type Output = usize;

    fn index(&self, axis: usize) -> &Self::Output {
        &self.0[axis]
    }
}

impl AsRef<[usize; 3]> for Padding {
    fn as_ref(&self) -> &[usize; 3] {
        &self.0
    }
}

// â”€â”€ Shared â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn updated_origin(
    origin: &Point<3>,
    spacing: &Spacing<3>,
    direction: &Direction<3>,
    lower: &Padding,
) -> Point<3> {
    Point::new(std::array::from_fn(|row| {
        origin[row]
            - (0..3)
                .map(|axis| lower[axis] as f64 * spacing[axis] * direction[(row, axis)])
                .sum::<f64>()
    }))
}

// â”€â”€ ConstantPadImageFilter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Constant-value padding filter.
///
/// Extends the image by inserting `constant` values in the padded region.
#[derive(Debug, Clone)]
pub struct ConstantPadImageFilter {
    /// Number of voxels to add on the lower side per axis `[lz, ly, lx]`.
    pub pad_lower: Padding,
    /// Number of voxels to add on the upper side per axis `[uz, uy, ux]`.
    pub pad_upper: Padding,
    /// Constant fill value. Default 0.0.
    pub constant: f32,
}

impl ConstantPadImageFilter {
    /// Construct with explicit parameters.
    pub fn new(pad_lower: Padding, pad_upper: Padding, constant: f32) -> Self {
        Self {
            pad_lower,
            pad_upper,
            constant,
        }
    }
}

impl Default for ConstantPadImageFilter {
    fn default() -> Self {
        Self::new(Padding::new([1, 1, 1]), Padding::new([1, 1, 1]), 0.0)
    }
}

impl ConstantPadImageFilter {
    /// Apply the constant pad filter.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let [nz, ny, nx] = image.shape();
        let [lz, ly, lx] = *self.pad_lower.as_array();
        let [uz, uy, ux] = *self.pad_upper.as_array();
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let (vals_vec, _) = extract_vec_infallible(image);
        let vals = &vals_vec;

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

        let new_origin = updated_origin(
            image.origin(),
            image.spacing(),
            image.direction(),
            &self.pad_lower,
        );
        Ok(rebuild_with_origin(out, [oz, oy, ox], new_origin, image))
    }

    /// Apply constant padding to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let [nz, ny, nx] = image.shape();
        let [lz, ly, lx] = *self.pad_lower.as_array();
        let [uz, uy, ux] = *self.pad_upper.as_array();
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];
        let values = image.data_slice()?;
        let mut output = vec![self.constant; oz * oy * ox];
        for iz in lz..lz + nz {
            for iy in ly..ly + ny {
                for ix in lx..lx + nx {
                    output[iz * oy * ox + iy * ox + ix] =
                        values[(iz - lz) * ny * nx + (iy - ly) * nx + ix - lx];
                }
            }
        }
        ritk_image::native::Image::from_flat_on(
            output,
            [oz, oy, ox],
            updated_origin(
                image.origin(),
                image.spacing(),
                image.direction(),
                &self.pad_lower,
            ),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

// â”€â”€ MirrorPadImageFilter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Mirror (reflective) padding filter.
///
/// Extends the image by symmetric reflection about each boundary.
#[derive(Debug, Clone)]
pub struct MirrorPadImageFilter {
    /// Number of voxels to add on the lower side per axis.
    pub pad_lower: Padding,
    /// Number of voxels to add on the upper side per axis.
    pub pad_upper: Padding,
}

impl MirrorPadImageFilter {
    /// Construct with explicit parameters.
    pub fn new(pad_lower: Padding, pad_upper: Padding) -> Self {
        Self {
            pad_lower,
            pad_upper,
        }
    }
}

impl Default for MirrorPadImageFilter {
    fn default() -> Self {
        Self::new(Padding::new([1, 1, 1]), Padding::new([1, 1, 1]))
    }
}

/// Mirror-reflect index `i` into the range `[0, n)` using ITK's symmetric
/// (whole-sample) convention: the boundary voxel **is repeated**, so index `-1`
/// maps to `0` and `n` maps to `n-1`. The extension is periodic with period
/// `2n` and fold `r < n ? r : 2n-1-r`, matching `itk::MirrorPadImageFilter` /
/// numpy `pad(mode="symmetric")`. (The reflect-without-repeat convention,
/// period `2(n-1)`, is *not* what ITK uses.) For `n = 1` always returns 0.
#[inline]
fn mirror_index(i: i64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * n as i64;
    let r = ((i % period) + period) % period;
    if r < n as i64 {
        r as usize
    } else {
        (period - 1 - r) as usize
    }
}

impl MirrorPadImageFilter {
    /// Apply the mirror pad filter.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let [nz, ny, nx] = image.shape();
        let [lz, ly, lx] = *self.pad_lower.as_array();
        let [uz, uy, ux] = *self.pad_upper.as_array();
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let (vals_vec, _) = extract_vec_infallible(image);
        let vals = &vals_vec;

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

        let new_origin = updated_origin(
            image.origin(),
            image.spacing(),
            image.direction(),
            &self.pad_lower,
        );
        Ok(rebuild_with_origin(out, [oz, oy, ox], new_origin, image))
    }

    /// Apply mirror padding to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        padded_native(
            image,
            backend,
            &self.pad_lower,
            &self.pad_upper,
            mirror_index,
        )
    }
}

// â”€â”€ WrapPadImageFilter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Wrap (periodic) padding filter.
///
/// Extends the image by periodic replication (tile the image in all directions).
#[derive(Debug, Clone)]
pub struct WrapPadImageFilter {
    /// Number of voxels to add on the lower side per axis.
    pub pad_lower: Padding,
    /// Number of voxels to add on the upper side per axis.
    pub pad_upper: Padding,
}

impl WrapPadImageFilter {
    /// Construct with explicit parameters.
    pub fn new(pad_lower: Padding, pad_upper: Padding) -> Self {
        Self {
            pad_lower,
            pad_upper,
        }
    }
}

impl Default for WrapPadImageFilter {
    fn default() -> Self {
        Self::new(Padding::new([1, 1, 1]), Padding::new([1, 1, 1]))
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
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let [nz, ny, nx] = image.shape();
        let [lz, ly, lx] = *self.pad_lower.as_array();
        let [uz, uy, ux] = *self.pad_upper.as_array();
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let (vals_vec, _) = extract_vec_infallible(image);
        let vals = &vals_vec;

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

        let new_origin = updated_origin(
            image.origin(),
            image.spacing(),
            image.direction(),
            &self.pad_lower,
        );
        Ok(rebuild_with_origin(out, [oz, oy, ox], new_origin, image))
    }

    /// Apply wrap padding to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        padded_native(image, backend, &self.pad_lower, &self.pad_upper, wrap_index)
    }
}

fn padded_native<B>(
    image: &ritk_image::native::Image<f32, B, 3>,
    backend: &B,
    lower: &Padding,
    upper: &Padding,
    index: fn(i64, usize) -> usize,
) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    let [nz, ny, nx] = image.shape();
    let [lz, ly, lx] = *lower.as_array();
    let [uz, uy, ux] = *upper.as_array();
    let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];
    let values = image.data_slice()?;
    let mut output = vec![0.0; oz * oy * ox];
    for iz in 0..oz {
        for iy in 0..oy {
            for ix in 0..ox {
                let z = index(iz as i64 - lz as i64, nz);
                let y = index(iy as i64 - ly as i64, ny);
                let x = index(ix as i64 - lx as i64, nx);
                output[iz * oy * ox + iy * ox + ix] = values[z * ny * nx + y * nx + x];
            }
        }
    }
    ritk_image::native::Image::from_flat_on(
        output,
        [oz, oy, ox],
        updated_origin(image.origin(), image.spacing(), image.direction(), lower),
        *image.spacing(),
        *image.direction(),
        backend,
    )
}

// â”€â”€ ZeroFluxNeumannPadImageFilter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Zero-flux Neumann (edge-replicate / clamp) padding filter.
///
/// Extends the image by repeating the nearest edge voxel â€” out-of-bounds index
/// `i` along an axis of length `n` reads `clamp(i, 0, n-1)`. Matches ITK
/// `ZeroFluxNeumannPadImageFilter` (`sitk.ZeroFluxNeumannPad`).
#[derive(Debug, Clone)]
pub struct ZeroFluxNeumannPadImageFilter {
    /// Number of voxels to add on the lower side per axis.
    pub pad_lower: Padding,
    /// Number of voxels to add on the upper side per axis.
    pub pad_upper: Padding,
}

impl ZeroFluxNeumannPadImageFilter {
    /// Construct with explicit parameters.
    pub fn new(pad_lower: Padding, pad_upper: Padding) -> Self {
        Self {
            pad_lower,
            pad_upper,
        }
    }
}

impl Default for ZeroFluxNeumannPadImageFilter {
    fn default() -> Self {
        Self::new(Padding::new([1, 1, 1]), Padding::new([1, 1, 1]))
    }
}

#[inline]
fn clamp_index(i: i64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    i.clamp(0, n as i64 - 1) as usize
}

impl ZeroFluxNeumannPadImageFilter {
    /// Apply the zero-flux Neumann (edge-replicate) pad filter.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let [nz, ny, nx] = image.shape();
        let [lz, ly, lx] = *self.pad_lower.as_array();
        let [uz, uy, ux] = *self.pad_upper.as_array();
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let (vals_vec, _) = extract_vec_infallible(image);
        let vals = &vals_vec;

        let mut out = vec![0.0f32; oz * oy * ox];

        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    let sz = clamp_index(iz as i64 - lz as i64, nz);
                    let sy = clamp_index(iy as i64 - ly as i64, ny);
                    let sx = clamp_index(ix as i64 - lx as i64, nx);
                    out[iz * oy * ox + iy * ox + ix] = vals[sz * ny * nx + sy * nx + sx];
                }
            }
        }

        let new_origin = updated_origin(
            image.origin(),
            image.spacing(),
            image.direction(),
            &self.pad_lower,
        );
        Ok(rebuild_with_origin(out, [oz, oy, ox], new_origin, image))
    }

    /// Coeus-native sister of [`ZeroFluxNeumannPadImageFilter::apply`].
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let [nz, ny, nx] = image.shape();
        let [lz, ly, lx] = *self.pad_lower.as_array();
        let [uz, uy, ux] = *self.pad_upper.as_array();
        let [oz, oy, ox] = [nz + lz + uz, ny + ly + uy, nx + lx + ux];

        let vals = image.data_slice()?;
        let mut out = vec![0.0f32; oz * oy * ox];

        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    let sz = clamp_index(iz as i64 - lz as i64, nz);
                    let sy = clamp_index(iy as i64 - ly as i64, ny);
                    let sx = clamp_index(ix as i64 - lx as i64, nx);
                    out[iz * oy * ox + iy * ox + ix] = vals[sz * ny * nx + sy * nx + sx];
                }
            }
        }

        let new_origin = updated_origin(
            image.origin(),
            image.spacing(),
            image.direction(),
            &self.pad_lower,
        );
        ritk_image::native::Image::from_flat_on(
            out,
            [oz, oy, ox],
            new_origin,
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_pad.rs"]
mod tests_pad;
