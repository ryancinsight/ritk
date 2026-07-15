//! FFT padding filter: pad to an FFT-efficient size.
//!
//! # Mathematical Specification
//!
//! `itk::FFTPadImageFilter` enlarges an image so each axis length becomes the
//! smallest value `>= N` whose greatest prime factor does not exceed a configured
//! limit `P` (default 5, i.e. a 2-3-5-smooth size). The extra voxels are split
//! symmetrically, with the smaller half on the lower boundary:
//!
//! ```text
//! G_d      = next_smooth_size(N_d, P)          // smallest >= N_d, gpf <= P
//! lower_d  = (G_d - N_d) / 2                    // integer floor
//! upper_d  = G_d - N_d - lower_d
//! ```
//!
//! The padded voxels are filled according to the chosen boundary condition, and
//! the origin shifts by `-lower_d * spacing_d` per axis — identical to the
//! underlying pad filters, which this filter delegates to. Padding is computed
//! independently per axis, so a unit axis (`N_d = 1`, already smooth) is left
//! unpadded.
//!
//! # ITK Parity
//!
//! `itk::FFTPadImageFilter` with `SetSizeGreatestPrimeFactor` and
//! `SetBoundaryCondition`. The boundary maps to the existing pad filters:
//! - [`FftPadBoundary::Zero`] -> [`ConstantPadImageFilter`] (constant 0)
//! - [`FftPadBoundary::ZeroFluxNeumann`] -> [`ZeroFluxNeumannPadImageFilter`] (default)
//! - [`FftPadBoundary::Periodic`] -> [`WrapPadImageFilter`]

use super::pad::{
    ConstantPadImageFilter, Padding, WrapPadImageFilter, ZeroFluxNeumannPadImageFilter,
};
use ritk_image::tensor::Backend;
use ritk_image::Image;

/// Boundary condition for the padded region, matching ITK `FFTPadImageFilter`'s
/// `SetBoundaryCondition` (sitk integer codes 0 / 1 / 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FftPadBoundary {
    /// Constant zero fill (`itk::ZeroFluxNeumannBoundaryCondition` is the ITK
    /// default; this code 0 is sitk's zero-pad option).
    Zero,
    /// Replicate the nearest edge voxel (ITK / sitk default).
    #[default]
    ZeroFluxNeumann,
    /// Periodic (wrap-around) extension.
    Periodic,
}

/// Greatest prime factor of `n` (`n >= 1`); returns 1 for `n == 1`.
///
/// Trial division by 2 then odd factors up to `sqrt(n)`; any residue `> 1` is a
/// prime larger than every tried factor and is therefore the greatest.
// `is_multiple_of` is not const-stable, and `% == 0` is the canonical trial-
// division form here.
#[allow(clippy::manual_is_multiple_of)]
const fn greatest_prime_factor(mut n: usize) -> usize {
    let mut g = 1;
    while n % 2 == 0 {
        n /= 2;
        g = 2;
    }
    let mut f = 3;
    while f * f <= n {
        while n % f == 0 {
            n /= f;
            g = f;
        }
        f += 2;
    }
    if n > 1 {
        n
    } else {
        g
    }
}

/// Smallest size `>= n` whose greatest prime factor is `<= limit`.
const fn next_smooth_size(n: usize, limit: usize) -> usize {
    let mut s = n;
    while greatest_prime_factor(s) > limit {
        s += 1;
    }
    s
}

/// FFT padding filter.
///
/// Enlarges each axis to the next prime-factor-bounded size and fills the new
/// voxels via the configured boundary condition.
#[derive(Debug, Clone, Copy)]
pub struct FftPadImageFilter {
    /// Largest permitted prime factor of each output axis length. Default 5.
    pub size_greatest_prime_factor: usize,
    /// How the padded region is filled. Default [`FftPadBoundary::ZeroFluxNeumann`].
    pub boundary: FftPadBoundary,
}

impl Default for FftPadImageFilter {
    fn default() -> Self {
        Self {
            size_greatest_prime_factor: 5,
            boundary: FftPadBoundary::ZeroFluxNeumann,
        }
    }
}

impl FftPadImageFilter {
    /// Construct with explicit parameters.
    pub fn new(size_greatest_prime_factor: usize, boundary: FftPadBoundary) -> Self {
        Self {
            size_greatest_prime_factor,
            boundary,
        }
    }

    /// Compute the symmetric lower/upper padding that enlarges `[nz, ny, nx]` to
    /// the next prime-factor-bounded size per axis.
    fn pad_extents(&self, shape: [usize; 3]) -> (Padding, Padding) {
        let p = self.size_greatest_prime_factor;
        let mut lower = [0usize; 3];
        let mut upper = [0usize; 3];
        let mut axis = 0;
        while axis < 3 {
            let n = shape[axis];
            let g = next_smooth_size(n, p);
            let lo = (g - n) / 2;
            lower[axis] = lo;
            upper[axis] = g - n - lo;
            axis += 1;
        }
        (Padding::new(lower), Padding::new(upper))
    }

    /// Apply the FFT pad filter.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (lower, upper) = self.pad_extents(image.shape());
        match self.boundary {
            FftPadBoundary::Zero => ConstantPadImageFilter::new(lower, upper, 0.0).apply(image),
            FftPadBoundary::ZeroFluxNeumann => {
                ZeroFluxNeumannPadImageFilter::new(lower, upper).apply(image)
            }
            FftPadBoundary::Periodic => WrapPadImageFilter::new(lower, upper).apply(image),
        }
    }

    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (lower, upper) = self.pad_extents(image.shape());
        match self.boundary {
            FftPadBoundary::Zero => {
                ConstantPadImageFilter::new(lower, upper, 0.0).apply_native(image, backend)
            }
            FftPadBoundary::ZeroFluxNeumann => {
                // TODO(native): replace this temporary Burn round-trip once
                // ZeroFluxNeumannPadImageFilter has a dedicated native path.
                let burn_image = ritk_image::Image::<burn_ndarray::NdArray<f32>, 3>::from_flat_on(
                    image.data_slice()?.to_vec(),
                    image.shape(),
                    *image.origin(),
                    *image.spacing(),
                    *image.direction(),
                    &Default::default(),
                );
                let padded = ZeroFluxNeumannPadImageFilter::new(lower, upper).apply(&burn_image)?;
                let (values, dims) = ritk_tensor_ops::extract_vec(&padded)?;
                ritk_image::native::Image::from_flat_on(
                    values,
                    dims,
                    *padded.origin(),
                    *padded.spacing(),
                    *padded.direction(),
                    backend,
                )
            }
            FftPadBoundary::Periodic => WrapPadImageFilter::new(lower, upper).apply_native(image, backend),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_fft_pad.rs"]
mod tests_fft_pad;
