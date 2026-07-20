//! Discrete Gaussian smoothing filter (ITK DiscreteGaussianImageFilter parity).
//!
//! # Mathematical Specification
//!
//! Given variance v_d (physical units^2) and voxel spacing h_d, the pixel
//! variance is t = v_d / h_d^2. The kernel is ITK's *discrete* Gaussian
//! (`GaussianOperator`), the discrete analog of the Gaussian (Lindeberg 1990):
//!   g\[k\] = e^{-t} · I_{|k|}(t)   for k in {-r,...,r}
//! where I_n is the modified Bessel function of the first kind. One-sided
//! coefficients are accumulated until the mass g\[0\] + 2·Sum_{i>=1} g\[i\]
//! reaches 1 - maximum_error (radius capped at 32), then normalised by that sum.
//! This is NOT a sampled continuous Gaussian — it is float-exact to SimpleITK.
//!
//! # Boundary Conditions
//! Replicate (edge) padding is used for all convolutions. This preserves the
//! invariant conv(constant_image, normalized_kernel) = constant_image for
//! ALL voxels, including boundaries.
//!
//! # ITK Parity
//!   - Parameterized by variance (sigma^2), not sigma.
//!   - maximum_error truncation criterion (default 0.01).
//!   - use_image_spacing flag (default true).
//!
//! # References
//! - Young et al. (1995). Fundamentals of Image Processing. Delft UT.
//! - ITK Software Guide 2nd Ed., Sec 6.3.1.

mod bessel;
mod convolve;

#[cfg(test)]
mod tests_native;

use bessel::{modified_bessel_i, modified_bessel_i0, modified_bessel_i1};
pub(crate) use convolve::convolve_separable;

use crate::edge::GaussianSigma;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_image::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Controls whether Gaussian variance is specified in physical or pixel units.
///
/// - `Physical` (default): variance is in physical (world) units; converted to
///   pixel units using image spacing (`sigma_pixel = sqrt(v) / h_d`).
/// - `Voxel`: variance is already in pixel (voxel) units; no conversion applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpacingMode {
    /// Ignore image spacing; treat variance as voxel-space variance.
    Voxel,
    /// Use physical spacing to convert variance to pixel sigma (ITK default).
    #[default]
    Physical,
}

impl std::fmt::Display for SpacingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpacingMode::Physical => f.write_str("physical"),
            SpacingMode::Voxel => f.write_str("voxel"),
        }
    }
}

impl std::str::FromStr for SpacingMode {
    type Err = String;

    /// Parse a spacing mode string.
    ///
    /// Accepts `"physical"` / `"true"` (legacy) and `"voxel"` / `"pixel"` /
    /// `"false"` (legacy).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "physical" | "true" => Ok(SpacingMode::Physical),
            "voxel" | "pixel" | "false" => Ok(SpacingMode::Voxel),
            _ => Err(format!("expected 'physical' or 'voxel', got '{s}'")),
        }
    }
}

/// Discrete Gaussian smoothing filter (ITK DiscreteGaussianImageFilter parity).
///
/// Applies separable 1-D Gaussian convolution along each image dimension.
/// Replicate (edge) padding preserves the uniform-image identity invariant.
pub struct DiscreteGaussianFilter<B: Backend> {
    variance: Vec<f64>,
    maximum_error: f64,
    spacing_mode: SpacingMode,
    _b: PhantomData<fn() -> B>,
}

impl<B: Backend> DiscreteGaussianFilter<B> {
    /// Create with per-dimension Gaussian sigmas (physical units).
    ///
    /// Variance is computed internally as `sigma²` for each dimension.
    /// Panics if `sigmas` is empty.
    pub fn new(sigmas: Vec<GaussianSigma>) -> Self {
        assert!(!sigmas.is_empty(), "sigmas list must not be empty");
        let variance: Vec<f64> = sigmas.iter().map(|s| s.get().powi(2)).collect();
        Self {
            variance,
            maximum_error: 0.01,
            spacing_mode: SpacingMode::Physical,
            _b: PhantomData,
        }
    }

    /// Create a filter with the same variance applied to all image dimensions.
    ///
    /// Equivalent to `new(vec![variance])` when the single variance value is
    /// broadcast to every dimension by `variance_for_dim`.
    pub fn new_isotropic(variance: f64) -> Self {
        assert!(variance >= 0.0, "variance must be >= 0, got {variance}");
        Self {
            variance: vec![variance],
            maximum_error: 0.01,
            spacing_mode: SpacingMode::Physical,
            _b: PhantomData,
        }
    }

    /// Set maximum_error (default 0.01). Panics if not in (0,1).
    pub fn with_maximum_error(mut self, maximum_error: f64) -> Self {
        assert!(
            maximum_error > 0.0 && maximum_error < 1.0,
            "maximum_error must be in (0, 1), got {maximum_error}"
        );
        self.maximum_error = maximum_error;
        self
    }

    /// Set spacing mode (default `Physical`).
    pub fn with_spacing_mode(mut self, mode: SpacingMode) -> Self {
        self.spacing_mode = mode;
        self
    }

    /// Apply the filter; spatial metadata preserved.
    pub fn apply<const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        let (tensor, origin, spacing, direction) = image.clone().into_parts();
        let smoothed = self.apply_inner(tensor, &spacing);
        Image::new(smoothed, origin, spacing, direction)
            .expect("filter preserves the statically validated image rank")
    }

    /// Build the per-axis discrete-Gaussian kernels for the given spacing.
    ///
    /// Delegates to the native [`discrete_gaussian_kernels`] free function
    /// (single source of truth for kernel construction), shared by the Coeus
    /// [`apply_inner`](Self::apply_inner) path, the Coeus-native
    /// [`apply_native`](Self::apply_native) path, and the native
    /// [`discrete_gaussian_smooth_flat`] core the Canny filters call.
    fn kernels_for_spacing<const D: usize>(
        &self,
        spacing: &ritk_spatial::Spacing<D>,
    ) -> [Option<Vec<f32>>; D] {
        discrete_gaussian_kernels::<D>(
            &self.variance,
            self.maximum_error,
            self.spacing_mode,
            spacing,
        )
    }

    #[inline]
    fn apply_inner<const D: usize>(
        &self,
        data: Tensor<f32, B>,
        spacing: &ritk_spatial::Spacing<D>,
    ) -> Tensor<f32, B> {
        let dims: [usize; D] = data
            .shape()
            .try_into()
            .expect("DiscreteGaussianFilter preserves the const-generic image rank");

        let kernels = self.kernels_for_spacing::<D>(spacing);
        if kernels.iter().all(|k| k.is_none()) {
            return data;
        }

        let flat = data.to_vec();
        let result = convolve_separable(flat, dims, &kernels);
        Tensor::<f32, B>::from_slice(dims, &result)
    }

    /// Coeus-native sister of [`DiscreteGaussianFilter::apply`].
    ///
    /// Runs the identical separable discrete-Gaussian convolution (ITK
    /// `GaussianOperator` kernel, replicate boundary) via the shared
    /// `kernels_for_spacing` kernel builder and the substrate-agnostic
    /// `convolve_separable` host core on the image's
    /// contiguous host buffer, so the result is bitwise-identical to the Coeus
    /// path. No Coeus tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt tensor fails shape validation.
    pub fn apply_native<BC>(
        &self,
        image: &ritk_image::Image<f32, BC, 3>,
    ) -> anyhow::Result<ritk_image::Image<f32, BC, 3>>
    where
        BC: coeus_core::ComputeBackend + Default,
        BC::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let kernels = self.kernels_for_spacing::<3>(image.spacing());
        let result = if kernels.iter().all(|k| k.is_none()) {
            vals
        } else {
            convolve_separable(vals, dims, &kernels)
        };
        ritk_tensor_ops::native::rebuild_image(result, dims, image, &BC::default())
    }
}

/// Minimum pixel variance below which a per-axis kernel is the identity impulse.
const VARIANCE_MIN: f64 = 1e-18;

/// Burn-free single source of truth for the per-axis discrete-Gaussian kernels.
///
/// Returns `[Option<Vec<f32>>; D]`: `None` for an axis whose pixel variance is
/// below [`VARIANCE_MIN`] or whose kernel collapses to the identity impulse
/// (length ≤ 1). `variance` is the per-axis physical-variance schedule
/// (broadcast from the last entry); `spacing_mode` selects the physical→pixel
/// conversion. Shared by [`DiscreteGaussianFilter::kernels_for_spacing`] and
/// [`discrete_gaussian_smooth_flat`].
pub(crate) fn discrete_gaussian_kernels<const D: usize>(
    variance: &[f64],
    maximum_error: f64,
    spacing_mode: SpacingMode,
    spacing: &ritk_spatial::Spacing<D>,
) -> [Option<Vec<f32>>; D] {
    std::array::from_fn(|d| {
        let v = if d < variance.len() {
            variance[d]
        } else {
            *variance
                .last()
                .expect("variance schedule must not be empty")
        };
        let sigma = match spacing_mode {
            SpacingMode::Physical => v.max(0.0).sqrt() / spacing[d].max(1e-12),
            SpacingMode::Voxel => v.max(0.0).sqrt(),
        };
        let pixel_variance = sigma * sigma;
        if pixel_variance >= VARIANCE_MIN {
            let kernel = gaussian_operator_1d(pixel_variance, maximum_error);
            if kernel.len() > 1 {
                return Some(kernel);
            }
        }
        None
    })
}

/// Burn-free host core: separable discrete-Gaussian smoothing on a flat z-major
/// buffer (ITK `GaussianOperator` kernel, replicate boundary via
/// `convolve_separable`). Bitwise-identical to
/// [`DiscreteGaussianFilter::apply`]/`apply_native`; used by the Canny filters
/// so their native paths need no Coeus backend to smooth.
pub(crate) fn discrete_gaussian_smooth_flat(
    vals: Vec<f32>,
    dims: [usize; 3],
    spacing: &ritk_spatial::Spacing<3>,
    variance: &[f64],
    maximum_error: f64,
    spacing_mode: SpacingMode,
) -> Vec<f32> {
    let kernels = discrete_gaussian_kernels::<3>(variance, maximum_error, spacing_mode, spacing);
    if kernels.iter().all(|k| k.is_none()) {
        vals
    } else {
        convolve_separable(vals, dims, &kernels)
    }
}

/// Maximum one-sided kernel radius (ITK `GaussianOperator` default).
const GAUSSIAN_MAX_KERNEL_RADIUS: usize = 32;

/// Build ITK's discrete Gaussian operator (`GaussianOperator`): the symmetric,
/// normalised coefficients `g[k] = e^{-t}·I_{|k|}(t)` where `t` is the pixel
/// variance and `I_n` is the modified Bessel function. One-sided coefficients
/// accumulate until the running mass reaches `1 − maximum_error`, capped at
/// radius 32. Float-exact to SimpleITK `DiscreteGaussian`.
pub(crate) fn gaussian_operator_1d(pixel_variance: f64, maximum_error: f64) -> Vec<f32> {
    let t = pixel_variance;
    let et = (-t).exp();
    let cap = 1.0 - maximum_error;

    let mut coeff = vec![et * modified_bessel_i0(t)];
    let mut sum = coeff[0];
    let g1 = et * modified_bessel_i1(t);
    coeff.push(g1);
    sum += 2.0 * g1;
    let mut i = 2;
    while sum < cap {
        let c = et * modified_bessel_i(i, t);
        if c <= 0.0 {
            break;
        }
        coeff.push(c);
        sum += 2.0 * c;
        if i >= GAUSSIAN_MAX_KERNEL_RADIUS {
            break;
        }
        i += 1;
    }

    let radius = coeff.len() - 1;
    let inv = 1.0 / sum;
    let mut k = Vec::with_capacity(2 * radius + 1);
    k.extend(coeff[1..].iter().rev().map(|&c| (c * inv) as f32));
    k.extend(coeff.iter().map(|&c| (c * inv) as f32));
    k
}

#[cfg(test)]
#[path = "../tests_discrete_gaussian.rs"]
mod tests;
