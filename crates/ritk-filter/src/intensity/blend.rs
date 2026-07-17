//! Alpha blending filter for pixel-level image fusion.
//!
//! # Mathematical Specification
//!
//! Let `A, B : ℤ³ → ℝ` be two images with identical shape `[nz, ny, nx]`,
//! and let `α ∈ [0, 1]` be a blending weight.
//!
//! `out(x) = (1 - α) * A(x) + α * B(x)`
//!
//! Spatial metadata (origin, spacing, direction) is taken from the **first** input image.
//! Both images must have identical shapes; a shape mismatch returns `Err`.
//!
//! # ITK Parity
//!
//! `itk::BlendImageFilter`

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_annotation::overlay::Opacity;
use ritk_image::tensor::Backend;
use ritk_image::{native::Image as NativeImage, Image};
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Linearly blend two co-registered images.
///
/// `out(x) = (1 - alpha) * a(x) + alpha * b(x)`
///
/// - `alpha = 0.0` yields exactly `a(x)`.
/// - `alpha = 1.0` yields exactly `b(x)`.
///
/// # ITK Parity: `BlendImageFilter`
#[derive(Debug, Clone)]
pub struct BlendImageFilter {
    pub alpha: Opacity,
}

impl Default for BlendImageFilter {
    fn default() -> Self {
        Self {
            alpha: Opacity::new(0.5),
        }
    }
}

impl BlendImageFilter {
    /// Create a new blend filter with the given alpha [0.0, 1.0].
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: Opacity::new(alpha),
        }
    }

    /// Set the alpha blending weight.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = Opacity::new(alpha);
        self
    }

    pub fn apply<B: Backend>(
        &self,
        a: &Image<B, 3>,
        b: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        anyhow::ensure!(
            a.shape() == b.shape(),
            "BlendImageFilter: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        );

        let (av_vec, dims) = extract_vec_infallible(a);
        let av = &av_vec;
        let (bv_vec, _) = extract_vec_infallible(b);
        let bv = &bv_vec;

        let alpha = self.alpha.get();
        let one_minus_alpha = 1.0 - alpha;

        let out: Vec<f32> = av
            .iter()
            .zip(bv.iter())
            .map(|(&x, &y)| one_minus_alpha * x + alpha * y)
            .collect();

        Ok(rebuild(out, dims, a))
    }

    /// Blend two co-registered Coeus-native images.
    pub fn apply_native<B>(
        &self,
        a: &NativeImage<f32, B, 3>,
        b: &NativeImage<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<NativeImage<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        anyhow::ensure!(
            a.shape() == b.shape(),
            "BlendImageFilter: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
        let alpha = self.alpha.get();
        let complement = 1.0 - alpha;
        let values = a
            .data_slice()?
            .iter()
            .zip(b.data_slice()?)
            .map(|(&left, &right)| complement * left + alpha * right)
            .collect();
        NativeImage::from_flat_on(
            values,
            a.shape(),
            *a.origin(),
            *a.spacing(),
            *a.direction(),
            backend,
        )
    }
}
