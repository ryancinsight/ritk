//! White and black top-hat filters for 3-D grayscale images.
//!
//! WTH_B(f)(x) = f(x) - opening_B(f)(x) = f(x) - D_B(E_B(f))(x)
//! BTH_B(f)(x) = closing_B(f)(x) - f(x) = E_B(D_B(f))(x) - f(x)
//!
//! Properties:
//! - WTH and BTH of constant images are 0.
//! - WTH(f) >= 0 (opening is anti-extensive).
//! - BTH(f) >= 0 (closing is extensive).
//!
//! References:
//! - Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press.
//! - Soille, P. (2003). Morphological Image Analysis, 2nd ed. Springer.

use super::grayscale_dilation::dilate_3d;
use super::grayscale_erosion::erode_3d;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// White top-hat filter: WTH_B(f) = f - opening_B(f).
/// Isolates bright structures smaller than the structuring element.
#[derive(Debug, Clone)]
pub struct WhiteTopHatFilter {
    pub radius: usize,
}
impl WhiteTopHatFilter {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let result = white_top_hat_vec(&vals, dims, self.radius);
        Ok(rebuild(result, dims, image))
    }

    /// Coeus-native sister of [`WhiteTopHatFilter::apply`].
    ///
    /// Runs the identical `f - D_B(E_B(f))` clamped subtraction via the shared
    /// `white_top_hat_vec` host core on the image's contiguous host buffer, so
    /// the result is bitwise-identical to the Coeus path. No tensor is
    /// constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            white_top_hat_vec(vals, dims, self.radius)
        })
    }
}

/// Black top-hat filter: BTH_B(f) = closing_B(f) - f.
/// Isolates dark structures smaller than the structuring element.
#[derive(Debug, Clone)]
pub struct BlackTopHatFilter {
    pub radius: usize,
}
impl BlackTopHatFilter {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let result = black_top_hat_vec(&vals, dims, self.radius);
        Ok(rebuild(result, dims, image))
    }

    /// Coeus-native sister of [`BlackTopHatFilter::apply`].
    ///
    /// Runs the identical `E_B(D_B(f)) - f` clamped subtraction via the shared
    /// `black_top_hat_vec` host core on the image's contiguous host buffer, so
    /// the result is bitwise-identical to the Coeus path. No tensor is
    /// constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            black_top_hat_vec(vals, dims, self.radius)
        })
    }
}

/// Substrate-agnostic host core for [`WhiteTopHatFilter`].
///
/// `WTH_B(f) = max(f - D_B(E_B(f)), 0)`. The opening here is the naive
/// erode→dilate pair (no safe-border padding), matching the historical Coeus
/// path. Non-negative for all inputs (opening is anti-extensive).
pub(crate) fn white_top_hat_vec(vals: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let eroded = erode_3d(vals, dims, radius);
    let opened = dilate_3d(&eroded, dims, radius);
    sub_clamp_vec(vals, &opened)
}

/// Substrate-agnostic host core for [`BlackTopHatFilter`].
///
/// `BTH_B(f) = max(E_B(D_B(f)) - f, 0)`. The closing here is the naive
/// dilate→erode pair (no safe-border padding), matching the historical Coeus
/// path. Non-negative for all inputs (closing is extensive).
pub(crate) fn black_top_hat_vec(vals: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let dilated = dilate_3d(vals, dims, radius);
    let closed = erode_3d(&dilated, dims, radius);
    sub_clamp_vec(&closed, vals)
}

/// Elementwise clamped difference `max(a - b, 0)`.
fn sub_clamp_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).max(0.0))
        .collect()
}

#[cfg(test)]
#[path = "tests_top_hat.rs"]
mod tests_top_hat;
