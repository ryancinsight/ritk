use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible as extract_vec, rebuild};

/// Pixelwise modulo filter — `out(x) = in(x) % dividend`.
///
/// # Mathematical Specification
///
/// Each (integral-valued) voxel is reduced modulo a positive `dividend` using
/// C/C++ truncated-toward-zero remainder semantics (matching ITK, which
/// `static_cast`s through the integer pixel type): `−7 % 3 = −1`, `−8 % 3 = −2`.
/// Rust's `%` on `i64` has the identical truncation rule.
///
/// Like ITK `ModulusImageFilter` / `sitk.Modulus`, this is defined on integer
/// images. ritk stores pixels as `f32`; each voxel is rounded to the nearest
/// integer before the reduction, so the result is exact for the integral pixel
/// values these label/scan images carry.
///
/// # References
/// - ITK `itk::ModulusImageFilter<TInputImage, TOutputImage>`.
/// - `SimpleITK::Modulus(image, dividend)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModulusImageFilter {
    dividend: i64,
}

impl ModulusImageFilter {
    /// Construct with the given (non-zero) `dividend`.
    ///
    /// # Panics
    /// Panics if `dividend == 0` (modulo by zero is undefined).
    pub fn new(dividend: i64) -> Self {
        assert!(
            dividend != 0,
            "ModulusImageFilter: dividend must be non-zero"
        );
        Self { dividend }
    }

    /// Apply the modulo pixelwise. Works for any spatial dimensionality `D`;
    /// spatial metadata is preserved identically.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        let (vals, dims) = extract_vec(image);
        let d = self.dividend;
        let out: Vec<f32> = vals
            .into_iter()
            .map(|v| ((v.round() as i64) % d) as f32)
            .collect();
        rebuild(out, dims, image)
    }
    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let d = self.dividend;
        let out: Vec<f32> = vals
            .into_iter()
            .map(|v| ((v.round() as i64) % d) as f32)
            .collect();
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_modulus.rs"]
mod tests;
