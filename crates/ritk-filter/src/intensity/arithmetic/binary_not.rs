use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible as extract_vec, rebuild};

/// Binary (two-valued) logical-NOT filter.
///
/// # Mathematical Specification
///
/// `out(x) = background  if in(x) == foreground`
/// `out(x) = foreground  otherwise`
///
/// Unlike [`super::unary::Not`] (which treats any nonzero pixel as "true" and
/// emits the `0`/`1` NumericTraits values), `BinaryNot` is parameterised by the
/// label pair `(foreground, background)` and flips them: a pixel equal to the
/// foreground label becomes background, and every other pixel becomes
/// foreground.  The two filters agree only for a `{0, 1}` mask with the default
/// labels.
///
/// # Defaults
/// `foreground = 1.0`, `background = 0.0` (matching ITK).
///
/// # References
/// - ITK `itk::BinaryNotImageFilter<TImage>`.
/// - `SimpleITK::BinaryNot(image, foregroundValue, backgroundValue)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryNotImageFilter {
    foreground: f32,
    background: f32,
}

impl Default for BinaryNotImageFilter {
    fn default() -> Self {
        Self {
            foreground: 1.0,
            background: 0.0,
        }
    }
}

impl BinaryNotImageFilter {
    /// Construct with the default ITK labels (`foreground = 1`, `background = 0`).
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with explicit foreground/background labels.
    pub fn with_labels(foreground: f32, background: f32) -> Self {
        Self {
            foreground,
            background,
        }
    }

    /// Apply the binary NOT pixelwise.  Works for any spatial dimensionality
    /// `D`; spatial metadata is preserved identically.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        let (vals, dims) = extract_vec(image);
        let (fg, bg) = (self.foreground, self.background);
        let out: Vec<f32> = vals
            .into_iter()
            .map(|v| if v == fg { bg } else { fg })
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
        let (fg, bg) = (self.foreground, self.background);
        let out: Vec<f32> = vals
            .into_iter()
            .map(|v| if v == fg { bg } else { fg })
            .collect();
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_binary_not.rs"]
mod tests;
