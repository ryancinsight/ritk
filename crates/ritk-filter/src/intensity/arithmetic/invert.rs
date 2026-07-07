use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec_infallible as extract_vec, rebuild};

/// Intensity inversion filter.
///
/// # Mathematical Specification
///
/// `out(x) = maximum - in(x)`
///
/// where `maximum` is either user-specified or derived from the image:
///   `maximum = max_{x} in(x)` (ITK default).
///
/// The mapping is an affine reflection of the intensity range about its midpoint.
/// The input's maximum voxel maps to `0.0`; the input's minimum voxel maps to
/// `maximum - min(in)`.
///
/// # Properties
/// - `InvertIntensity(InvertIntensity(I, M), M) = I` (involution when M is fixed).
/// - Constant image with value `c` maps to all-zero output (using auto maximum).
///
/// # References
/// - ITK `itk::InvertIntensityImageFilter<TImage>`.
/// - `SimpleITK::InvertIntensity(image, maximum)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct InvertIntensityFilter {
    /// Fixed inversion maximum.  When `None`, the image maximum is used.
    maximum: Option<f32>,
}

impl InvertIntensityFilter {
    /// Construct with automatic maximum (derived from the input image).
    pub fn new() -> Self {
        Self { maximum: None }
    }

    /// Construct with a fixed maximum value.
    ///
    /// `maximum` must be finite; the result is undefined if `maximum` is NaN or
    /// infinite (no error is raised; values saturate silently).
    pub fn with_maximum(maximum: f32) -> Self {
        Self {
            maximum: Some(maximum),
        }
    }

    /// Apply intensity inversion to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let max_val = self
            .maximum
            .unwrap_or_else(|| vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        let out: Vec<f32> = vals.into_iter().map(|v| max_val - v).collect();
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
#[path = "tests_invert.rs"]
mod tests;
