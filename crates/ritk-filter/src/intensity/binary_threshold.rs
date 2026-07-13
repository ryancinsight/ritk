//! Binary threshold indicator filter.
//!
//! # Mathematical Specification
//!
//! output(x) = if lower_threshold <= I(x) <= upper_threshold { foreground } else { background }
//!
//! This is the binary indicator function B(x) = 1[I(x) in [lo, hi]]
//! scaled to {foreground, background}.

use crate::morphology::types::ForegroundValue;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Maps pixels inside [lower_threshold, upper_threshold] to foreground, others to background.
#[derive(Debug, Clone)]
pub struct BinaryThresholdImageFilter {
    /// Lower bound (inclusive) of the foreground interval.
    pub lower_threshold: f32,
    /// Upper bound (inclusive) of the foreground interval.
    pub upper_threshold: f32,
    /// Output value for pixels inside [lower_threshold, upper_threshold].
    pub foreground: ForegroundValue,
    /// Output value for pixels outside [lower_threshold, upper_threshold].
    pub background: f32,
}

impl BinaryThresholdImageFilter {
    pub fn new(
        lower_threshold: f32,
        upper_threshold: f32,
        foreground: impl Into<ForegroundValue>,
        background: f32,
    ) -> Self {
        Self {
            lower_threshold,
            upper_threshold,
            foreground: foreground.into(),
            background,
        }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out = binary_threshold_vec(
            &vals,
            self.lower_threshold,
            self.upper_threshold,
            f32::from(self.foreground),
            self.background,
        );
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`BinaryThresholdImageFilter::apply`].
    ///
    /// Runs the identical binary-indicator map via the shared
    /// [`binary_threshold_vec`] host core on the image's contiguous host buffer,
    /// so the result is bitwise-identical to the Burn path. No Burn tensor is
    /// constructed. Spatial metadata (origin, spacing, direction) is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let fg = f32::from(self.foreground);
        crate::native_support::map_flat_image(image, backend, |vals, _dims| {
            binary_threshold_vec(
                vals,
                self.lower_threshold,
                self.upper_threshold,
                fg,
                self.background,
            )
        })
    }
}

/// Substrate-agnostic host core for [`BinaryThresholdImageFilter`].
///
/// Maps voxels in the inclusive interval `[lower, upper]` to `foreground` and
/// all others to `background` — the indicator function `1[v ∈ [lo, hi]]`.
pub(crate) fn binary_threshold_vec(
    vals: &[f32],
    lower: f32,
    upper: f32,
    foreground: f32,
    background: f32,
) -> Vec<f32> {
    vals.iter()
        .map(|&v| {
            if v >= lower && v <= upper {
                foreground
            } else {
                background
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>) -> Image<B, 3> {
        let n = vals.len();
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new([1, 1, n]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn get_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data_slice().into_owned()
    }

    #[test]
    fn test_inside_range_is_foreground() {
        let img = make_image(vec![3.0, 4.0, 5.0, 6.0, 7.0]);
        let f = BinaryThresholdImageFilter::new(3.0, 7.0, 1.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for &v in &result {
            assert_eq!(v, 1.0, "all values inside [3,7] -> foreground=1.0");
        }
    }

    #[test]
    fn test_outside_range_is_background() {
        let img = make_image(vec![0.0, 1.0, 2.0, 8.0, 9.0]);
        let f = BinaryThresholdImageFilter::new(3.0, 7.0, 1.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for &v in &result {
            assert_eq!(v, 0.0, "all values outside [3,7] -> background=0.0");
        }
    }

    #[test]
    fn test_boundary_lower_is_foreground() {
        let img = make_image(vec![3.0]); // exactly lower_threshold
        let f = BinaryThresholdImageFilter::new(3.0, 7.0, 1.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        assert_eq!(result[0], 1.0, "lower_threshold boundary -> foreground");
    }

    #[test]
    fn test_boundary_upper_is_foreground() {
        let img = make_image(vec![7.0]); // exactly upper_threshold
        let f = BinaryThresholdImageFilter::new(3.0, 7.0, 1.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        assert_eq!(result[0], 1.0, "upper_threshold boundary -> foreground");
    }

    #[test]
    fn test_mixed_image_individual_values() {
        // values: [1, 5, 10, 3, 7, 2]  threshold: [3, 7]
        let img = make_image(vec![1.0, 5.0, 10.0, 3.0, 7.0, 2.0]);
        let f = BinaryThresholdImageFilter::new(3.0, 7.0, 1.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        // 1.0 -> bg, 5.0 -> fg, 10.0 -> bg, 3.0 -> fg, 7.0 -> fg, 2.0 -> bg
        assert_eq!(result[0], 0.0, "1.0 -> background");
        assert_eq!(result[1], 1.0, "5.0 -> foreground");
        assert_eq!(result[2], 0.0, "10.0 -> background");
        assert_eq!(result[3], 1.0, "3.0 -> foreground (boundary lower)");
        assert_eq!(result[4], 1.0, "7.0 -> foreground (boundary upper)");
        assert_eq!(result[5], 0.0, "2.0 -> background");
    }
}
