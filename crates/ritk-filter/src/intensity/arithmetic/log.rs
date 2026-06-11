use ritk_core::filter::ops::{extract_vec_infallible as extract_vec, rebuild};
use ritk_core::image::Image;
use burn::tensor::backend::Backend;

/// Pixelwise natural logarithm filter.
///
/// # Mathematical Specification
///
/// `out(x) = ln(in(x))`
///
/// For `in(x) ≤ 0`, the IEEE 754 result is `-∞` (`in(x) = 0`) or `NaN`
/// (`in(x) < 0`) — matching ITK behaviour. In medical image contexts
/// with non-negative intensities, `in(x) = 0` maps to `-∞`.
///
/// # References
/// - ITK `itk::LogImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Log.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogImageFilter;

impl LogImageFilter {
    /// Construct a new `LogImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply pixelwise natural logarithm to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::ln).collect();
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_core::image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let t = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data_slice().into_owned()
    }

    /// ln(1) = 0.
    #[test]
    fn log_of_one_is_zero() {
        let img = make_image(vec![1.0, 1.0], [1, 1, 2]);
        let out = LogImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert!((v).abs() < 1e-6, "ln(1) = 0; got {v}");
        }
    }

    /// ln(e) ≈ 1.0.
    #[test]
    fn log_of_e_is_one() {
        let e = std::f32::consts::E;
        let img = make_image(vec![e], [1, 1, 1]);
        let out = LogImageFilter::new().apply(&img);
        let v = vals(&out)[0];
        assert!((v - 1.0).abs() < 1e-5, "ln(e) must be ≈ 1.0; got {v}");
    }

    /// ln(e²) ≈ 2.0 — verifies multiplicative correctness.
    ///
    /// # Derivation
    /// e² = exp(2) ≈ 7.389056 as f32. ln(e²) = 2.0 exactly in exact arithmetic;
    /// f32 round-trip introduces error < 1e-5.
    #[test]
    fn log_of_e_squared_is_two() {
        let e2 = (2.0_f32).exp(); // e² in f32
        let img = make_image(vec![e2], [1, 1, 1]);
        let out = LogImageFilter::new().apply(&img);
        let v = vals(&out)[0];
        assert!((v - 2.0).abs() < 1e-4, "ln(e²) must be ≈ 2.0; got {v}");
    }

    /// Spatial metadata is preserved.
    #[test]
    fn log_preserves_metadata() {
        let sp = Spacing::new([2.0, 2.0, 2.0]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![1.0_f32], Shape::new([1usize, 1, 1]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(t, Point::new([0.0, 0.0, 0.0]), sp, Direction::identity());
        let out = LogImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }
}
