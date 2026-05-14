use crate::filter::ops::{extract_vec_infallible as extract_vec, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Pixelwise absolute value filter.
///
/// # Mathematical Specification
///
/// `out(x) = |in(x)|`
///
/// # Properties
/// - Idempotent on non-negative images: `|f(x)| = f(x)` iff `f(x) ≥ 0`.
/// - Preserves spatial metadata.
/// - O(N) time, O(N) output space.
///
/// # References
/// - ITK `itk::AbsImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Abs.
#[derive(Debug, Clone, Copy, Default)]
pub struct AbsImageFilter;

impl AbsImageFilter {
    /// Construct a new `AbsImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the absolute-value transform to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::abs).collect();
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
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
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

/// Non-negative image: abs is identity.
    #[test]
    fn abs_nonneg_is_identity() {
        let img = make_image(vec![0.0, 1.0, 2.5, 10.0], [1, 2, 2]);
        let out = AbsImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(
            v,
            vec![0.0_f32, 1.0, 2.5, 10.0],
            "non-negative input must be unchanged by abs"
        );
    }

    /// Negative values become positive.
    #[test]
    fn abs_negates_negative_voxels() {
        let img = make_image(vec![-3.0, -1.0, 0.0, 2.0], [1, 2, 2]);
        let out = AbsImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(
            v,
            vec![3.0_f32, 1.0, 0.0, 2.0],
            "abs must negate each negative voxel: [-3,-1,0,2] → [3,1,0,2]"
        );
    }

    /// All-negative image: every output is the negation of input.
    #[test]
    fn abs_all_negative_all_positive() {
        let img = make_image(vec![-5.0, -2.0, -0.5], [1, 1, 3]);
        let out = AbsImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert!(v >= 0.0, "abs output must be non-negative; got {v}");
        }
    }

    /// Spatial metadata is preserved.
    #[test]
    fn abs_preserves_metadata() {
        let sp = Spacing::new([2.0, 3.0, 4.0]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![1.0_f32, -1.0], Shape::new([1usize, 1, 2]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = AbsImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    /// Constant positive image: unchanged.
    #[test]
    fn abs_constant_positive_unchanged() {
        let img = make_image(vec![7.0, 7.0, 7.0], [1, 1, 3]);
        let out = AbsImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 7.0_f32, "constant positive image unchanged by abs");
        }
    }
}
