use crate::filter::ops::{extract_vec_infallible as extract_vec, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Pixelwise square filter.
///
/// # Mathematical Specification
///
/// `out(x) = in(x)²`
///
/// # Properties
/// - Non-negative output for all real inputs.
/// - O(N) time.
///
/// # References
/// - ITK `itk::SquareImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Square.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquareImageFilter;

impl SquareImageFilter {
    /// Construct a new `SquareImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply pixelwise squaring to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(|v| v * v).collect();
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
        img.data_vec()
    }

    /// [1,2,3] → [1,4,9].
    #[test]
    fn square_known_values() {
        let img = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let out = SquareImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(v, vec![1.0_f32, 4.0, 9.0], "[1,2,3]² must be [1,4,9]");
    }

    /// Zero voxel → zero output.
    #[test]
    fn square_zero_is_zero() {
        let img = make_image(vec![0.0, 0.0], [1, 1, 2]);
        let out = SquareImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "0² = 0");
        }
    }

    /// Negative inputs → positive output (square of negative is positive).
    #[test]
    fn square_negative_positive_output() {
        let img = make_image(vec![-2.0, -3.0], [1, 1, 2]);
        let out = SquareImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(v, vec![4.0_f32, 9.0], "(-2)² = 4, (-3)² = 9");
    }

    /// Spatial metadata is preserved.
    #[test]
    fn square_preserves_metadata() {
        let sp = Spacing::new([3.0, 3.0, 3.0]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![2.0_f32], Shape::new([1usize, 1, 1]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(t, Point::new([0.0, 0.0, 0.0]), sp, Direction::identity());
        let out = SquareImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    /// Constant [3,3,3] → [9,9,9].
    #[test]
    fn square_constant() {
        let img = make_image(vec![3.0, 3.0, 3.0], [1, 1, 3]);
        let out = SquareImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 9.0_f32, "3² = 9 for each voxel");
        }
    }
}
