// Implementation lives in `unary::UnaryImageFilter<Sqrt>`.
pub use super::unary::SqrtImageFilter;

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

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

    /// [0,1,4,9] → [0,1,2,3].
    #[test]
    fn sqrt_perfect_squares() {
        let img = make_image(vec![0.0, 1.0, 4.0, 9.0], [1, 2, 2]);
        let out = SqrtImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(v, vec![0.0_f32, 1.0, 2.0, 3.0], "sqrt of perfect squares");
    }

    /// All-zero image → all-zero output.
    #[test]
    fn sqrt_zero_is_zero() {
        let img = make_image(vec![0.0, 0.0, 0.0], [1, 1, 3]);
        let out = SqrtImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "sqrt(0) = 0");
        }
    }

    /// Constant [4,4] → [2,2].
    #[test]
    fn sqrt_constant() {
        let img = make_image(vec![4.0, 4.0], [1, 1, 2]);
        let out = SqrtImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 2.0_f32, "sqrt(4) = 2");
        }
    }

    /// Spatial metadata is preserved.
    #[test]
    fn sqrt_preserves_metadata() {
        let sp = Spacing::new([1.0, 2.0, 4.0]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![9.0_f32], Shape::new([1usize, 1, 1]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(t, Point::new([0.0, 0.0, 0.0]), sp, Direction::identity());
        let out = SqrtImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }
}
