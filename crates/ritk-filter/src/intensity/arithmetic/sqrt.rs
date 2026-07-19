// Implementation lives in `unary::UnaryImageFilter<Sqrt>`.
pub use super::unary::SqrtImageFilter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_support::{make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    /// [0,1,4,9] → [0,1,2,3].
    #[test]
    fn sqrt_perfect_squares() {
        let img = make_native_image(vec![0.0, 1.0, 4.0, 9.0], [1, 2, 2]);
        let out = SqrtImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        let v = native_vals(&out);
        assert_eq!(v, vec![0.0_f32, 1.0, 2.0, 3.0], "sqrt of perfect squares");
    }

    /// All-zero image → all-zero output.
    #[test]
    fn sqrt_zero_is_zero() {
        let img = make_native_image(vec![0.0, 0.0, 0.0], [1, 1, 3]);
        let out = SqrtImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        for &v in native_vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "sqrt(0) = 0");
        }
    }

    /// Constant [4,4] → [2,2].
    #[test]
    fn sqrt_constant() {
        let img = make_native_image(vec![4.0, 4.0], [1, 1, 2]);
        let out = SqrtImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        for &v in native_vals(&out).iter() {
            assert_eq!(v, 2.0_f32, "sqrt(4) = 2");
        }
    }

    /// Spatial metadata is preserved.
    #[test]
    fn sqrt_preserves_metadata() {
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};
        let sp = Spacing::new([1.0, 2.0, 4.0]);
        let img = Image::from_flat_on(
            vec![9.0_f32],
            [1, 1, 1],
            Point::new([0.0, 0.0, 0.0]),
            sp,
            Direction::identity(),
            &SequentialBackend,
        )
        .unwrap();
        let out = SqrtImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }
}
