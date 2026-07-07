// Implementation lives in `unary::UnaryImageFilter<Square>`.
pub use super::unary::SquareImageFilter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_support::{make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    /// [1,2,3] → [1,4,9].
    #[test]
    fn square_known_values() {
        let img = make_native_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let out = SquareImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        let v = native_vals(&out);
        assert_eq!(v, vec![1.0_f32, 4.0, 9.0], "[1,2,3]² must be [1,4,9]");
    }

    /// Zero voxel → zero output.
    #[test]
    fn square_zero_is_zero() {
        let img = make_native_image(vec![0.0, 0.0], [1, 1, 2]);
        let out = SquareImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        for &v in native_vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "0² = 0");
        }
    }

    /// Negative inputs → positive output (square of negative is positive).
    #[test]
    fn square_negative_positive_output() {
        let img = make_native_image(vec![-2.0, -3.0], [1, 1, 2]);
        let out = SquareImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        let v = native_vals(&out);
        assert_eq!(v, vec![4.0_f32, 9.0], "(-2)² = 4, (-3)² = 9");
    }

    /// Spatial metadata is preserved.
    #[test]
    fn square_preserves_metadata() {
        use ritk_image::native::Image;
        use ritk_spatial::{Direction, Point, Spacing};
        let sp = Spacing::new([3.0, 3.0, 3.0]);
        let img = Image::from_flat_on(
            vec![2.0_f32],
            [1, 1, 1],
            Point::new([0.0, 0.0, 0.0]),
            sp,
            Direction::identity(),
            &SequentialBackend,
        )
        .unwrap();
        let out = SquareImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    /// Constant [3,3,3] → [9,9,9].
    #[test]
    fn square_constant() {
        let img = make_native_image(vec![3.0, 3.0, 3.0], [1, 1, 3]);
        let out = SquareImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        for &v in native_vals(&out).iter() {
            assert_eq!(v, 9.0_f32, "3² = 9 for each voxel");
        }
    }
}
