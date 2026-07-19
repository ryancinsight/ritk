// Implementation lives in `unary::UnaryImageFilter<Log>`.
pub use super::unary::LogImageFilter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_support::{make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    /// ln(1) = 0.
    #[test]
    fn log_of_one_is_zero() {
        let img = make_native_image(vec![1.0, 1.0], [1, 1, 2]);
        let out = LogImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        for &v in native_vals(&out).iter() {
            assert!((v).abs() < 1e-6, "ln(1) = 0; got {v}");
        }
    }

    /// ln(e) ≈ 1.0.
    #[test]
    fn log_of_e_is_one() {
        let e = std::f32::consts::E;
        let img = make_native_image(vec![e], [1, 1, 1]);
        let out = LogImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        let v = native_vals(&out)[0];
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
        let img = make_native_image(vec![e2], [1, 1, 1]);
        let out = LogImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        let v = native_vals(&out)[0];
        assert!((v - 2.0).abs() < 1e-4, "ln(e²) must be ≈ 2.0; got {v}");
    }

    /// Spatial metadata is preserved.
    #[test]
    fn log_preserves_metadata() {
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};
        let sp = Spacing::new([2.0, 2.0, 2.0]);
        let img = Image::from_flat_on(
            vec![1.0_f32],
            [1, 1, 1],
            Point::new([0.0, 0.0, 0.0]),
            sp,
            Direction::identity(),
            &SequentialBackend,
        )
        .unwrap();
        let out = LogImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .unwrap();
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }
}
