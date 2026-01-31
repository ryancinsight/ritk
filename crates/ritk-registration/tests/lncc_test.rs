
#[cfg(test)]
mod tests {
    use burn::tensor::{Tensor, Shape, Distribution};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_core::spatial::{Point, Spacing, Direction};
    use ritk_core::transform::TranslationTransform;
    use ritk_registration::metric::LocalNormalizedCrossCorrelation;
    use ritk_registration::metric::Metric;

    type B = NdArray<f32>;

    fn create_test_image(shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let data = Tensor::<B, 3>::random(
            Shape::new(shape),
            Distribution::Uniform(0.0, 1.0),
            &device
        );
        let origin = Point::new([0.0, 0.0, 0.0]);
        let spacing = Spacing::new([1.0, 1.0, 1.0]);
        let direction = Direction::identity();
        
        Image::new(data, origin, spacing, direction)
    }

    #[test]
    fn test_lncc_perfect_match() {
        let fixed = create_test_image([10, 10, 10]);
        let moving = fixed.clone();
        
        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::from_floats([0.0, 0.0, 0.0], &device));
        
        // Kernel sigma 1.0 mm
        let metric = LocalNormalizedCrossCorrelation::<B>::new(1.0);
        
        let score = metric.forward(&fixed, &moving, &transform);
        let value = score.into_scalar();
        
        println!("LNCC Perfect Match Score: {}", value);
        
        // LNCC is correlation, so perfect match should be 1.0.
        // The implementation returns negative mean, so expected is -1.0.
        assert!((value + 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_lncc_offset() {
        let fixed = create_test_image([10, 10, 10]);
        let moving = fixed.clone();
        
        // Small offset should reduce correlation (increase score from -1.0)
        let device = Default::default();
        let transform = TranslationTransform::new(Tensor::from_floats([0.5, 0.0, 0.0], &device));
        
        let metric = LocalNormalizedCrossCorrelation::<B>::new(1.0);
        
        let score = metric.forward(&fixed, &moving, &transform);
        let value = score.into_scalar();
        
        println!("LNCC Offset Score: {}", value);
        
        assert!(value > -1.0);
    }
}
