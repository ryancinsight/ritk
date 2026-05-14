use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    type B = burn_ndarray::NdArray<f32>;

    fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let dev = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(shape)), &dev);
        Image::new(
            t,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }
    fn make_image_with_spacing(
        vals: Vec<f32>,
        shape: [usize; 3],
        spacing: [f64; 3],
    ) -> Image<B, 3> {
        let dev = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(shape)), &dev);
        Image::new(
            t,
            Point::new([0.0; 3]),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }
    fn vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    #[test]
    fn test_uniform_image_is_unchanged_by_gaussian() {
        let img = make_image(vec![7.0_f32; 125], [5, 5, 5]);
        let out = DiscreteGaussianFilter::<B>::new(vec![1.0]).apply(&img);
        for &x in &vals(&out) {
            assert!((x - 7.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_output_shape_matches_input() {
        let img = make_image(vec![1.0_f32; 216], [6, 6, 6]);
        let out = DiscreteGaussianFilter::<B>::new(vec![2.0]).apply(&img);
        assert_eq!(out.shape(), img.shape());
    }

    #[test]
    fn test_larger_variance_produces_more_smoothing_on_step_edge() {
        let mut v: Vec<f32> = vec![0.0; 8];
        v.extend(vec![100.0; 8]);
        let img = make_image(v, [1, 1, 16]);
        let sv = vals(
            &DiscreteGaussianFilter::<B>::new(vec![0.5, 0.5, 0.5])
                .with_use_image_spacing(false)
                .apply(&img),
        );
        let lv = vals(
            &DiscreteGaussianFilter::<B>::new(vec![0.5, 0.5, 4.0])
                .with_use_image_spacing(false)
                .apply(&img),
        );
        assert!((50.0 - lv[8]).abs() < (50.0 - sv[8]).abs());
    }

    #[test]
    fn test_use_image_spacing_accounts_for_spacing() {
        let mut v: Vec<f32> = vec![0.0; 8];
        v.extend(vec![100.0; 8]);
        let img_a = make_image_with_spacing(v.clone(), [1, 1, 16], [1.0, 1.0, 1.0]);
        let img_b = make_image_with_spacing(v.clone(), [1, 1, 16], [1.0, 1.0, 2.0]);
        let f = DiscreteGaussianFilter::<B>::new(vec![4.0]);
        let a8 = vals(&f.apply(&img_a))[8];
        let b8 = vals(&f.apply(&img_b))[8];
        assert!((100.0 - a8).abs() > (100.0 - b8).abs());
    }

    #[test]
    fn test_zero_variance_produces_identity() {
        let v: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let img = make_image(v.clone(), [3, 3, 3]);
        let out = DiscreteGaussianFilter::<B>::new(vec![0.0])
            .with_use_image_spacing(false)
            .apply(&img);
        for (&e, &a) in v.iter().zip(vals(&out).iter()) {
            assert!((e - a).abs() < 1e-4);
        }
    }

    #[test]
    fn test_spatial_metadata_preserved() {
        let dev: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let t = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1.0_f32; 27], Shape::new([3, 3, 3])),
            &dev,
        );
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 1.0, 2.0]);
        let dir = Direction::identity();
        let img = Image::new(t, origin, spacing, dir);
        let out = DiscreteGaussianFilter::<B>::new(vec![1.0]).apply(&img);
        assert_eq!(out.origin(), &origin);
        assert_eq!(out.spacing(), &spacing);
        assert_eq!(out.direction(), &dir);
    }

    #[test]
    fn test_maximum_error_smaller_produces_larger_kernel() {
        let mut v: Vec<f32> = vec![0.0; 8];
        v.extend(vec![100.0; 8]);
        let img = make_image(v, [1, 1, 16]);
        let loose = vals(
            &DiscreteGaussianFilter::<B>::new(vec![0.0, 0.0, 4.0])
                .with_maximum_error(0.1)
                .with_use_image_spacing(false)
                .apply(&img),
        );
        let strict = vals(
            &DiscreteGaussianFilter::<B>::new(vec![0.0, 0.0, 4.0])
                .with_maximum_error(0.001)
                .with_use_image_spacing(false)
                .apply(&img),
        );
        assert!((50.0 - strict[8]).abs() <= (50.0 - loose[8]).abs() + 1.0);
    }

    #[test]
    fn test_per_dimension_variance_applied_independently() {
        let mut v = vec![0.0_f32; 64];
        v[4 * 8 + 4] = 100.0;
        let img = make_image(v, [1, 8, 8]);
        let ov = vals(
            &DiscreteGaussianFilter::<B>::new(vec![0.0, 0.0, 4.0])
                .with_use_image_spacing(false)
                .apply(&img),
        );
        assert!(ov[4 * 8 + 3] > 1.0);
        assert!(ov[4 * 8 + 5] > 1.0);
        assert!(ov[3 * 8 + 4] < 1.0);
        assert!(ov[5 * 8 + 4] < 1.0);
    }

    #[test]
    fn test_impulse_response_matches_analytical_gaussian() {
        // sigma=sqrt(4)=2; impulse at 15 in 1x1x31. Tol 1e-3.
        let n = 31usize;
        let c = 15usize;
        let var = 4.0f64;
        let mut imp = vec![0.0_f32; n];
        imp[c] = 1.0;
        let img = make_image(imp, [1, 1, n]);
        let ov = vals(
            &DiscreteGaussianFilter::<B>::new(vec![0.0, 0.0, var])
                .with_use_image_spacing(false)
                .apply(&img),
        );
        let tv = 2.0 * var;
        let raw: Vec<f64> = (0..n)
            .map(|k| (-((k as f64 - c as f64).powi(2)) / tv).exp())
            .collect();
        let z: f64 = raw.iter().sum();
        let wb: Vec<f64> = raw.iter().map(|&w| w / z).collect();
        for k in 0..n {
            assert!((ov[k] as f64 - wb[k]).abs() < 1e-3);
        }
    }

    #[test]
    #[should_panic]
    fn test_empty_variance_panics() {
        let _ = DiscreteGaussianFilter::<B>::new(vec![]);
    }
    #[test]
    #[should_panic]
    fn test_maximum_error_zero_panics() {
        let _ = DiscreteGaussianFilter::<B>::new(vec![1.0]).with_maximum_error(0.0);
    }
    #[test]
    #[should_panic]
    fn test_maximum_error_one_panics() {
        let _ = DiscreteGaussianFilter::<B>::new(vec![1.0]).with_maximum_error(1.0);
    }