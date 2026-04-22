//! Parity harness: validates RITK core implementations against analytically-derived
//! reference values. Each expected value is derived from a closed-form formula,
//! not from empirical RITK output observation. This documents and verifies that
//! RITK produces results consistent with ITK/SimpleITK for the same inputs.

use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::{image::Image, spatial::{Direction, Point, Spacing}};

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let t = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)), &Default::default());
    Image::new(t, Point::new([0.0; 3]), Spacing::new([1.0; 3]), Direction::identity())
}

fn vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data().clone().into_data().as_slice::<f32>().unwrap().to_vec()
}

/// Formula: v_out = (v - v_min) / (v_max - v_min) * (out_max - out_min) + out_min
/// Input [0, 25, 50, 75, 100], out=[0,1] -> expected [0.0, 0.25, 0.5, 0.75, 1.0]
#[test]
fn parity_rescale_intensity_maps_full_range() {
    use ritk_core::filter::intensity::RescaleIntensityFilter;
    let img = make_image(vec![0.0, 25.0, 50.0, 75.0, 100.0], [5, 1, 1]);
    let out = RescaleIntensityFilter::new(0.0, 1.0).apply(&img).unwrap();
    let v = vals(&out);
    for (i, (&got, &exp)) in v.iter().zip([0.0f32, 0.25, 0.5, 0.75, 1.0].iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "voxel {i}: got {got}, exp {exp}");
    }
}

/// Formula: out[x] = fg if lower<=in[x]<=upper else bg
/// Input [0,50,100,150,200], lower=50, upper=150, fg=1, bg=0 -> [0,1,1,1,0]
#[test]
fn parity_binary_threshold_indicator_function() {
    use ritk_core::filter::intensity::BinaryThresholdImageFilter;
    let img = make_image(vec![0.0, 50.0, 100.0, 150.0, 200.0], [5, 1, 1]);
    let out = BinaryThresholdImageFilter::new(50.0, 150.0, 1.0, 0.0).apply(&img).unwrap();
    let v = vals(&out);
    for (i, (&got, &exp)) in v.iter().zip([0.0f32, 1.0, 1.0, 1.0, 0.0].iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "voxel {i}: got {got}, exp {exp}");
    }
}

/// Formula: out[x] = outside if in[x] < threshold else in[x]
/// Input [0,1,2,3,4], threshold=2.0, outside=-1 -> [-1,-1,2,3,4]
#[test]
fn parity_threshold_below_preserves_at_and_above() {
    use ritk_core::filter::intensity::ThresholdImageFilter;
    let img = make_image(vec![0.0, 1.0, 2.0, 3.0, 4.0], [5, 1, 1]);
    let out = ThresholdImageFilter::below(2.0, -1.0).apply(&img).unwrap();
    let v = vals(&out);
    for (i, (&got, &exp)) in v.iter().zip([-1.0f32, -1.0, 2.0, 3.0, 4.0].iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "voxel {i}: got {got}, exp {exp}");
    }
}

/// out = (max-min)/(1+exp(-(in-alpha)/beta)) + min
/// alpha=0, beta=10, min=0, max=1
/// at in=-100: out->0, at in=0: out=0.5, at in=100: out->1
#[test]
fn parity_sigmoid_midpoint_and_asymptotes() {
    use ritk_core::filter::intensity::SigmoidImageFilter;
    let img = make_image(vec![-100.0, 0.0, 100.0], [3, 1, 1]);
    let out = SigmoidImageFilter::new(0.0, 10.0, 0.0, 1.0).apply(&img).unwrap();
    let v = vals(&out);
    assert!(v[0] < 0.01, "sigmoid(-100) should be ~0, got {}", v[0]);
    assert!((v[1] - 0.5).abs() < 5e-3, "sigmoid(0) should be 0.5, got {}", v[1]);
    assert!(v[2] > 0.99, "sigmoid(100) should be ~1, got {}", v[2]);
}

/// For constant f(x)=c: gradient = 0 everywhere, so |grad f| = 0.
#[test]
fn parity_gradient_magnitude_constant_image_is_zero() {
    use ritk_core::filter::edge::GradientMagnitudeFilter;
    let img = make_image(vec![5.0f32; 27], [3, 3, 3]);
    let out = GradientMagnitudeFilter::new([1.0, 1.0, 1.0]).apply(&img).unwrap();
    for (i, &v) in vals(&out).iter().enumerate() {
        assert!(v.abs() < 1e-5, "voxel {i}: grad of constant must be 0, got {v}");
    }
}

/// For linear f(z,y,x)=x: second derivative d2f/dx2=0, so Laplacian=0 at interior.
#[test]
fn parity_laplacian_linear_ramp_is_zero_at_interior() {
    use ritk_core::filter::edge::LaplacianFilter;
    let data: Vec<f32> = (0..7).map(|i| i as f32).collect();
    let img = make_image(data, [1, 1, 7]);
    let out = LaplacianFilter::new([1.0, 1.0, 1.0]).apply(&img).unwrap();
    let v = vals(&out);
    for i in 1..6 {
        assert!(v[i].abs() < 1e-4, "interior voxel {i}: Laplacian of linear must be 0, got {}", v[i]);
    }
}

/// Z-score: out = (in - mean) / std.  Mean of output must be ~0, std must be ~1.
/// Input [2,4,4,4,5,5,7,9]: mean=5, std=2.
#[test]
fn parity_zscore_zero_mean_unit_variance() {
    use ritk_core::statistics::ZScoreNormalizer;
    let img = make_image(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], [8, 1, 1]);
    let out = ZScoreNormalizer::new().normalize(&img);
    let v = vals(&out);
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    assert!(mean.abs() < 1e-4, "z-score mean must be ~0, got {mean}");
    assert!((var.sqrt() - 1.0).abs() < 1e-4, "z-score std must be ~1, got {}", var.sqrt());
}

/// dice(A, A) = 1.0 for any non-empty binary mask A.
#[test]
fn parity_dice_perfect_overlap() {
    use ritk_core::statistics::dice_coefficient;
    let seg = make_image(vec![1.0, 0.0, 1.0, 0.0, 1.0], [5, 1, 1]);
    let d = dice_coefficient(&seg, &seg);
    assert!((d - 1.0).abs() < 1e-5, "perfect overlap must yield 1.0, got {d}");
}

/// dice(A, complement(A)) = 0.0 when A and complement are disjoint.
#[test]
fn parity_dice_zero_overlap() {
    use ritk_core::statistics::dice_coefficient;
    let a = make_image(vec![1.0, 0.0, 1.0], [3, 1, 1]);
    let b = make_image(vec![0.0, 1.0, 0.0], [3, 1, 1]);
    let d = dice_coefficient(&a, &b);
    assert!(d.abs() < 1e-5, "disjoint segments must yield 0.0, got {d}");
}

/// RescaleIntensity on constant image: degenerate case (I_min == I_max).
/// ITK clamps output to out_min when denominator is zero.
#[test]
fn parity_rescale_intensity_constant_image_in_range() {
    use ritk_core::filter::intensity::RescaleIntensityFilter;
    let img = make_image(vec![7.0f32; 8], [2, 2, 2]);
    let out = RescaleIntensityFilter::new(0.0, 1.0).apply(&img).unwrap();
    for (i, &v) in vals(&out).iter().enumerate() {
        assert!(v >= 0.0 && v <= 1.0,
            "constant image rescale must be in [0,1] at voxel {i}: {v}");
    }
}
