use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, shape)
}

fn vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// [1,1,3,3] → mean=2, sample std=√(4/3) → ±√3/2 (ITK sample-σ normalisation).
///
/// # Derivation
/// values = [1,1,3,3], N=4, mean=8/4=2.0
/// sample variance = (1+1+1+1)/(4−1) = 4/3, std = √(4/3) = 1.1547005
/// normalized = (v − 2.0) / 1.1547005 = [−0.8660254, −0.8660254, 0.8660254, 0.8660254]
#[test]
fn normalize_known_values() {
    let img = make_image(vec![1.0, 1.0, 3.0, 3.0], [1, 2, 2]);
    let out = NormalizeImageFilter::new().apply(&img);
    let v = vals(&out);
    let s = 0.866_025_4_f32; // √3 / 2
    let expected = [-s, -s, s, s];
    for (a, b) in v.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "normalize [1,1,3,3]: got {a}, expected {b}"
        );
    }
}

/// Constant image → all zero (degenerate std=0 case).
#[test]
fn normalize_constant_image_all_zero() {
    let img = make_image(vec![5.0, 5.0, 5.0], [1, 1, 3]);
    let out = NormalizeImageFilter::new().apply(&img);
    for &v in vals(&out).iter() {
        assert_eq!(v, 0.0_f32, "constant image with std=0 must map to 0");
    }
}

/// Output mean ≈ 0 and output *sample* variance ≈ 1 for non-constant input.
///
/// # Derivation
/// values = [0,2,4,6,8,10], N=6, mean=5, sample variance = Σ(v−5)²/(N−1) = 70/5 = 14.
/// Normalising by the sample σ makes Σ(out)²/(N−1) = 1 exactly (ITK convention);
/// the population variance Σ(out)²/N is then (N−1)/N = 5/6.
#[test]
fn normalize_output_mean_zero_variance_one() {
    let img = make_image(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0], [1, 2, 3]);
    let out = NormalizeImageFilter::new().apply(&img);
    let v = vals(&out);
    let n = v.len() as f64;
    let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let sample_variance: f64 = v
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    assert!(mean.abs() < 1e-5, "output mean must be ≈ 0; got {mean}");
    assert!(
        (sample_variance - 1.0).abs() < 1e-4,
        "output sample variance must be ≈ 1; got {sample_variance}"
    );
}

/// Two-element [0,2]: mean=1, sample std=√2 → ∓√2/2 (ITK sample-σ).
///
/// # Derivation
/// sample variance = (1+1)/(2−1) = 2, std = √2 = 1.4142136;
/// out = (v−1)/√2 = [−0.7071068, 0.7071068].
#[test]
fn normalize_two_element() {
    let img = make_image(vec![0.0, 2.0], [1, 1, 2]);
    let out = NormalizeImageFilter::new().apply(&img);
    let v = vals(&out);
    let s = std::f32::consts::FRAC_1_SQRT_2; // √2 / 2
    assert!(
        (v[0] + s).abs() < 1e-5,
        "normalize([0,2])[0] must be −√2/2; got {}",
        v[0]
    );
    assert!(
        (v[1] - s).abs() < 1e-5,
        "normalize([0,2])[1] must be +√2/2; got {}",
        v[1]
    );
}

/// Spatial metadata is preserved.
#[test]
fn normalize_preserves_metadata() {
    let sp = Spacing::new([0.5, 1.0, 2.0]);
    let device: burn_ndarray::NdArrayDevice = Default::default();
    let td = TensorData::new(vec![1.0_f32, 3.0], Shape::new([1usize, 1, 2]));
    let t = Tensor::<B, 3>::from_data(td, &device);
    let img = Image::new(t, Point::new([0.0, 0.0, 0.0]), sp, Direction::identity());
    let out = NormalizeImageFilter::new().apply(&img);
    assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
}
