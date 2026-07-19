use super::AdaptiveHistogramEqualizationFilter;
use coeus_core::SequentialBackend;
use ritk_image::test_support as ts;
use ritk_image::Image as NativeImage;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn img(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// With `α = β = 0` the cumulative function reduces to `0.5·sgn(u−v)`, giving a
/// rank-based remap. For `[0, 100, 200]` over a full window (radius 2 in x), the
/// outputs are `iscale·(0.5 + mean_y 0.5·sgn(u−v)) + min`:
/// pixel 0 → 200·(0.5 − 1/3) = 33.33…, pixel 1 → 100, pixel 2 → 200·(0.5 + 1/3) = 166.67…
#[test]
fn adaptive_eq_rank_remap_alpha_beta_zero() {
    let f = AdaptiveHistogramEqualizationFilter {
        radius: [0, 0, 2],
        alpha: 0.0,
        beta: 0.0,
    };
    let out = f.apply(&img(vec![0.0, 100.0, 200.0], [1, 1, 3])).unwrap();
    let (v, _) = extract_vec_infallible(&out);
    assert!(
        (v[0] - 200.0 / 6.0).abs() < 1e-3,
        "v0 = {}, want 33.33",
        v[0]
    );
    assert!((v[1] - 100.0).abs() < 1e-3, "v1 = {}, want 100", v[1]);
    assert!(
        (v[2] - 1000.0 / 6.0).abs() < 1e-3,
        "v2 = {}, want 166.67",
        v[2]
    );
}

/// A constant image is unchanged (iscale = 0, identity).
#[test]
fn adaptive_eq_constant_is_identity() {
    let out = AdaptiveHistogramEqualizationFilter::default()
        .apply(&img(vec![7.0; 27], [3, 3, 3]))
        .unwrap();
    let (v, _) = extract_vec_infallible(&out);
    assert!(v.iter().all(|&x| x == 7.0), "constant image is unchanged");
}

/// Output geometry matches the input.
#[test]
fn adaptive_eq_preserves_geometry() {
    let dims = [2, 4, 5];
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
    let out = AdaptiveHistogramEqualizationFilter::new([1, 1, 1])
        .apply(&img(data, dims))
        .unwrap();
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], 1.0);
}

#[test]
fn native_adaptive_eq_constant_is_identity_and_preserves_metadata() {
    let image = NativeImage::from_flat_on(
        vec![7.0; 8],
        [2, 2, 2],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = AdaptiveHistogramEqualizationFilter::default()
        .apply_native(&image, &SequentialBackend)
        .expect("native adaptive equalization succeeds");

    assert_eq!(
        output.data_slice().expect("invariant: contiguous storage"),
        &[7.0; 8]
    );
    assert_eq!(output.origin(), image.origin());
    assert_eq!(output.spacing(), image.spacing());
    assert_eq!(output.direction(), image.direction());
}
