//! Tests for [`PercentileFilter`] and [`RankFilter`].

use super::*;

use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_morphology::StructuringElement;
use ritk_tensor_ops::extract_vec;
use std::borrow::Cow;

type B = coeus_core::SequentialBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, shape)
}

fn to_vec(image: &Image<f32, B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec(image).expect("infallible: validated precondition");
    v
}

// ── PercentileFilter ─────────────────────────────────────────────────────

#[test]
fn percentile_zero_is_minimum() {
    let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
    let img = make_image(data, [3, 3, 3]);
    let filter = PercentileFilter::new(0.0, 1);
    let out = filter
        .apply(&img)
        .expect("infallible: validated precondition");
    let out_vec = to_vec(out.as_ref());
    for &v in &out_vec {
        assert!(v >= 1.0, "percentile=0.0 must produce the minimum, got {v}");
    }
}

#[test]
fn percentile_hundred_is_maximum() {
    let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
    let img = make_image(data, [3, 3, 3]);
    let filter = PercentileFilter::new(100.0, 1);
    let out = filter
        .apply(&img)
        .expect("infallible: validated precondition");
    let out_vec = to_vec(out.as_ref());
    for &v in &out_vec {
        assert!(
            v <= 27.0,
            "percentile=100.0 must produce the maximum, got {v}"
        );
    }
}

#[test]
fn percentile_of_constant_image_is_constant() {
    let img = make_image(vec![5.0_f32; 27], [3, 3, 3]);
    for p in [0.0_f32, 25.0, 50.0, 75.0, 100.0] {
        let filter = PercentileFilter::new(p, 1);
        let out = filter
            .apply(&img)
            .expect("infallible: validated precondition");
        let out_vec = to_vec(out.as_ref());
        for &v in &out_vec {
            assert!(
                (v - 5.0).abs() < 1e-6,
                "percentile {p} of constant 5.0 must be 5.0"
            );
        }
    }
}

#[test]
fn percentile_radius_zero_is_cow_borrowed() {
    let img = make_image(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
    let filter = PercentileFilter::new(50.0, 0);
    let out = filter
        .apply(&img)
        .expect("infallible: validated precondition");
    assert!(
        matches!(out, Cow::Borrowed(_)),
        "radius=0 must return Cow::Borrowed"
    );
}

#[test]
fn percentile_out_of_range_returns_err() {
    let img = make_image(vec![0.0_f32; 27], [3, 3, 3]);
    let filter = PercentileFilter::new(150.0, 1);
    let err = filter.apply(&img).unwrap_err();
    assert!(err.to_string().contains("percentile"));
}

#[test]
fn percentile_nan_rejected() {
    let img = make_image(vec![0.0_f32; 27], [3, 3, 3]);
    let filter = PercentileFilter::new(f32::NAN, 1);
    assert!(filter.apply(&img).is_err());
}

#[test]
fn percentile_cross_matches_cube_subset() {
    let mut data = vec![0.0_f32; 125];
    data[3 * 25 + 2 * 5 + 2] = 100.0;
    let img = make_image(data, [5, 5, 5]);
    let filter = PercentileFilter::cross(100.0, 1);
    let out = to_vec(
        filter
            .apply(&img)
            .expect("infallible: validated precondition")
            .as_ref(),
    );
    assert!((out[3 * 25 + 2 * 5 + 2] - 100.0).abs() < 1e-6);
    assert!((out[3 * 25 + 2 * 5 + 3] - 100.0).abs() < 1e-6);
}

#[test]
fn percentile_50_central_voxel_value() {
    let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
    let img = make_image(data, [3, 3, 3]);
    let filter = PercentileFilter::new(50.0, 1);
    let out = to_vec(
        filter
            .apply(&img)
            .expect("infallible: validated precondition")
            .as_ref(),
    );
    assert!((out[13] - 14.0).abs() < 1e-6);
}

// ── RankFilter ─────────────────────────────────────────────────────────

#[test]
fn rank_zero_is_minimum() {
    let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
    let img = make_image(data, [3, 3, 3]);
    let filter = RankFilter::new(0, 1);
    let out = to_vec(
        filter
            .apply(&img)
            .expect("infallible: validated precondition")
            .as_ref(),
    );
    for &v in &out {
        assert!(v >= 1.0, "rank=0 must be the minimum");
    }
}

#[test]
fn rank_last_is_maximum() {
    let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
    let img = make_image(data, [3, 3, 3]);
    let se = StructuringElement::cube(1);
    let last = se.len() - 1;
    let filter = RankFilter::with_structuring_element(last, se);
    let out = to_vec(
        filter
            .apply(&img)
            .expect("infallible: validated precondition")
            .as_ref(),
    );
    for &v in &out {
        assert!(v <= 27.0, "rank=|B|-1 must be the maximum");
    }
}

#[test]
fn rank_median_of_constant_image_is_constant() {
    let img = make_image(vec![7.0_f32; 27], [3, 3, 3]);
    let se = StructuringElement::cube(1);
    let filter = RankFilter::with_structuring_element(13, se);
    let out = to_vec(
        filter
            .apply(&img)
            .expect("infallible: validated precondition")
            .as_ref(),
    );
    for &v in &out {
        assert!((v - 7.0).abs() < 1e-6);
    }
}

#[test]
fn rank_out_of_range_returns_err() {
    let img = make_image(vec![0.0_f32; 27], [3, 3, 3]);
    let filter = RankFilter::new(50, 1);
    assert!(filter.apply(&img).is_err());
}

#[test]
fn rank_radius_zero_is_cow_borrowed() {
    let img = make_image(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
    let filter = RankFilter::new(0, 0);
    let out = filter
        .apply(&img)
        .expect("infallible: validated precondition");
    assert!(matches!(out, Cow::Borrowed(_)));
}

#[test]
fn rank_ball_se_succeeds() {
    let data: Vec<f32> = (1..=27).map(|i| i as f32).collect();
    let img = make_image(data, [3, 3, 3]);
    let filter = RankFilter::ball(0, 1);
    let out = to_vec(
        filter
            .apply(&img)
            .expect("infallible: validated precondition")
            .as_ref(),
    );
    for &v in &out {
        assert!(
            (1.0..=27.0).contains(&v),
            "ball rank=0 must produce input value, got {v}"
        );
    }
}
