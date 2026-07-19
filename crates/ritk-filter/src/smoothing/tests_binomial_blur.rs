use super::*;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

/// rep=1 on an interior impulse: `[¼,½,¼]·4 = [1,2,1]` (mass conserved).
#[test]
fn rep1_interior_impulse_is_kernel() {
    let mut v = vec![0.0f32; 7];
    v[3] = 4.0;
    let img = ts::make_image::<f32, B, 3>(v, [1, 1, 7]);
    let out = BinomialBlurImageFilter::new(1).apply(&img);
    assert_eq!(
        out.data_slice()
            .expect("invariant: contiguous host storage")
            .to_vec(),
        vec![0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0]
    );
}

/// rep=2: `[¼,½,¼]` applied twice = `[1,4,6,4,1]/16 · 4`.
#[test]
fn rep2_interior_impulse() {
    let mut v = vec![0.0f32; 7];
    v[3] = 4.0;
    let img = ts::make_image::<f32, B, 3>(v, [1, 1, 7]);
    let out = BinomialBlurImageFilter::new(2).apply(&img);
    assert_eq!(
        out.data_slice()
            .expect("invariant: contiguous host storage")
            .to_vec(),
        vec![0.0, 0.25, 1.0, 1.5, 1.0, 0.25, 0.0]
    );
}

/// Reflect boundary: an edge impulse `[4,0,0,0,0]` → `[2,1,0,0,0]`
/// (reflect, not zero-flux which would give 3 at the edge).
#[test]
fn reflect_boundary_edge_impulse() {
    let img = ts::make_image::<f32, B, 3>(vec![4.0, 0.0, 0.0, 0.0, 0.0], [1, 1, 5]);
    let out = BinomialBlurImageFilter::new(1).apply(&img);
    assert_eq!(
        out.data_slice()
            .expect("invariant: contiguous host storage")
            .to_vec(),
        vec![2.0, 1.0, 0.0, 0.0, 0.0]
    );
}

/// Low-end reflect: impulse at index 1 → `[2,2,1,0,0]` (disambiguates reflect
/// from zero-flux, which would give `[1,2,1,0,0]`).
#[test]
fn reflect_boundary_near_edge_impulse() {
    let img = ts::make_image::<f32, B, 3>(vec![0.0, 4.0, 0.0, 0.0, 0.0], [1, 1, 5]);
    let out = BinomialBlurImageFilter::new(1).apply(&img);
    assert_eq!(
        out.data_slice()
            .expect("invariant: contiguous host storage")
            .to_vec(),
        vec![2.0, 2.0, 1.0, 0.0, 0.0]
    );
}

/// High-end clamp (ITK asymmetry): impulse at the LAST index → `[0,0,0,1,3]`
/// (out[N−1] = ¼·I_{N−2}+¾·I_{N−1}; symmetric reflect would give 2 here).
#[test]
fn clamp_boundary_high_edge_impulse() {
    let img = ts::make_image::<f32, B, 3>(vec![0.0, 0.0, 0.0, 0.0, 4.0], [1, 1, 5]);
    let out = BinomialBlurImageFilter::new(1).apply(&img);
    assert_eq!(
        out.data_slice()
            .expect("invariant: contiguous host storage")
            .to_vec(),
        vec![0.0, 0.0, 0.0, 1.0, 3.0]
    );
}

/// A constant image is preserved (reflect makes every weighted sum = c).
#[test]
fn constant_image_preserved() {
    let img = ts::make_image::<f32, B, 3>(vec![10.0; 27], [3, 3, 3]);
    let out = BinomialBlurImageFilter::new(3).apply(&img);
    for &x in out
        .data_slice()
        .expect("invariant: contiguous host storage")
        .iter()
    {
        assert!((x - 10.0).abs() < 1e-5, "got {x}");
    }
}

/// `repetitions = 0` is the identity.
#[test]
fn zero_repetitions_is_identity() {
    let data: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = ts::make_image::<f32, B, 3>(data.clone(), [3, 3, 3]);
    let out = BinomialBlurImageFilter::new(0).apply(&img);
    assert_eq!(
        out.data_slice()
            .expect("invariant: contiguous host storage")
            .to_vec(),
        data
    );
}
