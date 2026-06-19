use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;

type B = NdArray<f32>;

fn img(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

fn vals(image: &Image<B, 3>) -> Vec<f32> {
    image.data_slice().into_owned()
}

/// An isolated bright peak of height 10 over flat background, h = 3.
/// Reconstruction lowers the peak by exactly h (10 → 7); the background
/// returns to the mask level (0). Hand-computed geodesic dilation.
#[test]
fn hmaxima_lowers_isolated_peak_by_h() {
    let f = img(vec![0.0, 0.0, 10.0, 0.0, 0.0], [1, 1, 5]);
    let out = HMaximaFilter::new(3.0).apply(&f).unwrap();
    let expected = [0.0f32, 0.0, 7.0, 0.0, 0.0];
    for (got, exp) in vals(&out).iter().zip(expected) {
        assert!(
            (got - exp).abs() < 1e-5,
            "hmaxima: got {got}, expected {exp}"
        );
    }
}

/// h = 0 ⇒ marker = f ⇒ reconstruction is the identity (bit-exact).
#[test]
fn hmaxima_h_zero_is_identity() {
    let f = img(vec![1.0, 5.0, 2.0, 8.0, 3.0], [1, 1, 5]);
    let out = HMaximaFilter::new(0.0).apply(&f).unwrap();
    assert_eq!(vals(&out), vals(&f), "HMaxima(f, 0) must equal f");
}

/// H-convex extracts the suppressed bright dynamic: f − HMAX_h(f) = h on the
/// peak, 0 elsewhere.
#[test]
fn hconvex_extracts_peak_dynamic() {
    let f = img(vec![0.0, 0.0, 10.0, 0.0, 0.0], [1, 1, 5]);
    let out = HConvexFilter::new(3.0).apply(&f).unwrap();
    let expected = [0.0f32, 0.0, 3.0, 0.0, 0.0];
    for (got, exp) in vals(&out).iter().zip(expected) {
        assert!(
            (got - exp).abs() < 1e-5,
            "hconvex: got {got}, expected {exp}"
        );
    }
}

/// An isolated dark pit (0) in a bright plateau (10), h = 3. Reconstruction by
/// erosion raises the pit by exactly h (0 → 3); the plateau is unchanged.
#[test]
fn hminima_raises_isolated_pit_by_h() {
    let f = img(vec![10.0, 10.0, 0.0, 10.0, 10.0], [1, 1, 5]);
    let out = HMinimaFilter::new(3.0).apply(&f).unwrap();
    let expected = [10.0f32, 10.0, 3.0, 10.0, 10.0];
    for (got, exp) in vals(&out).iter().zip(expected) {
        assert!(
            (got - exp).abs() < 1e-5,
            "hminima: got {got}, expected {exp}"
        );
    }
}

/// H-concave extracts the suppressed dark dynamic: HMIN_h(f) − f = h in the pit.
#[test]
fn hconcave_extracts_pit_dynamic() {
    let f = img(vec![10.0, 10.0, 0.0, 10.0, 10.0], [1, 1, 5]);
    let out = HConcaveFilter::new(3.0).apply(&f).unwrap();
    let expected = [0.0f32, 0.0, 3.0, 0.0, 0.0];
    for (got, exp) in vals(&out).iter().zip(expected) {
        assert!(
            (got - exp).abs() < 1e-5,
            "hconcave: got {got}, expected {exp}"
        );
    }
}

/// Structural invariants: HMAX ≤ f everywhere, HMIN ≥ f everywhere.
#[test]
fn h_extrema_respect_ordering_invariants() {
    let f = img(vec![2.0, 7.0, 1.0, 9.0, 4.0, 6.0], [1, 1, 6]);
    let hmax = vals(&HMaximaFilter::new(2.5).apply(&f).unwrap());
    let hmin = vals(&HMinimaFilter::new(2.5).apply(&f).unwrap());
    for ((&a, &lo), &hi) in vals(&f).iter().zip(hmax.iter()).zip(hmin.iter()) {
        assert!(lo <= a + 1e-6, "HMaxima must be ≤ input: {lo} > {a}");
        assert!(hi >= a - 1e-6, "HMinima must be ≥ input: {hi} < {a}");
    }
}
