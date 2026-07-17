use super::*;
use crate::native_support::LegacyBurnBackend;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};

type B = LegacyBurnBackend;

fn img(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::burn_compat::make_image::<B, 3>(vals, dims)
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

#[test]
fn hminima_native_matches_legacy_and_preserves_geometry() {
    let values = vec![10.0, 10.0, 0.0, 10.0, 10.0];
    let legacy = img(values.clone(), [1, 1, 5]);
    let origin = Point::new([2.0, 3.0, 5.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    let native = NativeImage::from_flat_on(
        values,
        [1, 1, 5],
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
    .unwrap();
    let filter = HMinimaFilter::new(3.0);
    let expected = filter.apply(&legacy).unwrap();
    let actual = filter.apply_native(&native, &SequentialBackend).unwrap();
    assert_eq!(actual.data_slice().unwrap(), expected.data_slice().as_ref());
    assert_eq!(*actual.origin(), origin);
    assert_eq!(*actual.spacing(), spacing);
    assert_eq!(*actual.direction(), direction);
}

#[test]
fn hminima_validation_errors_are_exact() {
    let image = img(vec![0.0], [1, 1, 1]);
    for height in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        assert_eq!(
            HMinimaFilter::new(height)
                .apply(&image)
                .unwrap_err()
                .to_string(),
            format!("h-transform height must be finite and nonnegative, got {height}")
        );
    }
    let invalid = img(vec![f32::NAN], [1, 1, 1]);
    assert_eq!(
        HMinimaFilter::new(1.0)
            .apply(&invalid)
            .unwrap_err()
            .to_string(),
        "h-transform sample at flat index 0 must be finite, got NaN"
    );

    assert_eq!(
        HMinimaFilter::new(f32::MAX)
            .apply(&img(vec![f32::MAX], [1, 1, 1]))
            .unwrap_err()
            .to_string(),
        "h-transform marker at flat index 0 must remain finite after shift, got inf"
    );
    assert_eq!(
        HMaximaFilter::new(f32::MAX)
            .apply(&img(vec![-f32::MAX], [1, 1, 1]))
            .unwrap_err()
            .to_string(),
        "h-transform marker at flat index 0 must remain finite after shift, got -inf"
    );

    let native = NativeImage::from_flat_on(
        vec![f32::MAX],
        [1, 1, 1],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    assert_eq!(
        HMinimaFilter::new(f32::MAX)
            .apply_native(&native, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        "h-transform marker at flat index 0 must remain finite after shift, got inf"
    );
    assert_eq!(
        HMinimaFilter::new(-1.0)
            .apply_native(&native, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        "h-transform height must be finite and nonnegative, got -1"
    );
    let native_invalid = NativeImage::from_flat_on(
        vec![f32::NAN],
        [1, 1, 1],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    assert_eq!(
        HMinimaFilter::new(1.0)
            .apply_native(&native_invalid, &SequentialBackend)
            .unwrap_err()
            .to_string(),
        "h-transform sample at flat index 0 must be finite, got NaN"
    );
}
