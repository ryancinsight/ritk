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

// Signal with two maxima plateaus (10; 6,6), a boundary max (last 2), pits, and
// minima plateaus. Verified against sitk.RegionalMaxima / RegionalMinima.
const SIGNAL: [f32; 9] = [2.0, 2.0, 10.0, 2.0, 6.0, 6.0, 2.0, 0.0, 2.0];

#[test]
fn regional_maxima_binary_matches_reference() {
    let f = img(SIGNAL.to_vec(), [1, 1, 9]);
    let out = RegionalMaximaFilter::new().apply(&f).unwrap();
    let expected = [0.0f32, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    assert_eq!(vals(&out), expected, "regional maxima binary mask");
}

#[test]
fn regional_minima_binary_matches_reference() {
    let f = img(SIGNAL.to_vec(), [1, 1, 9]);
    let out = RegionalMinimaFilter::new().apply(&f).unwrap();
    let expected = [1.0f32, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    assert_eq!(vals(&out), expected, "regional minima binary mask");
}

#[test]
fn valued_regional_maxima_keeps_value_else_min() {
    let f = img(SIGNAL.to_vec(), [1, 1, 9]);
    let out = vals(&ValuedRegionalMaximaFilter::new().apply(&f).unwrap());
    let lo = f32::MIN;
    let expected = [lo, lo, 10.0, lo, 6.0, 6.0, lo, lo, 2.0];
    assert_eq!(out, expected, "valued regional maxima");
}

#[test]
fn valued_regional_minima_keeps_value_else_max() {
    let f = img(SIGNAL.to_vec(), [1, 1, 9]);
    let out = vals(&ValuedRegionalMinimaFilter::new().apply(&f).unwrap());
    let hi = f32::MAX;
    let expected = [2.0, 2.0, hi, 2.0, hi, hi, hi, 0.0, hi];
    assert_eq!(out, expected, "valued regional minima");
}

/// Sub-unit contrast: two maxima of 0.5 and 0.2 separated by 0.0. Both are
/// genuine regional maxima — the `f − Recon(f − 1)` shortcut would (wrongly)
/// merge them; the flat-zone flood detects both.
#[test]
fn subunit_contrast_detects_both_maxima() {
    let f = img(vec![0.0, 0.5, 0.0, 0.2, 0.0], [1, 1, 5]);
    let out = vals(&RegionalMaximaFilter::new().apply(&f).unwrap());
    assert_eq!(
        out,
        [0.0, 1.0, 0.0, 1.0, 0.0],
        "both sub-unit maxima detected"
    );
}

/// A constant image is one flat zone with no more-extreme neighbour, hence a
/// single regional maximum AND minimum (ITK flatIsExtremum semantics).
#[test]
fn constant_image_is_single_extremum() {
    let f = img(vec![3.0; 8], [2, 2, 2]);
    let max = vals(&RegionalMaximaFilter::new().apply(&f).unwrap());
    let min = vals(&RegionalMinimaFilter::new().apply(&f).unwrap());
    assert!(
        max.iter().all(|&v| v == 1.0),
        "constant ⇒ all regional maxima"
    );
    assert!(
        min.iter().all(|&v| v == 1.0),
        "constant ⇒ all regional minima"
    );
}

/// Custom foreground/background values are honoured.
#[test]
fn binary_honours_custom_values() {
    let f = img(SIGNAL.to_vec(), [1, 1, 9]);
    let out = vals(
        &RegionalMaximaFilter::new()
            .with_values(7.0, -1.0)
            .apply(&f)
            .unwrap(),
    );
    let expected = [-1.0f32, -1.0, 7.0, -1.0, 7.0, 7.0, -1.0, -1.0, 7.0];
    assert_eq!(out, expected, "custom fg/bg");
}
