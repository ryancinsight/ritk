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

/// Opening by reconstruction removes a thin bright spike (which a radius-1 SE
/// cannot contain) while fully restoring the wide plateau's contour — unlike a
/// plain opening, which would also erode the plateau edges. Hand-computed.
#[test]
fn opening_by_reconstruction_removes_spike_keeps_plateau() {
    let f = img(
        vec![0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 10.0, 0.0, 0.0],
        [1, 1, 9],
    );
    let out = OpeningByReconstructionFilter::new(1).apply(&f).unwrap();
    let expected = [0.0f32, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0];
    for (got, exp) in vals(&out).iter().zip(expected) {
        assert!((got - exp).abs() < 1e-5, "OBR: got {got}, expected {exp}");
    }
}

/// Closing by reconstruction (dual) fills a thin dark pit while keeping the wide
/// dark plateau at its level.
#[test]
fn closing_by_reconstruction_fills_pit_keeps_plateau() {
    let f = img(
        vec![10.0, 10.0, 5.0, 5.0, 5.0, 10.0, 0.0, 10.0, 10.0],
        [1, 1, 9],
    );
    let out = ClosingByReconstructionFilter::new(1).apply(&f).unwrap();
    let expected = [10.0f32, 10.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0];
    for (got, exp) in vals(&out).iter().zip(expected) {
        assert!((got - exp).abs() < 1e-5, "CBR: got {got}, expected {exp}");
    }
}

/// Radius 0 ⇒ erosion/dilation is identity ⇒ reconstruction of f under f is f.
#[test]
fn radius_zero_is_identity() {
    let f = img(vec![1.0, 7.0, 2.0, 9.0, 3.0], [1, 1, 5]);
    let obr = OpeningByReconstructionFilter::new(0).apply(&f).unwrap();
    let cbr = ClosingByReconstructionFilter::new(0).apply(&f).unwrap();
    assert_eq!(vals(&obr), vals(&f), "OBR(f, r=0) must equal f");
    assert_eq!(vals(&cbr), vals(&f), "CBR(f, r=0) must equal f");
}

/// Opening by reconstruction is anti-extensive (≤ f); closing is extensive (≥ f).
#[test]
fn ordering_invariants() {
    let f = img(vec![2.0, 9.0, 1.0, 6.0, 3.0, 8.0, 4.0], [1, 1, 7]);
    let obr = vals(&OpeningByReconstructionFilter::new(1).apply(&f).unwrap());
    let cbr = vals(&ClosingByReconstructionFilter::new(1).apply(&f).unwrap());
    for ((&a, &o), &c) in vals(&f).iter().zip(obr.iter()).zip(cbr.iter()) {
        assert!(o <= a + 1e-6, "OBR must be ≤ input: {o} > {a}");
        assert!(c >= a - 1e-6, "CBR must be ≥ input: {c} < {a}");
    }
}
