use super::*;
use ritk_image::test_support as ts;

type B = coeus_core::SequentialBackend;

fn img(vals: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(vals, dims)
}

fn vals(image: &Image<f32, B, 3>) -> Vec<f32> {
    image.data_slice().into_owned()
}

/// A 5×5 image with a bright column connected to the top border (preserved) and
/// an enclosed bright peak in the interior (ground down to the surround). Border
/// is untouched. Hand-computed dual of fill-hole.
#[test]
fn grind_peak_removes_enclosed_peak_keeps_border_connected() {
    #[rustfmt::skip]
    let f = vec![
        10.0, 50.0, 10.0, 10.0, 10.0,
        10.0, 50.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 50.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0,
    ];
    #[rustfmt::skip]
    let expected = [
        10.0f32, 50.0, 10.0, 10.0, 10.0,
        10.0, 50.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0,
    ];
    let out = GrayscaleGrindPeakFilter::new()
        .apply(&img(f, [1, 5, 5]))
        .unwrap();
    for (i, (got, exp)) in vals(&out).iter().zip(expected).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "grind_peak[{i}]: got {got}, expected {exp}"
        );
    }
}

/// Grind-peak is anti-extensive: `g[x] ≤ f[x]` everywhere.
#[test]
fn grind_peak_is_anti_extensive() {
    #[rustfmt::skip]
    let f = vec![
        2.0, 9.0, 3.0, 1.0,
        4.0, 8.0, 7.0, 2.0,
        1.0, 6.0, 9.0, 3.0,
        5.0, 2.0, 4.0, 1.0,
    ];
    let out = vals(
        &GrayscaleGrindPeakFilter::new()
            .apply(&img(f.clone(), [1, 4, 4]))
            .unwrap(),
    );
    for (&g, &a) in out.iter().zip(f.iter()) {
        assert!(g <= a + 1e-6, "grind_peak must be ≤ input: {g} > {a}");
    }
}
