use super::ThresholdMaximumConnectedComponentsFilter;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

fn textured(dims: [usize; 3]) -> Vec<f32> {
    let n: usize = dims.iter().product();
    (0..n)
        .map(|i| ((i as f32 * 0.7).sin() * 80.0 + 100.0).round())
        .collect()
}

/// The output is binary, taking only the inside/outside values.
#[test]
fn tmcc_output_is_binary() {
    let dims = [1usize, 8, 8];
    let img = make(textured(dims), dims);
    let out = ThresholdMaximumConnectedComponentsFilter::default().apply(&img);
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|&v| v == 1.0 || v == 0.0),
        "output must be the inside/outside values"
    );
}

/// Custom inside/outside values are honoured.
#[test]
fn tmcc_custom_label_values() {
    let dims = [1usize, 6, 6];
    let img = make(textured(dims), dims);
    let f = ThresholdMaximumConnectedComponentsFilter {
        inside_value: 5.0,
        outside_value: 2.0,
        ..Default::default()
    };
    let out = f.apply(&img);
    let (ov, _) = extract_vec_infallible(&out);
    assert!(ov.iter().all(|&v| v == 5.0 || v == 2.0));
}

/// The filter is deterministic and preserves geometry.
#[test]
fn tmcc_deterministic_and_preserves_geometry() {
    let dims = [1usize, 7, 9];
    let img = make(textured(dims), dims);
    let f = ThresholdMaximumConnectedComponentsFilter::new(3);
    let (a, _) = extract_vec_infallible(&f.apply(&img));
    let (b, _) = extract_vec_infallible(&f.apply(&img));
    assert_eq!(a, b, "filter must be deterministic");
    let out = f.apply(&img);
    assert_eq!(out.shape(), dims);
    assert_eq!(out.spacing()[0], img.spacing()[0]);
}
