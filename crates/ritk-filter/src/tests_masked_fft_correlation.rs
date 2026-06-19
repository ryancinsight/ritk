use super::MaskedFftNormalizedCorrelationFilter;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn img(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// Auto-correlation: a fully-overlapping match of an image with itself peaks at
/// 1.0, and the output extent is `fixed + moving − 1`. Values stay in [−1, 1].
#[test]
fn masked_fft_ncc_self_correlation_peaks_at_one() {
    let pat = vec![1.0, 2.0, 0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 1.0];
    let dims = [1, 3, 3];
    let mask = img(vec![1.0; 9], dims);
    let out = MaskedFftNormalizedCorrelationFilter::default()
        .apply(
            &img(pat.clone(), dims),
            &img(pat, dims),
            &mask,
            &img(vec![1.0; 9], dims),
        )
        .unwrap();
    assert_eq!(out.shape(), [1, 5, 5], "output extent is fixed+moving-1");
    let (v, _) = extract_vec_infallible(&out);
    let max = v.iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        (max - 1.0).abs() < 1e-3,
        "self-correlation peak = {max}, want 1"
    );
    assert!(
        v.iter().all(|&x| (-1.0..=1.0).contains(&x)),
        "values in [-1, 1]"
    );
}

/// A shape mismatch between an image and its mask is rejected.
#[test]
fn masked_fft_ncc_shape_mismatch_errors() {
    let fixed = img(vec![0.0; 9], [1, 3, 3]);
    let bad_mask = img(vec![1.0; 6], [1, 2, 3]);
    let moving = img(vec![0.0; 9], [1, 3, 3]);
    let mask = img(vec![1.0; 9], [1, 3, 3]);
    let f = MaskedFftNormalizedCorrelationFilter::default();
    assert!(f.apply(&fixed, &moving, &bad_mask, &mask).is_err());
}
