use super::BitwiseNotImageFilter;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

fn img(data: Vec<f32>) -> Image<B, 3> {
    let n = data.len();
    ts::burn_compat::make_image::<B, 3>(data, [1, 1, n])
}

/// Unsigned 8-bit complement: `~x = 255 − x`.
#[test]
fn bitwise_not_uint8() {
    let out =
        BitwiseNotImageFilter::unsigned(8).apply(&img(vec![0.0, 1.0, 2.0, 5.0, 200.0, 255.0]));
    let (v, _) = extract_vec_infallible(&out);
    assert_eq!(v, vec![255.0, 254.0, 253.0, 250.0, 55.0, 0.0]);
}

/// Unsigned 16-bit complement: `~x = 65535 − x`.
#[test]
fn bitwise_not_uint16() {
    let out = BitwiseNotImageFilter::unsigned(16).apply(&img(vec![0.0, 1.0, 1000.0, 65535.0]));
    let (v, _) = extract_vec_infallible(&out);
    assert_eq!(v, vec![65535.0, 65534.0, 64535.0, 0.0]);
}

/// Signed two's-complement: `~x = −x − 1`.
#[test]
fn bitwise_not_signed() {
    let out = BitwiseNotImageFilter::signed().apply(&img(vec![0.0, 1.0, -1.0, 100.0, -100.0]));
    let (v, _) = extract_vec_infallible(&out);
    assert_eq!(v, vec![-1.0, -2.0, 0.0, -101.0, 99.0]);
}
