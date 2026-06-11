//! Tests for [`super::InverseFftFilter`].
//!
//! # Mathematical invariants under test
//!
//! 1. **Shape contract**: `apply_2d([H, 2W]) -> [H, W]`;
//!    `apply_3d([D, H, 2W]) -> [D, H, W]`.
//! 2. **Zero annihilation**: `IFFT(0_complex) = 0_real` element-wise.
//! 3. **DC linearity**: for input `F` with `F[0,0] = H*W` (only DC nonzero),
//!    `IFFT_normalized(F)[x,y] = 1.0` for all `(x, y)`.
//!
//! # DC derivation (invariant 3)
//!
//! rustfft unnormalized row IFFT of `[H*W, 0, 0, 0]` (length W):
//!   `IFFT_row[c] = H*W  for all c`  (DC only -> constant along row)
//!
//! After row pass: `buf[0,c] = H*W` for all c; `buf[r>0,c] = 0`.
//!
//! Unnormalized col IFFT of `[H*W, 0, 0, 0]` (length H):
//!   `IFFT_col[r] = H*W  for all r`
//!
//! After col pass: `buf[r,c] = H*W  for all (r,c)`.
//!
//! Normalization: `out[r,c] = buf[r,c].re / (H*W) = 1.0  for all (r,c)`.

use super::InverseFftFilter;
use ritk_core::filter::ops::extract_vec;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Construct a 2-D complex image with shape `[h, w_complex]`.
///
/// `w_complex` must equal `2 * w_real`. Re and Im are interleaved in the last
/// dimension: Re at column `2c`, Im at column `2c+1`.
fn make_complex_2d(data: Vec<f32>, h: usize, w_complex: usize) -> Image<B, 2> {
    let device = Default::default();
    let td = TensorData::new(data, Shape::new([h, w_complex]));
    let tensor = Tensor::<B, 2>::from_data(td, &device);
    Image::new(
        tensor,
        Point::origin(),
        Spacing::new([1.0, 1.0]),
        Direction::identity(),
    )
}

/// Construct a 3-D complex image with shape `[depth, h, w_complex]`.
///
/// `w_complex` must equal `2 * w_real`.
fn make_complex_3d(data: Vec<f32>, depth: usize, h: usize, w_complex: usize) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(data, Shape::new([depth, h, w_complex]));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::origin(),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Shape contract: `apply_2d([4, 12])` -> `[4, 6]`.
///
/// `w_real = w_complex / 2 = 12 / 2 = 6`.
#[test]
fn output_shape_2d() {
    let data = vec![0.0_f32; 4 * 12];
    let img = make_complex_2d(data, 4, 12);
    let result = InverseFftFilter::new().apply(&img).unwrap();
    assert_eq!(
        result.shape(),
        [4, 6],
        "apply_2d([4,12]) must produce shape [4,6]"
    );
}

/// Shape contract: `apply_3d([3, 4, 12])` -> `[3, 4, 6]`.
///
/// `w_real = w_complex / 2 = 12 / 2 = 6`.
#[test]
fn output_shape_3d() {
    let data = vec![0.0_f32; 3 * 4 * 12];
    let img = make_complex_3d(data, 3, 4, 12);
    let result = InverseFftFilter::new().apply(&img).unwrap();
    assert_eq!(
        result.shape(),
        [3, 4, 6],
        "apply_3d([3,4,12]) must produce shape [3,4,6]"
    );
}

/// Zero annihilation: `IFFT(0_complex) = 0_real` element-wise.
///
/// The zero complex image maps to the zero spatial image regardless of
/// normalization: `IFFT(0)[n] = sum_k 0 * e^{...} = 0`.
#[test]
fn all_zero_complex_image_gives_zero_real() {
    // H=4, W=4; complex shape [4, 8] = [H, 2*W].
    let data = vec![0.0_f32; 4 * 8];
    let img = make_complex_2d(data, 4, 8);
    let result = InverseFftFilter::new().apply(&img).unwrap();

    assert_eq!(
        result.shape(),
        [4, 4],
        "IFFT of zero [4,8] must have output shape [4,4]"
    );

    let (vals, _) = extract_vec(&result).unwrap();
    assert_eq!(vals.len(), 16, "output must contain H*W = 4*4 = 16 pixels");
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "pixel {} must be 0.0 for all-zero complex input, got {}",
            i,
            v
        );
    }
}

/// DC-only complex image -> constant 1.0 spatial image.
///
/// For input `F` with shape `[4, 8]` (H=4, W=4):
///   `F[0,0] = 16 + 0i` (= H*W), all other coefficients zero.
///
/// Expected output: every pixel equals 1.0 (within 1e-4), per the DC derivation
/// in the module-level proof.
#[test]
fn dc_only_complex_image_2d() {
    // Complex image shape [4, 8] = [H, 2*W] with H=4, W=4.
    // data[0] = Re(F[0,0]) = 16.0  (= H*W)
    // data[1] = Im(F[0,0]) = 0.0
    // data[2..32] = 0.0  (all other coefficients)
    let mut data = vec![0.0_f32; 4 * 8];
    data[0] = 16.0;
    data[1] = 0.0;

    let img = make_complex_2d(data, 4, 8);
    let result = InverseFftFilter::new().apply(&img).unwrap();

    assert_eq!(
        result.shape(),
        [4, 4],
        "DC-only IFFT of [4,8] must produce shape [4,4]"
    );

    let (vals, _) = extract_vec(&result).unwrap();
    assert_eq!(vals.len(), 16, "output must contain H*W = 4*4 = 16 pixels");
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 1.0_f32).abs() < 1e-4,
            "pixel {} must be ~1.0 (DC-only IFFT), got {}",
            i,
            v
        );
    }
}
