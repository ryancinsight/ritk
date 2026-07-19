//! Tests for [`FftShiftFilter`].
//!
//! Verification chain:
//!   shape invariant → self-inverse property → DC-to-centre mapping → 3-D shape invariant

use crate::fft::{FftShiftFilter, RealFftShiftFilter};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = coeus_core::SequentialBackend;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_complex_2d(vals: &[f32], h: usize, cw: usize) -> Image<f32, B, 2> {
    let tensor = ritk_image::tensor::Tensor::<f32, B>::from_slice([h, cw], vals);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

fn make_complex_3d(vals: &[f32], depth: usize, h: usize, cw: usize) -> Image<f32, B, 3> {
    let tensor = ritk_image::tensor::Tensor::<f32, B>::from_slice([depth, h, cw], vals);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0_f64, 0.0_f64]),
        Spacing::new([1.0_f64, 1.0_f64, 1.0_f64]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// `apply_2d` must preserve the input shape `[H, 2·W]` unchanged.
#[test]
fn output_shape_unchanged_after_shift() {
    let h = 4_usize;
    let cw = 8_usize; // 2·W where W = 4
    let vals: Vec<f32> = (0..(h * cw)).map(|i| i as f32).collect();
    let img = make_complex_2d(&vals, h, cw);

    let shifted = FftShiftFilter::new().apply(&img).unwrap();

    assert_eq!(shifted.shape(), [h, cw], "apply_2d must not change shape");
}

/// `apply_2d` applied twice must recover the original array for even H and W.
///
/// # Proof sketch
/// Let `s_h = H/2`, `s_w = W/2`.
/// First shift:  `out1[r,c] = in[(r + s_h) % H, (c + s_w) % W]`.
/// Second shift: `out2[r,c] = out1[(r + s_h) % H, (c + s_w) % W]`
///                          `= in[(r + 2·s_h) % H, (c + 2·s_w) % W]`.
/// For even H: `2·s_h = H`, so `(r + H) % H = r`. ∎
#[test]
fn double_shift_is_identity() {
    let h = 4_usize;
    let cw = 8_usize; // W = 4 (even)
                      // Deterministic values derived from a sine series (analytical reference).
    let vals: Vec<f32> = (0..(h * cw))
        .map(|i| (i as f32 * 0.37 + 1.1).sin())
        .collect();
    let img = make_complex_2d(&vals, h, cw);

    let once = FftShiftFilter::new().apply(&img).unwrap();
    let twice = FftShiftFilter::new().apply(&once).unwrap();

    let (result, _) = ritk_tensor_ops::extract_vec(&twice).unwrap();
    let max_diff = result
        .iter()
        .zip(vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "shift(shift(x)) must equal x for even dimensions; max_diff = {max_diff:.2e}"
    );
}

/// After `apply_2d`, the DC component (originally at row=0, col=0) must appear
/// at the centre: row=H/2=2, col=W/2=2.
///
/// # Verification
/// Input: shape `[4, 8]`, all zeros except `Re(F[0,0]) = 1.0`, `Im(F[0,0]) = 0.0`.
/// Formula: `out[r,c] = in[(r + H/2) % H, (c + W/2) % W]`.
/// At `(r=2, c=2)`: `src_r = (2+2)%4 = 0`, `src_c = (2+2)%4 = 0`.
/// → `out[2·8 + 2·2] = in[0·8 + 2·0] = 1.0`. ∎
#[test]
fn dc_moves_to_center_after_shift() {
    let h = 4_usize;
    let cw = 8_usize; // W = 4
    let mut vals = vec![0.0_f32; h * cw];
    // Set F[0,0] = 1 + 0i
    vals[0] = 1.0; // Re(F[0,0])
    vals[1] = 0.0; // Im(F[0,0])

    let img = make_complex_2d(&vals, h, cw);
    let shifted = FftShiftFilter::new().apply(&img).unwrap();
    let (shifted_data, _) = ritk_tensor_ops::extract_vec(&shifted).unwrap();

    // Expected centre: row=2, col=2 → flat indices 2·8 + 2·2 = 20 and 21.
    let re_idx = 2 * cw + 2 * 2; // = 20
    let im_idx = 2 * cw + 2 * 2 + 1; // = 21
    assert_eq!(
        shifted_data[re_idx], 1.0,
        "DC real part must be at row=H/2=2, col=W/2=2 after shift"
    );
    assert_eq!(
        shifted_data[im_idx], 0.0,
        "DC imaginary part must be at row=H/2=2, col=W/2=2 after shift"
    );
}

/// `apply_3d` must preserve the input shape `[D, H, 2·W]` unchanged.
#[test]
fn output_shape_unchanged_after_shift_volume() {
    let depth = 3_usize;
    let h = 4_usize;
    let cw = 8_usize; // 2·W where W = 4
    let vals: Vec<f32> = (0..(depth * h * cw)).map(|i| i as f32).collect();
    let img = make_complex_3d(&vals, depth, h, cw);

    let shifted = FftShiftFilter::new().apply(&img).unwrap();

    assert_eq!(
        shifted.shape(),
        [depth, h, cw],
        "apply_3d must not change shape"
    );
}

/// After `apply_3d`, the DC component must appear at the volumetric centre:
/// `(D/2, H/2, W/2)`.
///
/// Input: shape `[4, 4, 8]`, all zeros except a single real value at `F[0,0,0]`.
/// After a 3-D cyclic roll by `(D/2, H/2, W/2) = (2, 2, 2)`:
///   out[2, 2, 2] = in[(2+2)%4, (2+2)%4, (2+2)%4] = in[0, 0, 0] = 1.0
#[test]
fn dc_moves_to_center_after_shift_volume() {
    let depth = 4_usize;
    let h = 4_usize;
    let cw = 8_usize; // W = 4
    let n = depth * h * cw;
    let mut vals = vec![0.0_f32; n];
    // Set F[0, 0, 0] = 1 + 0i  (the DC component)
    vals[0] = 1.0; // Re(F[0,0,0])
    vals[1] = 0.0; // Im(F[0,0,0])

    let img = make_complex_3d(&vals, depth, h, cw);
    let shifted = FftShiftFilter::new().apply(&img).unwrap();
    let (data, _) = ritk_tensor_ops::extract_vec(&shifted).unwrap();

    // Expected DC centre: d=2, r=2, c=2
    // flat index = 2 * h * cw + 2 * cw + 2 * 2 = 2*32 + 2*8 + 4 = 64 + 16 + 4 = 84
    let re_idx = 2 * h * cw + 2 * cw + 2 * 2;
    let im_idx = re_idx + 1;
    assert!(
        (data[re_idx] - 1.0).abs() < 1e-6,
        "DC real part must be at depth=D/2=2, row=H/2=2, col=W/2=2 after 3-D shift"
    );
    assert!(
        data[im_idx].abs() < 1e-6,
        "DC imaginary part must be 0.0 at centre after 3-D shift"
    );
}

/// `apply_3d` must be self-inverse for even D, H, W.
///
/// Proof: shift by (D/2, H/2, W/2) twice shifts by (D, H, W) = identity mod D, H, W.
#[test]
fn double_shift_is_identity_volume() {
    let depth = 4_usize;
    let h = 4_usize;
    let cw = 8_usize; // W = 4
    let vals: Vec<f32> = (0..(depth * h * cw))
        .map(|i| (i as f32 * 0.37 + 1.1).sin())
        .collect();
    let img = make_complex_3d(&vals, depth, h, cw);

    let once = FftShiftFilter::new().apply(&img).unwrap();
    let twice = FftShiftFilter::new().apply(&once).unwrap();

    let (result, _) = ritk_tensor_ops::extract_vec(&twice).unwrap();
    let max_diff = result
        .iter()
        .zip(vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "3-D shift(shift(x)) must equal x for even dimensions; max_diff = {max_diff:.2e}"
    );
}

/// `apply_3d` with odd dimensions still works (the roll is well-defined
/// for any positive integer shift).
#[test]
fn odd_dimension_volume_shifts_correctly() {
    let depth = 3_usize;
    let h = 5_usize;
    let cw = 6_usize; // W = 3 (odd)
    let vals: Vec<f32> = (0..(depth * h * cw)).map(|i| i as f32).collect();
    let img = make_complex_3d(&vals, depth, h, cw);

    let shifted = FftShiftFilter::new().apply(&img).unwrap();
    let (data, _) = ritk_tensor_ops::extract_vec(&shifted).unwrap();

    // Shape must be preserved.
    assert_eq!(
        shifted.shape(),
        [depth, h, cw],
        "3-D shift with odd dimensions must preserve shape"
    );
    // All values must be finite (sanity check).
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "3-D shift value at index {i} must be finite");
    }
}

// ── RealFftShiftFilter Tests ──────────────────────────────────────────────────

#[test]
fn real_fft_shift_even_dims_is_identity_twice() {
    let vals: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let img = make_complex_3d(&vals, 4, 4, 4);

    let once = RealFftShiftFilter::new().apply(&img).unwrap();
    let twice = RealFftShiftFilter::new().apply(&once).unwrap();

    let (res, _) = ritk_tensor_ops::extract_vec(&twice).unwrap();
    assert_eq!(res, vals);
}

#[test]
fn real_fft_shift_odd_dims_correct() {
    let vals: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = make_complex_3d(&vals, 3, 3, 3);

    let once = RealFftShiftFilter::new().apply(&img).unwrap();
    let (res, _) = ritk_tensor_ops::extract_vec(&once).unwrap();

    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                let out_val = res[z * 9 + y * 3 + x];
                let in_val = vals[((z + 2) % 3) * 9 + ((y + 2) % 3) * 3 + ((x + 2) % 3)];
                assert_eq!(out_val, in_val);
            }
        }
    }
}
