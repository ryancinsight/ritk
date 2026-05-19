//! 8×8 inverse discrete cosine transform (IDCT) for JPEG baseline decode.
//!
//! # Specification
//! ITU-T T.81 §A.3.3 defines the 2D IDCT over an 8×8 block of quantized
//! frequency coefficients. This module implements the separable 1D form:
//! apply the 1D IDCT across all rows, transpose, apply to columns.
//!
//! The 1D 8-point IDCT formula (T.81 Eq. A.3.3):
//!   f[x] = (1/2) · Σ_{u=0}^{7} C(u) · F[u] · cos((2x+1)·u·π/16)
//! where C(0) = 1/√2, C(u>0) = 1.

use std::f64::consts::{PI, SQRT_2};

/// Cosine basis table: `COSINE[u][x] = C(u) · cos((2x+1)·u·π/16)`.
///
/// Using f64 precision for the table to minimise accumulated rounding error,
/// then cast to f32 for arithmetic on block samples.
fn cosine_table() -> [[f32; 8]; 8] {
    let mut c = [[0.0f32; 8]; 8];
    for (u, row) in c.iter_mut().enumerate() {
        let cu = if u == 0 { 1.0 / SQRT_2 } else { 1.0_f64 };
        for (x, val) in row.iter_mut().enumerate() {
            *val = (cu * ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos()) as f32;
        }
    }
    c
}

/// Apply 1D IDCT in-place to a slice of 8 `f32` coefficients.
#[inline]
fn idct_1d(f: &mut [f32], cosines: &[[f32; 8]; 8]) {
    debug_assert_eq!(f.len(), 8);
    let mut tmp = [0.0f32; 8];
    for x in 0..8 {
        let mut s = 0.0f32;
        for u in 0..8 {
            s += cosines[u][x] * f[u];
        }
        tmp[x] = s * 0.5;
    }
    f.copy_from_slice(&tmp);
}

/// Apply 2D IDCT in-place to a flattened 8×8 block (row-major: index = row*8+col).
///
/// After transform, level-shift and clamping are applied by the caller.
pub(crate) fn idct_8x8(block: &mut [f32; 64]) {
    let cos = cosine_table();
    // Row-wise 1D IDCT
    for row in 0..8 {
        let start = row * 8;
        idct_1d(&mut block[start..start + 8], &cos);
    }
    // Transpose in-place
    for r in 0..8 {
        for c in (r + 1)..8 {
            block.swap(r * 8 + c, c * 8 + r);
        }
    }
    // Column-wise 1D IDCT (operates on transposed layout → original columns)
    for row in 0..8 {
        let start = row * 8;
        idct_1d(&mut block[start..start + 8], &cos);
    }
    // Transpose back
    for r in 0..8 {
        for c in (r + 1)..8 {
            block.swap(r * 8 + c, c * 8 + r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_all_zero_coefficients_produces_zero() {
        let mut block = [0.0f32; 64];
        idct_8x8(&mut block);
        for v in block {
            assert!(v.abs() < 1e-5, "expected 0, got {v}");
        }
    }

    /// DC-only block: f[x][y] = F[0][0] / 8 for all x, y.
    #[test]
    fn idct_dc_only_gives_constant_output() {
        let mut block = [0.0f32; 64];
        block[0] = 8.0 * 32.0; // DC coefficient = 8 * desired spatial value
        idct_8x8(&mut block);
        let v0 = block[0];
        for (i, &v) in block.iter().enumerate() {
            assert!((v - v0).abs() < 1e-3, "block[{i}] = {v}, expected {v0}");
        }
        assert!(
            (v0 - 32.0).abs() < 1e-2,
            "DC roundtrip: expected 32.0, got {v0}"
        );
    }
}
