//! 8×8 inverse discrete cosine transform (IDCT) for JPEG baseline decode.
//!
//! # Specification
//! ITU-T T.81 §A.3.3 defines the 2D IDCT over an 8×8 block of quantized
//! frequency coefficients. This module implements the separable 1D form:
//! apply the 1D IDCT across all rows, transpose, apply to columns.
//!
//! The 1D 8-point IDCT formula (T.81 Eq. A.3.3):
//!   `f[x] = (1/2) · Σ_{u=0}^{7} C(u) · F[u] · cos((2x+1)·u·π/16)`
//! where C(0) = 1/√2, C(u>0) = 1.

use std::f64::consts::{PI, SQRT_2};
use std::sync::LazyLock;

use crate::jpeg::constants::{DCT_BLOCK_CELLS, DCT_BLOCK_DIM};

/// Cosine basis table: `COSINE[u][x] = C(u) · cos((2x+1)·u·π/16)`.
///
/// The basis depends only on the block dimension, so it is built once for the
/// process rather than per block: a 512×512 image decodes 4096 blocks, and
/// rebuilding the table at each one would evaluate a quarter-million `cos`
/// calls whose results never differ. `cos` is not a `const fn`, so the table
/// is initialised on first use rather than at compile time.
///
/// Built in f64 to minimise accumulated rounding error, then narrowed to f32
/// for arithmetic on block samples.
static COSINE: LazyLock<[[f32; DCT_BLOCK_DIM]; DCT_BLOCK_DIM]> = LazyLock::new(|| {
    let mut c = [[0.0f32; DCT_BLOCK_DIM]; DCT_BLOCK_DIM];
    for (u, row) in c.iter_mut().enumerate() {
        let cu = if u == 0 { 1.0 / SQRT_2 } else { 1.0_f64 };
        for (x, val) in row.iter_mut().enumerate() {
            *val = (cu * ((2 * x + 1) as f64 * u as f64 * PI / (2 * DCT_BLOCK_DIM) as f64).cos())
                as f32;
        }
    }
    c
});

/// Apply 1D IDCT in-place to a slice of 8 `f32` coefficients.
#[inline]
fn idct_1d(f: &mut [f32], cosines: &[[f32; DCT_BLOCK_DIM]; DCT_BLOCK_DIM]) {
    debug_assert_eq!(f.len(), DCT_BLOCK_DIM);
    let mut tmp = [0.0f32; DCT_BLOCK_DIM];
    for x in 0..DCT_BLOCK_DIM {
        let mut s = 0.0f32;
        for u in 0..DCT_BLOCK_DIM {
            s += cosines[u][x] * f[u];
        }
        tmp[x] = s * 0.5;
    }
    f.copy_from_slice(&tmp);
}

/// Apply 2D IDCT in-place to a flattened 8×8 block (row-major: index = row*8+col).
///
/// After transform, level-shift and clamping are applied by the caller.
pub(crate) fn idct_8x8(block: &mut [f32; DCT_BLOCK_CELLS]) {
    let cos = &*COSINE;
    // Row-wise 1D IDCT
    for row in 0..DCT_BLOCK_DIM {
        let start = row * DCT_BLOCK_DIM;
        idct_1d(&mut block[start..start + DCT_BLOCK_DIM], cos);
    }
    // Transpose in-place
    for r in 0..DCT_BLOCK_DIM {
        for c in (r + 1)..DCT_BLOCK_DIM {
            block.swap(r * DCT_BLOCK_DIM + c, c * DCT_BLOCK_DIM + r);
        }
    }
    // Column-wise 1D IDCT (operates on transposed layout → original columns)
    for row in 0..DCT_BLOCK_DIM {
        let start = row * DCT_BLOCK_DIM;
        idct_1d(&mut block[start..start + DCT_BLOCK_DIM], cos);
    }
    // Transpose back
    for r in 0..DCT_BLOCK_DIM {
        for c in (r + 1)..DCT_BLOCK_DIM {
            block.swap(r * DCT_BLOCK_DIM + c, c * DCT_BLOCK_DIM + r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct_all_zero_coefficients_produces_zero() {
        let mut block = [0.0f32; DCT_BLOCK_CELLS];
        idct_8x8(&mut block);
        for v in block {
            assert!(v.abs() < 1e-5, "expected 0, got {v}");
        }
    }

    /// DC-only block: f[x][y] = F[0][0] / 8 for all x, y.
    #[test]
    fn idct_dc_only_gives_constant_output() {
        let mut block = [0.0f32; DCT_BLOCK_CELLS];
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

    /// The 2-D IDCT straight from T.81 Eq. A.3.3, without separation.
    ///
    /// `block[r * 8 + c]` holds `F[v = r][u = c]` on input and `f[y = r][x = c]`
    /// on output. Evaluated in f64 so it is the accuracy reference, not a
    /// restatement of the code under test.
    fn direct_idct_reference(block: &[f32; DCT_BLOCK_CELLS]) -> [f64; DCT_BLOCK_CELLS] {
        let c = |k: usize| if k == 0 { 1.0 / SQRT_2 } else { 1.0 };
        let basis = |pos: usize, freq: usize| {
            ((2 * pos + 1) as f64 * freq as f64 * PI / (2 * DCT_BLOCK_DIM) as f64).cos()
        };
        let mut out = [0.0f64; DCT_BLOCK_CELLS];
        for y in 0..DCT_BLOCK_DIM {
            for x in 0..DCT_BLOCK_DIM {
                let mut sum = 0.0;
                for v in 0..DCT_BLOCK_DIM {
                    for u in 0..DCT_BLOCK_DIM {
                        sum += c(u)
                            * c(v)
                            * f64::from(block[v * DCT_BLOCK_DIM + u])
                            * basis(x, u)
                            * basis(y, v);
                    }
                }
                out[y * DCT_BLOCK_DIM + x] = sum / 4.0;
            }
        }
        out
    }

    /// The separable implementation must equal the spec's 2-D definition.
    ///
    /// The zero and DC-only cases above are satisfied by any implementation
    /// with the right scale factor — they cannot distinguish a transposed or
    /// mis-normalised basis. A full-spectrum block driven through the direct
    /// definition can, because every coefficient contributes to every output
    /// sample through a distinct pair of basis functions.
    ///
    /// Tolerance: the implementation accumulates 8 f32 terms per pass over two
    /// passes with coefficients up to 200, so intermediate magnitudes reach
    /// ~1e3 and f32's 1.2e-7 relative precision admits ~1e-4 per sample after
    /// the two accumulations. 1e-3 clears that with margin while remaining far
    /// below the error any basis or transpose mistake would produce.
    #[test]
    fn idct_matches_the_direct_two_dimensional_definition() {
        // Deterministic full-spectrum coefficients: every (u, v) is populated,
        // with signs alternating so no term is masked by cancellation with a
        // neighbour.
        let mut block = [0.0f32; DCT_BLOCK_CELLS];
        for (i, coeff) in block.iter_mut().enumerate() {
            let magnitude = 200.0 - (i as f32) * 2.0;
            *coeff = if i % 3 == 0 { -magnitude } else { magnitude };
        }
        let expected = direct_idct_reference(&block);

        idct_8x8(&mut block);

        for (i, (&actual, &want)) in block.iter().zip(expected.iter()).enumerate() {
            assert!(
                (f64::from(actual) - want).abs() < 1e-3,
                "sample {i} (row {}, col {}): got {actual}, spec gives {want}",
                i / DCT_BLOCK_DIM,
                i % DCT_BLOCK_DIM
            );
        }
    }

    /// The IDCT is linear, so it commutes with scaling and addition.
    ///
    /// Independent of the basis values themselves: an implementation with a
    /// wrong table still passes, while one that clamps, saturates, or carries
    /// state between calls fails. The shared basis table makes the last of
    /// those a live concern.
    #[test]
    fn idct_is_linear_and_carries_no_state_between_calls() {
        let mut left = [0.0f32; DCT_BLOCK_CELLS];
        let mut right = [0.0f32; DCT_BLOCK_CELLS];
        for i in 0..DCT_BLOCK_CELLS {
            left[i] = (i as f32) - 32.0;
            right[i] = 64.0 - (i as f32) * 1.5;
        }
        let mut combined = [0.0f32; DCT_BLOCK_CELLS];
        for i in 0..DCT_BLOCK_CELLS {
            combined[i] = 3.0f32.mul_add(left[i], right[i]);
        }

        idct_8x8(&mut left);
        idct_8x8(&mut right);
        idct_8x8(&mut combined);

        for i in 0..DCT_BLOCK_CELLS {
            let superposed = 3.0f32.mul_add(left[i], right[i]);
            assert!(
                (combined[i] - superposed).abs() < 1e-2,
                "sample {i}: transform of the sum gave {}, sum of transforms {superposed}",
                combined[i]
            );
        }
    }
}
