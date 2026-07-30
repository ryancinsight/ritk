//! Pure-Rust JPEG-LS (ISO 14495-1 / ITU-T T.87) encoder.
//!
//! Produces the single-component, non-interleaved grayscale profile supported
//! by RITK's DICOM boundary: lossless (NEAR = 0, TS
//! 1.2.840.10008.1.2.4.80) and near-lossless (NEAR > 0, TS
//! 1.2.840.10008.1.2.4.81). The lossless path mirrors
//! `super::scan::decode_scan` line-for-line; the shared context model in
//! `super::context` is the single source of truth for both.
//!
//! # Evidence tier
//! Differential round-trip tests: encoder and decoder are independent code
//! paths over the shared context model. Lossless ⟹ exact reconstruction
//! (`|decoded − original| = 0`); near-lossless ⟹ `|decoded − original| ≤ NEAR`
//! (ISO 14495-1 §A.4.4), asserted value-by-value in `jpeg_ls::tests` and in
//! the ritk-io DICOM round-trip suite. Cross-implementation conformance was
//! verified one-time against the CharLS reference decoder before the charls
//! dependency was removed.

use super::bitstream::BitWriter;
use super::context::{
    compute_k, context_index, default_thresholds, error_correction, modulo_reduce, quant,
    quantize_error, reconstruct, sign_normalize, update_context, CodingParams, ContextModel,
};
use super::reconstruction::ReconstructionRows;
use super::scan::{predict, Predictor, J};

mod validation;

use validation::validate_encoding;
pub use validation::JpegLsEncodeError;

/// Encode a grayscale image as a complete JPEG-LS stream (SOI … EOI).
///
/// # Parameters
/// - `samples`: row-major sample values; each must satisfy `v < 2^bpp`.
/// - `rows` / `cols`: image dimensions (1..=65535).
/// - `bpp`: bits per sample (8..=16; DICOM grayscale range).
/// - `near`: maximum allowed reconstruction error per sample (0 = lossless).
///
/// # Errors
/// Returns [`JpegLsEncodeError`] when the dimensions are empty or cannot be
/// represented by SOF55, the sample count or range does not match the
/// declared image, `bpp` is outside 8..=16, or `near` exceeds the coded range.
///
/// # Examples
/// ```
/// use ritk_codecs::jpeg_ls::encoder::{
///     encode_grayscale_jpeg_ls, JpegLsEncodeError,
/// };
///
/// let samples = [0, 64, 128, 255];
/// let stream = encode_grayscale_jpeg_ls(&samples, 2, 2, 8, 0)?;
/// assert_eq!(&stream[..2], &[0xFF, 0xD8]);
/// assert_eq!(&stream[stream.len() - 2..], &[0xFF, 0xD9]);
/// # Ok::<(), JpegLsEncodeError>(())
/// ```
pub fn encode_grayscale_jpeg_ls(
    samples: &[u16],
    rows: u32,
    cols: u32,
    bpp: u32,
    near: u32,
) -> Result<Vec<u8>, JpegLsEncodeError> {
    let validated = validate_encoding(samples, rows, cols, bpp, near)?;
    let scan_bytes = encode_scan(
        samples,
        validated.rows,
        validated.cols,
        u32::from(validated.precision),
        validated.near,
    );

    let mut out = Vec::with_capacity(scan_bytes.len() + 32);
    // SOI.
    out.extend_from_slice(&super::SOI.to_be_bytes());
    // SOF55 (JPEG-LS frame header): Lf=11, P, Y, X, Nf=1, [C1, HV=0x11, Tq=0].
    out.extend_from_slice(&super::SOF55.to_be_bytes());
    out.extend_from_slice(&11u16.to_be_bytes());
    out.push(validated.precision);
    out.extend_from_slice(
        &u16::try_from(rows)
            .expect("invariant: validated JPEG-LS rows fit in SOF55")
            .to_be_bytes(),
    );
    out.extend_from_slice(
        &u16::try_from(cols)
            .expect("invariant: validated JPEG-LS columns fit in SOF55")
            .to_be_bytes(),
    );
    out.push(1); // Nf = 1 component
    out.push(1); // C1
    out.push(0x11); // H1/V1
    out.push(0); // Tq1
                 // SOS: Ls=8, Ns=1, [Cs1=1, Td/Ta=0], NEAR, ILV=0, point transform=0.
    out.extend_from_slice(&super::SOS.to_be_bytes());
    out.extend_from_slice(&8u16.to_be_bytes());
    out.push(1); // Ns
    out.push(1); // Cs1
    out.push(0); // mapping table selector
    out.push(u8::try_from(near).expect("invariant: validated JPEG-LS NEAR fits in the SOS field")); // NEAR
    out.push(0); // ILV (single component)
    out.push(0); // point transform
                 // Entropy-coded scan.
    out.extend_from_slice(&scan_bytes);
    // EOI.
    out.extend_from_slice(&super::EOI.to_be_bytes());
    Ok(out)
}

/// Forward error mapping (ISO 14495-1 §A.5.3): inverse of
/// [`super::context::inverse_map`].
#[inline(always)]
fn forward_map(errval: i32) -> u32 {
    if errval >= 0 {
        (errval as u32) << 1
    } else {
        ((-errval as u32) << 1) - 1
    }
}

/// Encode one single-component scan — the mirror of
/// `super::scan::decode_scan`, generalised by the NEAR parameter.
fn encode_scan(samples: &[u16], rows: usize, cols: usize, bpp: u32, near: i32) -> Vec<u8> {
    let cp = CodingParams::new(bpp, near);
    let maxval = cp.maxval;
    let range = cp.range;
    let qbpp = cp.qbpp;
    let limit = cp.limit;
    let (t1, t2, t3) = default_thresholds(maxval, near);

    let mut ctx = ContextModel::new(cp.a_init);
    let mut bw = BitWriter::new();

    // Prediction is causal: each sample reads only the row above and samples
    // already reconstructed in the current row. Reuse two rows so scratch is
    // O(cols), independent of image height.
    let mut reconstructed = ReconstructionRows::new(cols);

    for r in 0..rows {
        let current_line_left_guard = reconstructed.current_line_left_guard();

        let mut c = 0usize;
        while c < cols {
            let x = i32::from(samples[r * cols + c]);
            let (a, b, cc, d) = reconstructed.neighborhood(c, current_line_left_guard);

            let q1 = quant(d - b, t1, t2, t3, near);
            let q2 = quant(b - cc, t1, t2, t3, near);
            let q3 = quant(cc - a, t1, t2, t3, near);

            if q1 == 0 && q2 == 0 && q3 == 0 {
                // ── Run mode (ISO 14495-1 §A.7) ──────────────────────────────
                let run_val = a;
                let remaining = cols - c;
                let mut run_len = 0usize;
                while run_len < remaining
                    && (i32::from(samples[r * cols + c + run_len]) - run_val).abs() <= near
                {
                    run_len += 1;
                }

                // Emit run-length code mirroring the decoder's hit/miss loop.
                let mut acc = 0usize;
                loop {
                    let idx = ctx.run_index.min(31);
                    let seg = 1usize << J[idx];
                    if acc + seg <= run_len {
                        // Hit: a full 2^J[idx] segment of the run.
                        bw.write_bit(1);
                        acc += seg;
                        if acc >= remaining {
                            // Run filled the line exactly; decoder clamps.
                            if ctx.run_index < 31 {
                                ctx.run_index += 1;
                            }
                            break;
                        }
                        if ctx.run_index < 31 {
                            ctx.run_index += 1;
                        }
                    } else if run_len == remaining {
                        // Run reaches end-of-line mid-segment: one overshoot
                        // hit; the decoder clamps `run_len` to `remaining`.
                        bw.write_bit(1);
                        if ctx.run_index < 31 {
                            ctx.run_index += 1;
                        }
                        break;
                    } else {
                        // Miss: remainder in J[idx] bits, then the interrupt
                        // sample.
                        bw.write_bit(0);
                        if J[idx] > 0 {
                            bw.write_bits((run_len - acc) as u32, J[idx]);
                        }
                        break;
                    }
                }
                // Run samples reconstruct to run_val (within NEAR of source).
                reconstructed.fill_current(c, run_len, run_val);
                c += run_len;

                if run_len < remaining {
                    // ── Run interruption sample (§A.7.2) ─────────────────────
                    let xi = i32::from(samples[r * cols + c]);
                    let (rb, ra) = reconstructed.interruption_neighbors(c);
                    let run_index = ctx.run_index.min(31);
                    let ri_ctx = if (rb - ra).abs() <= near {
                        &mut ctx.run_int_same
                    } else {
                        &mut ctx.run_int_diff
                    };
                    let ritype = ri_ctx.run_interruption_type();
                    let k = ri_ctx.compute_k(qbpp);

                    // Prediction error relative to the decoder's reconstruction.
                    let sgn = if ritype == 1 { 1 } else { (rb - ra).signum() };
                    let pred = if ritype == 1 { ra } else { rb };
                    let errval = modulo_reduce(quantize_error(sgn * (xi - pred), near), range);

                    // Inverse of `compute_error_value`: pick temp parity so the
                    // decoder reproduces `errval`'s sign.
                    let cond = ri_ctx.negative_maps_to_odd(k);
                    let abs = errval.unsigned_abs();
                    let temp = if errval == 0 {
                        0
                    } else if (errval < 0) == cond {
                        2 * abs - 1
                    } else {
                        2 * abs
                    };
                    let me = temp - ritype;
                    bw.write_golomb(me, k, limit - J[run_index] - 1, qbpp);
                    ri_ctx.update(errval, me);

                    if ctx.run_index > 0 {
                        ctx.run_index -= 1;
                    }
                    reconstructed.set_current(c, reconstruct(pred, sgn * errval, &cp));
                    c += 1;
                }
                continue;
            }

            // ── Regular mode (ISO 14495-1 §A.4–A.5) ──────────────────────────
            let (nq1, nq2, nq3, sign) = sign_normalize(q1, q2, q3);
            let qi = context_index(nq1, nq2, nq3);
            let ctx_reg = &mut ctx.regular[qi];

            let k = compute_k(ctx_reg.a, ctx_reg.n, qbpp);

            let px_raw = predict(a, b, cc, d, Predictor::Adaptive, r, c);
            let signed_c = if sign < 0 { -ctx_reg.c } else { ctx_reg.c };
            let px = (px_raw + signed_c).clamp(0, maxval);

            // Canonical error the decoder must end with after its k=0 XOR step.
            let errval_canon = modulo_reduce(quantize_error(sign * (x - px), near), range);
            // Pre-map value: undo the decoder's XOR correction (involution).
            let premap = if k == 0 {
                errval_canon ^ error_correction(ctx_reg, near as u32)
            } else {
                errval_canon
            };
            bw.write_golomb(forward_map(premap), k, limit, qbpp);

            update_context(ctx_reg, errval_canon, near as u32);
            reconstructed.set_current(c, reconstruct(px, sign * errval_canon, &cp));
            c += 1;
        }
        reconstructed.finish_row(current_line_left_guard);
    }

    bw.finish()
}
