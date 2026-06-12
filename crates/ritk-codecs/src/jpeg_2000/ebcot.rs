//! EBCOT tier-1 encoder and decoder (ISO 15444-1 Annex D).
//!
//! # Overview
//! EBCOT (Embedded Block Coding with Optimised Truncation) processes each
//! code-block's samples bit-plane by bit-plane, from the most-significant to
//! the least-significant bit.  Each bit-plane has three coding passes:
//!
//! 1. **Significance Propagation Pass (SPP)**: encodes non-significant samples
//!    that have at least one already-significant neighbour.
//! 2. **Magnitude Refinement Pass (MRP)**: refines significant samples that
//!    became significant before this bit-plane.
//! 3. **Cleanup Pass (CUP)**: encodes all remaining samples, using run-length
//!    coding when four consecutive column entries are all non-significant.
//!
//! # Context labels (ISO 15444-1 §D.3)
//! - ZC  (significance)       : 0–8
//! - SC  (sign)               : 9–13
//! - MR  (magnitude refine)   : 14–16
//! - UNI (uniform)            : 17
//! - AGG (aggregation / RLC)  : 18

use super::mq_coder::{initial_contexts, MqDecoder, MqEncoder};

// ── Context indices ───────────────────────────────────────────────────────────

const CTX_ZC_BASE: usize = 0; // 9 significance contexts (0..8)
const CTX_SC_BASE: usize = 9; // 5 sign contexts (9..13)
const CTX_MR_BASE: usize = 14; // 3 magnitude-refinement contexts (14..16)
const CTX_UNI: usize = 17; // uniform
const CTX_AGG: usize = 18; // aggregation / run-length

// ── Subband orientation ───────────────────────────────────────────────────────

/// Subband orientation, used to select the significance context function.
#[allow(dead_code)] // Hl and Hh needed when DWT support is added (J2K-DECODE-DWT)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubbandOrientation {
    /// LL (lowest frequency) or LH (horizontal high-pass, vertical low-pass).
    LlOrLh,
    /// HL (horizontal low-pass, vertical high-pass).
    Hl,
    /// HH (both high-pass).
    Hh,
}

// ── Context helper functions (ISO 15444-1 §D.3) ───────────────────────────────

/// Significance context for LL / LH subbands (ISO 15444-1 Table D.1, columns H/V/D).
///
/// - `h`: number of significant horizontal (left + right) neighbours (0–2).
/// - `v`: number of significant vertical (above + below) neighbours (0–2).
/// - `d`: number of significant diagonal neighbours (NE+NW+SE+SW) (0–4).
#[inline]
fn zc_ll_lh(h: u32, v: u32, d: u32) -> usize {
    CTX_ZC_BASE
        + match (h, v, d) {
            (2, _, _) => 8,
            (1, v, _) if v >= 1 => 7,
            (1, 0, d) if d >= 1 => 6,
            (1, 0, 0) => 5,
            (0, 2, _) => 4,
            (0, 1, _) => 3,
            (0, 0, d) if d >= 2 => 2,
            (0, 0, 1) => 1,
            _ => 0,
        }
}

/// Significance context for HL subband (H and V roles swapped).
#[inline]
fn zc_hl(h: u32, v: u32, d: u32) -> usize {
    zc_ll_lh(v, h, d)
}

/// Significance context for HH subband (ISO 15444-1 Table D.2).
#[inline]
fn zc_hh(h: u32, v: u32, d: u32) -> usize {
    CTX_ZC_BASE
        + match (d, h + v) {
            (d, _) if d >= 3 => 8,
            (2, hv) if hv >= 1 => 7,
            (2, _) => 6,
            (1, hv) if hv >= 2 => 5,
            (1, 1) => 4,
            (1, 0) => 3,
            (0, hv) if hv >= 2 => 2,
            (0, 1) => 1,
            _ => 0,
        }
}

/// Choose the significance context given orientation and neighbour counts.
#[inline]
fn zc_context(orient: SubbandOrientation, h: u32, v: u32, d: u32) -> usize {
    match orient {
        SubbandOrientation::LlOrLh => zc_ll_lh(h, v, d),
        SubbandOrientation::Hl => zc_hl(h, v, d),
        SubbandOrientation::Hh => zc_hh(h, v, d),
    }
}

/// Sign context from horizontal/vertical sign contributions (ISO 15444-1 Table D.3).
///
/// Returns `(ctx_index, xor_bit)` where `xor_bit` inverts the coded sign.
///
/// `kh` / `kv` are the net sign contributions (−1, 0, or +1) of the horizontal
/// and vertical significant neighbours respectively.
#[inline]
fn sc_context(kh: i32, kv: i32) -> (usize, u32) {
    match (kh, kv) {
        (1, 1) => (CTX_SC_BASE + 4, 0),
        (1, 0) => (CTX_SC_BASE + 3, 0),
        (1, -1) => (CTX_SC_BASE + 2, 0),
        (0, 1) => (CTX_SC_BASE + 1, 0),
        (0, 0) => (CTX_SC_BASE, 0),
        (0, -1) => (CTX_SC_BASE + 1, 1),
        (-1, 1) => (CTX_SC_BASE + 2, 1),
        (-1, 0) => (CTX_SC_BASE + 3, 1),
        (-1, -1) => (CTX_SC_BASE + 4, 1),
        _ => (CTX_SC_BASE, 0), // unreachable in practice
    }
}

/// Magnitude refinement context (ISO 15444-1 §D.3.3).
///
/// - `has_sig_other`: any of the 8 neighbours is significant.
/// - `refined_before`: this sample has been magnitude-refined at least once.
#[inline]
fn mr_context(has_sig_other: bool, refined_before: bool) -> usize {
    if refined_before {
        CTX_MR_BASE + 2
    } else if has_sig_other {
        CTX_MR_BASE + 1
    } else {
        CTX_MR_BASE
    }
}

// ── Test-only symbol trace (CUP differential debugging) ──────────────────────

#[cfg(test)]
thread_local! {
    /// (context index, bit) pairs recorded by the cleanup pass.
    pub(crate) static CUP_TRACE: std::cell::RefCell<Vec<(usize, u32)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn cup_trace_take() -> Vec<(usize, u32)> {
    CUP_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()))
}

#[inline(always)]
#[allow(unused_variables)]
fn trace(ctx: usize, bit: u32) {
    #[cfg(test)]
    CUP_TRACE.with(|t| t.borrow_mut().push((ctx, bit)));
}
// ── Per-sample state flags ────────────────────────────────────────────────────

/// Compact per-sample state used during EBCOT processing.
#[derive(Clone, Copy, Default)]
struct SampleState {
    sig: bool,    // sample is significant
    sign: bool,   // sign (true = negative)
    visit: bool,  // visited in current SPP
    refine: bool, // has been magnitude-refined at least once
}

// ── Decoded code-block result ─────────────────────────────────────────────────

/// Output of `decode_code_block`: reconstructed sign-magnitude samples in
/// row-major order.  Values are the DC-shifted integer sample values (i32).
/// The caller applies DC un-shift and DICOM rescale.
pub struct DecodedBlock {
    /// Reconstructed samples (row-major), in the order they appear in the tile.
    pub samples: Vec<i32>,
}

// ── EBCOT decoder ─────────────────────────────────────────────────────────────

/// Decode one EBCOT code-block.
///
/// # Parameters
/// - `data`: raw code-block bitstream bytes.
/// - `width` / `height`: code-block dimensions.
/// - `num_bit_planes`: number of bit-planes coded in this block (determines
///   the MSB position; the value is the `P` in the packet header: number of
///   bit-planes from the MSB of the dynamic range down to the last included one).
/// - `num_passes`: total number of coding passes included (SPP + MRP + CUP
///   counted from the highest bit-plane downward).
/// - `guard_bits`: number of guard bits (from QCD); used to derive the first
///   bit-plane position when `num_bit_planes` is the "missing MSBs" count.
/// - `orient`: subband orientation (selects the ZC context function).
///
/// Returns the decoded DC-shifted samples in row-major order.
pub fn decode_code_block(
    data: &[u8],
    width: usize,
    height: usize,
    num_bit_planes: u8,
    num_passes: u32,
    orient: SubbandOrientation,
) -> DecodedBlock {
    let n = width * height;
    let mut state = vec![SampleState::default(); n];
    let mut mag = vec![0u32; n]; // accumulated magnitude

    if data.is_empty() || num_bit_planes == 0 || num_passes == 0 {
        return DecodedBlock {
            samples: vec![0i32; n],
        };
    }

    let mut mq = MqDecoder::new(data);
    let mut ctxs = initial_contexts();

    let total_bit_planes = num_bit_planes as u32;
    let mut passes_remaining = num_passes;

    // Iterate bit-planes from MSB (highest) downward. The first (MSB) plane
    // carries only a cleanup pass (ISO 15444-1 §D.4.1): nothing can be
    // significant yet, so SPP/MRP are skipped and the total pass count for a
    // block with P planes is 3P − 2.
    for bp in (0..total_bit_planes).rev() {
        let first_plane = bp + 1 == total_bit_planes;
        // ── Significance Propagation Pass ────────────────────────────────────
        if !first_plane {
            if passes_remaining == 0 {
                break;
            }
            passes_remaining -= 1;
            // Stripe-oriented scan (ISO 15444-1 §D.2): 4-row stripes, columns
            // within each stripe, rows within each column.
            let mut sy = 0;
            while sy < height {
                for x in 0..width {
                    for y in sy..height.min(sy + 4) {
                        let idx = y * width + x;
                        if state[idx].sig || state[idx].visit {
                            continue;
                        }
                        let (h, v, d) = neighbour_sig_counts(&state, width, height, x, y);
                        if h + v + d == 0 {
                            continue;
                        }
                        state[idx].visit = true;
                        let ctx = zc_context(orient, h, v, d);
                        let sig_bit = mq.decode(&mut ctxs[ctx]);
                        trace(ctx, sig_bit);
                        if sig_bit == 1 {
                            mag[idx] |= 1 << bp;
                            state[idx].sig = true;
                            let (kh, kv) = sign_contributions(&state, width, height, x, y);
                            let (sc_ctx, xor_bit) = sc_context(kh, kv);
                            let raw_sign = mq.decode(&mut ctxs[sc_ctx]);
                            trace(sc_ctx, raw_sign);
                            let sign_bit = raw_sign ^ xor_bit;
                            state[idx].sign = sign_bit != 0;
                        }
                    }
                }
                sy += 4;
            }

            // ── Magnitude Refinement Pass ─────────────────────────────────────────
            if passes_remaining == 0 {
                // Reset visit flags before leaving.
                for s in &mut state {
                    s.visit = false;
                }
                break;
            }
            passes_remaining -= 1;
            let mut sy = 0;
            while sy < height {
                for x in 0..width {
                    for y in sy..height.min(sy + 4) {
                        let idx = y * width + x;
                        if !state[idx].sig || state[idx].visit {
                            continue;
                        }
                        let has_sig_other = any_neighbour_sig(&state, width, height, x, y);
                        let ctx = mr_context(has_sig_other, state[idx].refine);
                        let bit = mq.decode(&mut ctxs[ctx]);
                        trace(ctx, bit);
                        mag[idx] |= bit << bp;
                        state[idx].refine = true;
                    }
                }
                sy += 4;
            }
        } // end !first_plane (SPP + MRP)

        // ── Cleanup Pass ──────────────────────────────────────────────────────
        if passes_remaining == 0 {
            for s in &mut state {
                s.visit = false;
            }
            break;
        }
        passes_remaining -= 1;
        let mut y = 0;
        while y < height {
            let mut x = 0;
            while x < width {
                // Check run-length condition: 4 consecutive rows in this column,
                // all non-sig, non-visited, and no significant neighbours.
                let can_rlc = y + 4 <= height
                    && (y..y + 4).all(|yy| {
                        let i = yy * width + x;
                        !state[i].sig
                            && !state[i].visit
                            && neighbour_sig_total(&state, width, height, x, yy) == 0
                    });

                if can_rlc {
                    // Decode aggregate bit: are all 4 samples non-significant?
                    let agg = mq.decode(&mut ctxs[CTX_AGG]);
                    trace(CTX_AGG, agg);
                    if agg == 0 {
                        // All 4 samples are non-significant at this bit-plane.
                        x += 1;
                        y += if x >= width { 4 } else { 0 };
                        x %= width;
                        continue;
                    }
                    // One of the 4 became significant: decode 2-bit position.
                    let pos0 = mq.decode(&mut ctxs[CTX_UNI]);
                    trace(CTX_UNI, pos0);
                    let pos1 = mq.decode(&mut ctxs[CTX_UNI]);
                    trace(CTX_UNI, pos1);
                    let run_pos = (pos0 << 1) | pos1; // row offset 0..3
                    for row_off in 0..4usize {
                        let yy = y + row_off;
                        let idx = yy * width + x;
                        if row_off < run_pos as usize {
                            // Non-significant
                        } else if row_off == run_pos as usize {
                            // This sample becomes significant.
                            mag[idx] |= 1 << bp;
                            state[idx].sig = true;
                            let (kh, kv) = sign_contributions(&state, width, height, x, yy);
                            let (sc_ctx, xor_bit) = sc_context(kh, kv);
                            let raw_sign = mq.decode(&mut ctxs[sc_ctx]);
                            trace(sc_ctx, raw_sign);
                            let sign_bit = raw_sign ^ xor_bit;
                            state[idx].sign = sign_bit != 0;
                        } else {
                            // Remaining samples: code normally via ZC.
                            if !state[idx].sig {
                                let (h, v, d) = neighbour_sig_counts(&state, width, height, x, yy);
                                let ctx = zc_context(orient, h, v, d);
                                let sig_bit = mq.decode(&mut ctxs[ctx]);
                                trace(ctx, sig_bit);
                                if sig_bit == 1 {
                                    mag[idx] |= 1 << bp;
                                    state[idx].sig = true;
                                    let (kh, kv) = sign_contributions(&state, width, height, x, yy);
                                    let (sc_ctx, xor_bit) = sc_context(kh, kv);
                                    let raw_sign = mq.decode(&mut ctxs[sc_ctx]);
                                    trace(sc_ctx, raw_sign);
                                    let sign_bit = raw_sign ^ xor_bit;
                                    state[idx].sign = sign_bit != 0;
                                }
                            }
                        }
                    }
                    x += 1;
                    y += if x >= width { 4 } else { 0 };
                    x %= width;
                    continue;
                }

                // Normal cleanup: decode each sample not yet sig/visited.
                for yy in y..height.min(y + 4) {
                    let idx = yy * width + x;
                    if state[idx].sig || state[idx].visit {
                        continue;
                    }
                    let (h, v, d) = neighbour_sig_counts(&state, width, height, x, yy);
                    let ctx = zc_context(orient, h, v, d);
                    let sig_bit = mq.decode(&mut ctxs[ctx]);
                    trace(ctx, sig_bit);
                    if sig_bit == 1 {
                        mag[idx] |= 1 << bp;
                        state[idx].sig = true;
                        let (kh, kv) = sign_contributions(&state, width, height, x, yy);
                        let (sc_ctx, xor_bit) = sc_context(kh, kv);
                        let raw_sign = mq.decode(&mut ctxs[sc_ctx]);
                        trace(sc_ctx, raw_sign);
                        let sign_bit = raw_sign ^ xor_bit;
                        state[idx].sign = sign_bit != 0;
                    }
                }
                x += 1;
            }
            y += 4;
        }

        // Clear visit flags for next bit-plane.
        for s in &mut state {
            s.visit = false;
        }
    }

    // Reconstruct signed integer samples from sign + magnitude.
    let samples = state
        .iter()
        .zip(mag.iter())
        .map(|(s, &m)| {
            if !s.sig {
                0i32
            } else {
                let v = m as i32;
                if s.sign {
                    -v
                } else {
                    v
                }
            }
        })
        .collect();

    DecodedBlock { samples }
}

// ── EBCOT encoder ─────────────────────────────────────────────────────────────

/// Encoded code-block data (bitstream + metadata for the tier-2 packet header).
pub struct EncodedBlock {
    /// MQ-coded bytes.
    pub bytes: Vec<u8>,
    /// Number of coded bit-planes (P in the packet header).
    pub num_bit_planes: u8,
    /// Total number of coding passes included.
    pub num_passes: u32,
}

/// Encode one EBCOT code-block.
///
/// `samples` are the DC-shifted integer sample values in row-major order.
/// Returns the compressed byte stream and metadata needed for tier-2.
pub fn encode_code_block(
    samples: &[i32],
    width: usize,
    height: usize,
    orient: SubbandOrientation,
) -> EncodedBlock {
    assert_eq!(
        samples.len(),
        width * height,
        "EBCOT encode: samples length must equal width × height"
    );

    // Determine sign and magnitude from DC-shifted samples.
    let n = width * height;
    let sign: Vec<bool> = samples.iter().map(|&v| v < 0).collect();
    let mag: Vec<u32> = samples.iter().map(|&v| v.unsigned_abs()).collect();

    let max_mag = *mag.iter().max().unwrap_or(&0);
    if max_mag == 0 {
        // All samples are zero: produce an empty bitstream.
        return EncodedBlock {
            bytes: vec![0u8; 2],
            num_bit_planes: 0,
            num_passes: 0,
        };
    }

    // Number of bit-planes = floor(log2(max_mag)) + 1.
    let num_bit_planes = (u32::BITS - max_mag.leading_zeros()) as u8;

    // Per-sample significance / refinement state (mirrors the decoder's view).
    let mut state = vec![SampleState::default(); n];

    // Assign sign from the input samples.
    for i in 0..n {
        state[i].sign = sign[i];
    }

    let mut mq = MqEncoder::new();
    let mut ctxs = initial_contexts();
    let mut total_passes = 0u32;

    // The first (MSB) plane is cleanup-only (ISO 15444-1 §D.4.1): SPP/MRP are
    // skipped, giving the standard pass count 3P − 2 for P coded bit-planes.
    for bp in (0..num_bit_planes as u32).rev() {
        let first_plane = bp + 1 == u32::from(num_bit_planes);
        if !first_plane {
            // ── SPP ──────────────────────────────────────────────────────────────
            // Stripe-oriented scan (ISO 15444-1 §D.2): 4-row stripes, columns
            // within each stripe, rows within each column.
            let mut sy = 0;
            while sy < height {
                for x in 0..width {
                    for y in sy..height.min(sy + 4) {
                        let idx = y * width + x;
                        if state[idx].sig || state[idx].visit {
                            continue;
                        }
                        let (h, v, d) = neighbour_sig_counts(&state, width, height, x, y);
                        if h + v + d == 0 {
                            continue;
                        }
                        state[idx].visit = true;
                        let sig_bit = (mag[idx] >> bp) & 1;
                        let ctx = zc_context(orient, h, v, d);
                        trace(ctx, sig_bit);
                        mq.encode(sig_bit, &mut ctxs[ctx]);
                        if sig_bit == 1 {
                            state[idx].sig = true;
                            let (kh, kv) = sign_contributions(&state, width, height, x, y);
                            let (sc_ctx, xor_bit) = sc_context(kh, kv);
                            {
                                let sb = u32::from(state[idx].sign) ^ xor_bit;
                                trace(sc_ctx, sb);
                                mq.encode(sb, &mut ctxs[sc_ctx]);
                            }
                        }
                    }
                }
                sy += 4;
            }
            total_passes += 1;

            // ── MRP ──────────────────────────────────────────────────────────────
            let mut sy = 0;
            while sy < height {
                for x in 0..width {
                    for y in sy..height.min(sy + 4) {
                        let idx = y * width + x;
                        if !state[idx].sig || state[idx].visit {
                            continue;
                        }
                        let has_sig_other = any_neighbour_sig(&state, width, height, x, y);
                        let ctx = mr_context(has_sig_other, state[idx].refine);
                        let bit = (mag[idx] >> bp) & 1;
                        trace(ctx, bit);
                        mq.encode(bit, &mut ctxs[ctx]);
                        state[idx].refine = true;
                    }
                }
                sy += 4;
            }
            total_passes += 1;
        } // end !first_plane (SPP + MRP)

        // ── CUP ──────────────────────────────────────────────────────────────
        let mut y = 0;
        while y < height {
            let mut x = 0;
            while x < width {
                let can_rlc = y + 4 <= height
                    && (y..y + 4).all(|yy| {
                        let i = yy * width + x;
                        !state[i].sig
                            && !state[i].visit
                            && neighbour_sig_total(&state, width, height, x, yy) == 0
                    });

                if can_rlc {
                    // Check if all 4 rows are zero at this bit-plane.
                    let all_zero = (y..y + 4).all(|yy| (mag[yy * width + x] >> bp) & 1 == 0);
                    trace(CTX_AGG, u32::from(!all_zero));
                    mq.encode(u32::from(!all_zero), &mut ctxs[CTX_AGG]);
                    if !all_zero {
                        // Find the first non-zero row.
                        let run_pos = (y..y + 4)
                            .position(|yy| (mag[yy * width + x] >> bp) & 1 == 1)
                            .unwrap_or(0) as u32;
                        trace(CTX_UNI, (run_pos >> 1) & 1);
                        mq.encode((run_pos >> 1) & 1, &mut ctxs[CTX_UNI]);
                        trace(CTX_UNI, run_pos & 1);
                        mq.encode(run_pos & 1, &mut ctxs[CTX_UNI]);

                        for row_off in 0..4usize {
                            let yy = y + row_off;
                            let idx = yy * width + x;
                            if row_off == run_pos as usize {
                                // Became significant: encode sign.
                                state[idx].sig = true;
                                let (kh, kv) = sign_contributions(&state, width, height, x, yy);
                                let (sc_ctx, xor_bit) = sc_context(kh, kv);
                                {
                                    let sb = u32::from(state[idx].sign) ^ xor_bit;
                                    trace(sc_ctx, sb);
                                    mq.encode(sb, &mut ctxs[sc_ctx]);
                                }
                            } else if row_off > run_pos as usize && !state[idx].sig {
                                let sig_bit = (mag[idx] >> bp) & 1;
                                let (h, v, d) = neighbour_sig_counts(&state, width, height, x, yy);
                                let ctx = zc_context(orient, h, v, d);
                                trace(ctx, sig_bit);
                                mq.encode(sig_bit, &mut ctxs[ctx]);
                                if sig_bit == 1 {
                                    state[idx].sig = true;
                                    let (kh, kv) = sign_contributions(&state, width, height, x, yy);
                                    let (sc_ctx, xor_bit) = sc_context(kh, kv);
                                    mq.encode(
                                        u32::from(state[idx].sign) ^ xor_bit,
                                        &mut ctxs[sc_ctx],
                                    );
                                }
                            }
                        }
                    }
                    x += 1;
                    y += if x >= width { 4 } else { 0 };
                    x %= width;
                    continue;
                }

                // Normal cleanup coding.
                for yy in y..height.min(y + 4) {
                    let idx = yy * width + x;
                    if state[idx].sig || state[idx].visit {
                        continue;
                    }
                    let sig_bit = (mag[idx] >> bp) & 1;
                    let (h, v, d) = neighbour_sig_counts(&state, width, height, x, yy);
                    let ctx = zc_context(orient, h, v, d);
                    trace(ctx, sig_bit);
                    mq.encode(sig_bit, &mut ctxs[ctx]);
                    if sig_bit == 1 {
                        state[idx].sig = true;
                        let (kh, kv) = sign_contributions(&state, width, height, x, yy);
                        let (sc_ctx, xor_bit) = sc_context(kh, kv);
                        {
                            let sb = u32::from(state[idx].sign) ^ xor_bit;
                            trace(sc_ctx, sb);
                            mq.encode(sb, &mut ctxs[sc_ctx]);
                        }
                    }
                }
                x += 1;
            }
            y += 4;
        }
        total_passes += 1;

        // Clear visit flags.
        for s in &mut state {
            s.visit = false;
        }
    }

    let bytes = mq.finish();
    EncodedBlock {
        bytes,
        num_bit_planes,
        num_passes: total_passes,
    }
}

// ── Neighbour utilities ───────────────────────────────────────────────────────

/// Count significant horizontal (H), vertical (V) and diagonal (D) neighbours.
#[inline]
fn neighbour_sig_counts(
    state: &[SampleState],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> (u32, u32, u32) {
    let mut h = 0u32;
    let mut v = 0u32;
    let mut d = 0u32;
    if x > 0 && state[y * width + x - 1].sig {
        h += 1;
    }
    if x + 1 < width && state[y * width + x + 1].sig {
        h += 1;
    }
    if y > 0 && state[(y - 1) * width + x].sig {
        v += 1;
    }
    if y + 1 < height && state[(y + 1) * width + x].sig {
        v += 1;
    }
    if x > 0 && y > 0 && state[(y - 1) * width + x - 1].sig {
        d += 1;
    }
    if x + 1 < width && y > 0 && state[(y - 1) * width + x + 1].sig {
        d += 1;
    }
    if x > 0 && y + 1 < height && state[(y + 1) * width + x - 1].sig {
        d += 1;
    }
    if x + 1 < width && y + 1 < height && state[(y + 1) * width + x + 1].sig {
        d += 1;
    }
    (h, v, d)
}

/// Total significant neighbour count (H + V + D).
#[inline]
fn neighbour_sig_total(
    state: &[SampleState],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> u32 {
    let (h, v, d) = neighbour_sig_counts(state, width, height, x, y);
    h + v + d
}

/// `true` if any of the 8 neighbours is significant (for MR context).
#[inline]
fn any_neighbour_sig(
    state: &[SampleState],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> bool {
    neighbour_sig_total(state, width, height, x, y) > 0
}

/// Net sign contributions κ_h and κ_v (ISO 15444-1 Table D.3).
///
/// Each significant horizontal/vertical neighbour contributes +1 (positive)
/// or −1 (negative) based on its sign.  κ = signum(sum of contributions).
#[inline]
fn sign_contributions(
    state: &[SampleState],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> (i32, i32) {
    let mut h_raw = 0i32;
    let mut v_raw = 0i32;

    if x > 0 {
        let s = &state[y * width + x - 1];
        if s.sig {
            h_raw += if s.sign { -1 } else { 1 };
        }
    }
    if x + 1 < width {
        let s = &state[y * width + x + 1];
        if s.sig {
            h_raw += if s.sign { -1 } else { 1 };
        }
    }
    if y > 0 {
        let s = &state[(y - 1) * width + x];
        if s.sig {
            v_raw += if s.sign { -1 } else { 1 };
        }
    }
    if y + 1 < height {
        let s = &state[(y + 1) * width + x];
        if s.sig {
            v_raw += if s.sign { -1 } else { 1 };
        }
    }

    (h_raw.signum(), v_raw.signum())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_dec_roundtrip(samples: &[i32], w: usize, h: usize, orient: SubbandOrientation) {
        let enc = encode_code_block(samples, w, h, orient);
        let dec = decode_code_block(&enc.bytes, w, h, enc.num_bit_planes, enc.num_passes, orient);
        assert_eq!(dec.samples.len(), samples.len());
        for (i, (&orig, &decoded)) in samples.iter().zip(dec.samples.iter()).enumerate() {
            assert_eq!(
                decoded, orig,
                "sample[{i}] round-trip: expected {orig}, got {decoded}"
            );
        }
    }

    #[test]
    fn ebcot_all_zeros_round_trip() {
        enc_dec_roundtrip(&[0i32; 16], 4, 4, SubbandOrientation::LlOrLh);
    }

    #[test]
    fn ebcot_uniform_positive_round_trip() {
        let samples = vec![50i32; 16];
        enc_dec_roundtrip(&samples, 4, 4, SubbandOrientation::LlOrLh);
    }

    #[test]
    fn ebcot_gradient_round_trip() {
        let samples: Vec<i32> = (0..16).map(|i| i - 8).collect();
        enc_dec_roundtrip(&samples, 4, 4, SubbandOrientation::LlOrLh);
    }

    #[test]
    fn ebcot_signed_mixed_round_trip() {
        let samples = vec![-128i32, -1, 0, 127];
        enc_dec_roundtrip(&samples, 2, 2, SubbandOrientation::LlOrLh);
    }

    #[test]
    fn ebcot_1x1_nonzero_round_trip() {
        enc_dec_roundtrip(&[42i32], 1, 1, SubbandOrientation::LlOrLh);
    }

    /// Mirror of the encoder's symbol generation for a 1-row block, used to
    /// drive the MQ pair directly and localise encoder/decoder divergence.
    fn trace_encode_symbols(samples: &[i32], w: usize) -> Vec<(u32, usize)> {
        let h = 1usize;
        let n = w;
        let sign: Vec<bool> = samples.iter().map(|&v| v < 0).collect();
        let mag: Vec<u32> = samples.iter().map(|&v| v.unsigned_abs()).collect();
        let max_mag = *mag.iter().max().unwrap();
        let num_bit_planes = u32::BITS - max_mag.leading_zeros();
        let mut state = vec![SampleState::default(); n];
        for i in 0..n {
            state[i].sign = sign[i];
        }
        let mut out = Vec::new();
        for bp in (0..num_bit_planes).rev() {
            // SPP
            for x in 0..w {
                let idx = x;
                if state[idx].sig || state[idx].visit {
                    continue;
                }
                let (hh, vv, dd) = neighbour_sig_counts(&state, w, h, x, 0);
                if hh + vv + dd == 0 {
                    continue;
                }
                state[idx].visit = true;
                let sig_bit = (mag[idx] >> bp) & 1;
                out.push((sig_bit, zc_context(SubbandOrientation::LlOrLh, hh, vv, dd)));
                if sig_bit == 1 {
                    state[idx].sig = true;
                    let (kh, kv) = sign_contributions(&state, w, h, x, 0);
                    let (sc_ctx, xor_bit) = sc_context(kh, kv);
                    out.push((u32::from(state[idx].sign) ^ xor_bit, sc_ctx));
                }
            }
            // MRP
            for x in 0..w {
                let idx = x;
                if !state[idx].sig || state[idx].visit {
                    continue;
                }
                let has = any_neighbour_sig(&state, w, h, x, 0);
                let ctx = mr_context(has, state[idx].refine);
                out.push(((mag[idx] >> bp) & 1, ctx));
                state[idx].refine = true;
            }
            // CUP (height 1 → no RLC)
            for x in 0..w {
                let idx = x;
                if state[idx].sig || state[idx].visit {
                    continue;
                }
                let sig_bit = (mag[idx] >> bp) & 1;
                let (hh, vv, dd) = neighbour_sig_counts(&state, w, h, x, 0);
                out.push((sig_bit, zc_context(SubbandOrientation::LlOrLh, hh, vv, dd)));
                if sig_bit == 1 {
                    state[idx].sig = true;
                    let (kh, kv) = sign_contributions(&state, w, h, x, 0);
                    let (sc_ctx, xor_bit) = sc_context(kh, kv);
                    out.push((u32::from(state[idx].sign) ^ xor_bit, sc_ctx));
                }
            }
            for s in &mut state {
                s.visit = false;
            }
        }
        out
    }

    #[test]
    fn ebcot_1x7_symbol_trace_round_trips_at_mq_level() {
        // Drive the MQ pair with the exact EBCOT symbol/context trace of the
        // failing 1×7 input; isolates MQ-level loss from EBCOT divergence.
        let trace = trace_encode_symbols(&[51i32, 90, 124, 50, 69, 68, 8], 7);
        let mut enc_ctxs = initial_contexts();
        let mut enc = MqEncoder::new();
        for &(sym, ctx) in &trace {
            enc.encode(sym, &mut enc_ctxs[ctx]);
        }
        let _ = enc.finish();
        // Every prefix of the trace must round-trip exactly at the MQ level:
        // FLUSH termination must be lossless for any truncation point.
        for len in 1..=trace.len() {
            let prefix = &trace[..len];
            let mut e_ctxs = initial_contexts();
            let mut e = MqEncoder::new();
            for &(sym, ctx) in prefix {
                e.encode(sym, &mut e_ctxs[ctx]);
            }
            let bytes = e.finish();
            let mut d_ctxs = initial_contexts();
            let mut d = MqDecoder::new(&bytes);
            for (i, &(expected, ctx)) in prefix.iter().enumerate() {
                let got = d.decode(&mut d_ctxs[ctx]);
                assert_eq!(
                    got, expected,
                    "prefix len {len}: symbol[{i}] ctx={ctx} bytes {bytes:02X?}"
                );
            }
        }
    }

    #[test]
    fn ebcot_1x7_tail_refinement_shifted_round_trip() {
        // Same data ×2: errors should move with the tail if flush is lossy.
        enc_dec_roundtrip(
            &[102i32, 180, 248, 100, 138, 136, 16],
            7,
            1,
            SubbandOrientation::LlOrLh,
        );
    }

    #[test]
    fn ebcot_1x7_tail_refinement_round_trip() {
        // Regression (proptest seed 3841344251786497144): the final-plane MRP
        // bits sit at the stream tail; a lossy MQ flush decoded LSB 0 → 1.
        enc_dec_roundtrip(
            &[51i32, 90, 124, 50, 69, 68, 8],
            7,
            1,
            SubbandOrientation::LlOrLh,
        );
    }

    #[test]
    fn zc_ll_lh_context_two_horizontal_gives_8() {
        assert_eq!(zc_ll_lh(2, 0, 0), CTX_ZC_BASE + 8);
    }

    #[test]
    fn zc_ll_lh_no_neighbours_gives_0() {
        assert_eq!(zc_ll_lh(0, 0, 0), CTX_ZC_BASE);
    }

    #[test]
    fn sc_context_both_positive_gives_context_13() {
        let (ctx, xor) = sc_context(1, 1);
        assert_eq!(ctx, CTX_SC_BASE + 4); // absolute 13
        assert_eq!(xor, 0);
    }

    #[test]
    fn sc_context_symmetric_xor_bit() {
        // (-1, -1) must equal (1, 1) with xor_bit flipped.
        let (ctx_pp, xor_pp) = sc_context(1, 1);
        let (ctx_nn, xor_nn) = sc_context(-1, -1);
        assert_eq!(ctx_pp, ctx_nn);
        assert_ne!(xor_pp, xor_nn);
    }
}
