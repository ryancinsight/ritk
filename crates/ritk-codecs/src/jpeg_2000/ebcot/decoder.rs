use super::contexts::{mr_context, sc_context, zc_context, SubbandOrientation, CTX_AGG, CTX_UNI};
use super::{
    any_neighbour_sig, neighbour_sig_counts, neighbour_sig_total, sign_contributions, trace,
    SampleState,
};
use crate::jpeg_2000::mq_coder::{initial_contexts, MqDecoder};

// ── Decoded code-block result ─────────────────────────────────────────────────

/// Output of `decode_code_block`: reconstructed sign-magnitude samples in
/// row-major order.  Values are the DC-shifted integer sample values (i32).
/// The caller applies DC un-shift and DICOM rescale.
pub struct DecodedBlock {
    /// Reconstructed samples (row-major), in the order they appear in the tile.
    pub samples: Vec<i32>,
    /// Index of the lowest bit-plane reached by any executed coding pass
    /// (`0` when the block decoded down to the LSB, i.e. fully decoded).
    /// A truncated block stops at a higher plane, so the magnitudes carry zero
    /// in bits `0..lowest_bitplane`; the irreversible reconstruction uses this
    /// to place the dequantized value at the midpoint of the still-undecoded
    /// interval (ISO 15444-1 §E.1.1.2) rather than a fixed half-step.
    pub lowest_bitplane: u32,
}

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
            lowest_bitplane: num_bit_planes as u32,
        };
    }

    let mut mq = MqDecoder::new(data);
    let mut ctxs = initial_contexts();

    let total_bit_planes = num_bit_planes as u32;
    let mut passes_remaining = num_passes;
    // Lowest bit-plane index touched by an executed pass; starts above the MSB
    // and descends to the plane of the last pass run (0 ⟺ fully decoded).
    let mut lowest_bp = total_bit_planes;

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
            lowest_bp = bp;
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
            lowest_bp = bp;
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
        lowest_bp = bp;
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

    DecodedBlock {
        samples,
        lowest_bitplane: lowest_bp,
    }
}
