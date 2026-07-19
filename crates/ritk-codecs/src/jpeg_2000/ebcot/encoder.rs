use super::contexts::{mr_context, sc_context, zc_context, SubbandOrientation, CTX_AGG, CTX_UNI};
use super::{
    any_neighbour_sig, neighbour_sig_counts, neighbour_sig_total, sign_contributions, trace,
    SampleState,
};
use crate::jpeg_2000::mq_coder::{initial_contexts, MqEncoder};

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
                                    let sb = u32::from(state[idx].sign) ^ xor_bit;
                                    trace(sc_ctx, sb);
                                    mq.encode(sb, &mut ctxs[sc_ctx]);
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
