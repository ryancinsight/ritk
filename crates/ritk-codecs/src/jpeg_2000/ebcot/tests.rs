use super::contexts::{
    mr_context, sc_context, zc_context, zc_ll_lh, SubbandOrientation, CTX_SC_BASE, CTX_ZC_BASE,
};
use super::*;
use crate::jpeg_2000::mq_coder::{initial_contexts, MqDecoder, MqEncoder};

#[test]
fn sample_state_packs_independent_flags_into_one_byte() {
    assert_eq!(std::mem::size_of::<SampleState>(), 1);

    let mut state = SampleState::default();
    state.set_significant();
    state.mark_negative();
    state.set_visited();
    state.set_refined();
    state.clear_visited();

    assert!(state.is_significant());
    assert!(state.is_negative());
    assert!(!state.was_visited());
    assert!(state.was_refined());
}

fn enc_dec_roundtrip(samples: &[i32], w: usize, h: usize, orient: SubbandOrientation) {
    let enc = encode_code_block(samples, w, h, orient);
    let dec = decode_code_block(&enc.bytes, w, h, enc.num_bit_planes, enc.num_passes, orient)
        .expect("encoded code-block must decode");
    assert_eq!(dec.samples.len(), samples.len());
    for (i, (&orig, &decoded)) in samples.iter().zip(dec.samples.iter()).enumerate() {
        assert_eq!(
            decoded, orig,
            "sample[{i}] round-trip: expected {orig}, got {decoded}"
        );
    }
}

#[test]
fn ebcot_rejects_empty_body_with_declared_pass() {
    let err = decode_code_block(&[], 1, 1, 1, 1, SubbandOrientation::LlOrLh)
        .expect_err("a declared coding pass requires entropy bytes");
    assert!(format!("{err:#}").contains("empty code-block body"));
}

#[test]
fn ebcot_rejects_body_exhaustion_before_declared_passes() {
    let samples: Vec<i32> = (0..4096)
        .map(|index| ((index * 7919) % 65_535) - 32_768)
        .collect();
    let encoded = encode_code_block(&samples, 64, 64, SubbandOrientation::LlOrLh);
    let err = decode_code_block(
        &encoded.bytes[..1],
        64,
        64,
        encoded.num_bit_planes,
        encoded.num_passes,
        SubbandOrientation::LlOrLh,
    )
    .expect_err("a truncated entropy body must not synthesize coefficients");
    assert!(format!("{err:#}").contains("ended before"));
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
fn ebcot_signed_16_bit_extrema_round_trip() {
    let samples = [-32_768, -16_384, -1, 0, 1, 16_384, 32_767, 42];
    enc_dec_roundtrip(&samples, 4, 2, SubbandOrientation::LlOrLh);
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
    let max_mag = samples
        .iter()
        .map(|sample| sample.unsigned_abs())
        .max()
        .expect("infallible: validated precondition");
    let num_bit_planes = u32::BITS - max_mag.leading_zeros();
    let mut state = vec![SampleState::default(); n];
    for i in 0..n {
        if samples[i] < 0 {
            state[i].mark_negative();
        }
    }
    let mut out = Vec::new();
    for bp in (0..num_bit_planes).rev() {
        // SPP
        for x in 0..w {
            let idx = x;
            if state[idx].is_significant() || state[idx].was_visited() {
                continue;
            }
            let (hh, vv, dd) = neighbour_sig_counts(&state, w, h, x, 0);
            if hh + vv + dd == 0 {
                continue;
            }
            state[idx].set_visited();
            let sig_bit = (samples[idx].unsigned_abs() >> bp) & 1;
            out.push((sig_bit, zc_context(SubbandOrientation::LlOrLh, hh, vv, dd)));
            if sig_bit == 1 {
                state[idx].set_significant();
                let (kh, kv) = sign_contributions(&state, w, h, x, 0);
                let (sc_ctx, xor_bit) = sc_context(kh, kv);
                out.push((u32::from(state[idx].is_negative()) ^ xor_bit, sc_ctx));
            }
        }
        // MRP
        for x in 0..w {
            let idx = x;
            if !state[idx].is_significant() || state[idx].was_visited() {
                continue;
            }
            let has = any_neighbour_sig(&state, w, h, x, 0);
            let ctx = mr_context(has, state[idx].was_refined());
            out.push(((samples[idx].unsigned_abs() >> bp) & 1, ctx));
            state[idx].set_refined();
        }
        // CUP (height 1 → no RLC)
        for x in 0..w {
            let idx = x;
            if state[idx].is_significant() || state[idx].was_visited() {
                continue;
            }
            let sig_bit = (samples[idx].unsigned_abs() >> bp) & 1;
            let (hh, vv, dd) = neighbour_sig_counts(&state, w, h, x, 0);
            out.push((sig_bit, zc_context(SubbandOrientation::LlOrLh, hh, vv, dd)));
            if sig_bit == 1 {
                state[idx].set_significant();
                let (kh, kv) = sign_contributions(&state, w, h, x, 0);
                let (sc_ctx, xor_bit) = sc_context(kh, kv);
                out.push((u32::from(state[idx].is_negative()) ^ xor_bit, sc_ctx));
            }
        }
        for s in &mut state {
            s.clear_visited();
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
