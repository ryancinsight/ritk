use super::*;

/// Encode `symbols` with a fresh context array, then decode them back.
/// All symbols must reconstruct to the original values exactly.
fn round_trip(symbols: &[(u32, usize)]) {
    let mut enc_ctxs = initial_contexts();
    let mut enc = MqEncoder::new();
    for &(sym, ctx_idx) in symbols {
        enc.encode(sym, &mut enc_ctxs[ctx_idx]);
    }
    let bytes = enc.finish();

    let mut dec_ctxs = initial_contexts();
    let mut dec = MqDecoder::new(&bytes);
    for (i, &(expected, ctx_idx)) in symbols.iter().enumerate() {
        let got = dec.decode(&mut dec_ctxs[ctx_idx]);
        assert_eq!(
            got, expected,
            "symbol[{i}] ctx={ctx_idx}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn mq_round_trip_uniform_zeros() {
    // 64 zeros on context 0 (MPS=0 path throughout).
    let symbols: Vec<(u32, usize)> = (0..64).map(|_| (0, 0)).collect();
    round_trip(&symbols);
}

#[test]
fn mq_round_trip_alternating() {
    let symbols: Vec<(u32, usize)> = (0..32).map(|i| (i % 2, 0)).collect();
    round_trip(&symbols);
}

#[test]
fn mq_round_trip_all_ones_ctx_17() {
    // Context 17 = UNI (near-uniform).
    let symbols: Vec<(u32, usize)> = (0..64).map(|_| (1, 17)).collect();
    round_trip(&symbols);
}

#[test]
fn mq_round_trip_mixed_contexts() {
    let symbols: Vec<(u32, usize)> = (0..100)
        .map(|i| (((i * 7 + 3) % 2) as u32, i % NUM_CONTEXTS))
        .collect();
    round_trip(&symbols);
}

proptest::proptest! {
    /// Any (symbol, context) sequence must round-trip exactly through the
    /// MQ encoder/decoder pair (flush termination must be lossless).
    #[test]
    fn mq_round_trip_random(seq in proptest::collection::vec((0u32..2, 0usize..NUM_CONTEXTS), 1..200)) {
        let mut enc_ctxs = initial_contexts();
        let mut enc = MqEncoder::new();
        for &(sym, ctx_idx) in &seq {
            enc.encode(sym, &mut enc_ctxs[ctx_idx]);
        }
        let bytes = enc.finish();
        let mut dec_ctxs = initial_contexts();
        let mut dec = MqDecoder::new(&bytes);
        for (i, &(expected, ctx_idx)) in seq.iter().enumerate() {
            let got = dec.decode(&mut dec_ctxs[ctx_idx]);
            proptest::prop_assert_eq!(got, expected, "symbol[{}] ctx={}", i, ctx_idx);
        }
    }
}

#[test]
fn trace_openjp2_reference_prefix() {
    // First bytes of the 62-byte code-block body captured from OpenJPEG
    // 2.5.2 (8Ã—8, prec 8, numres 1).
    let body: [u8; 12] = [
        0x12, 0x51, 0x7A, 0x62, 0x3E, 0xFC, 0x7B, 0x8E, 0x3E, 0x6C, 0xBF, 0x33,
    ];
    // Intended symbol/context prefix per the EBCOT cleanup algorithm.
    let prefix: [(usize, u32); 10] = [
        (18, 1),
        (17, 0),
        (17, 0),
        (9, 1),
        (3, 0),
        (0, 0),
        (0, 0),
        (5, 1),
        (12, 0),
        (3, 1),
    ];
    let mut ctxs = initial_contexts();
    let mut dec = MqDecoder::new(&body);
    for (i, &(ctx, expect)) in prefix.iter().enumerate() {
        let before = dec.registers();
        let st = (ctxs[ctx].state, ctxs[ctx].mps);
        let got = dec.decode(&mut ctxs[ctx]);
        eprintln!(
            "sym{i:2} ctx={ctx:2} st={st:?} before(a={:04X},chigh={:04X},ct={}) got={got} expect={expect}",
            before.0,
            before.1 >> 16,
            before.2
        );
    }
}

/// Verbatim port of OpenJPEG's `opj_mqc_*` encoder (mqc.c) used purely as
/// an in-test reference for register-level differential comparison.
struct RefMq {
    a: u32,
    c: u32,
    ct: u32,
    b: u32,
    first: bool,
    out: Vec<u8> }

impl RefMq {
    fn new() -> Self {
        Self {
            a: 0x8000,
            c: 0,
            ct: 12,
            b: 0,
            first: true,
            out: Vec::new() }
    }
    fn commit(&mut self) {
        if self.first {
            self.first = false;
        } else {
            self.out.push(self.b as u8);
        }
    }
    fn byteout(&mut self) {
        if self.b == 0xFF {
            self.commit();
            self.b = self.c >> 20;
            self.c &= 0xF_FFFF;
            self.ct = 7;
        } else if self.c & 0x800_0000 == 0 {
            self.commit();
            self.b = (self.c >> 19) & 0xFF;
            self.c &= 0x7_FFFF;
            self.ct = 8;
        } else {
            self.b += 1;
            if self.b == 0xFF {
                self.c &= 0x7FF_FFFF;
                self.commit();
                self.b = self.c >> 20;
                self.c &= 0xF_FFFF;
                self.ct = 7;
            } else {
                self.commit();
                self.b = (self.c >> 19) & 0xFF;
                self.c &= 0x7_FFFF;
                self.ct = 8;
            }
        }
    }
    fn renorme(&mut self) {
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                self.byteout();
            }
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }
    fn encode(&mut self, d: u32, ctx: &mut CtxState) {
        let entry = QE_TABLE[ctx.state as usize];
        let qe = u32::from(entry.qe);
        if u32::from(ctx.mps) == d {
            // codemps
            self.a -= qe;
            if self.a & 0x8000 == 0 {
                if self.a < qe {
                    self.a = qe;
                } else {
                    self.c += qe;
                }
                ctx.state = entry.nmps;
                self.renorme();
            } else {
                // No renorm â†’ no state transition (opj_mqc_codemps).
                self.c += qe;
            }
        } else {
            // codelps
            self.a -= qe;
            if self.a < qe {
                self.c += qe;
            } else {
                self.a = qe;
            }
            if entry.switch_flag {
                ctx.mps = 1 - ctx.mps;
            }
            ctx.state = entry.nlps;
            self.renorme();
        }
    }
}

#[test]
fn reference_port_register_differential() {
    let prefix: [(usize, u32); 10] = [
        (18, 1),
        (17, 0),
        (17, 0),
        (9, 1),
        (3, 0),
        (0, 0),
        (0, 0),
        (5, 1),
        (12, 0),
        (3, 1),
    ];
    let mut ours_ctx = initial_contexts();
    let mut refs_ctx = initial_contexts();
    let mut ours = MqEncoder::new();
    let mut refs = RefMq::new();
    for (i, &(ctx, bit)) in prefix.iter().enumerate() {
        ours.encode(bit, &mut ours_ctx[ctx]);
        refs.encode(bit, &mut refs_ctx[ctx]);
        let o = ours.enc_registers();
        eprintln!(
            "sym{i:2} ours(a={:04X} c={:07X} ct={:2} b={:02X}) ref(a={:04X} c={:07X} ct={:2} b={:02X})",
            o.0, o.1, o.2, o.3, refs.a, refs.c, refs.ct, refs.b
        );
        assert_eq!(
            (o.0, o.1, o.2, o.3),
            (refs.a, refs.c, refs.ct, refs.b),
            "register divergence at symbol {i}"
        );
    }
}

#[test]
fn trace_v1_mid_reference() {
    // OpenJPEG body for the +1 impulse at (4,4) of an 8Ã—8 block (single
    // cleanup pass). Our cleanup script: stripe0 = 8 AGG zeros; stripe1 =
    // 4 AGG zeros, col4 RLC (AGG, UNI, UNI, sign), col4 rows 5-7 ZC,
    // col5 4 ZC, cols 6-7 AGG zeros.
    let body: [u8; 2] = [0x50, 0x6F];
    let script: [usize; 25] = [
        18, 18, 18, 18, 18, 18, 18, 18, // stripe 0
        18, 18, 18, 18, // stripe 1 cols 0-3
        18, 17, 17, 9, // col 4 RLC + sign
        3, 0, 0, // col 4 rows 5-7
        5, 1, 0, 0, // col 5
        18, 18, // cols 6-7
    ];
    let ours: [u32; 25] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let mut ctxs = initial_contexts();
    let mut dec = MqDecoder::new(&body);
    for (i, (&ctx, &expect)) in script.iter().zip(ours.iter()).enumerate() {
        let st = (ctxs[ctx].state, ctxs[ctx].mps);
        let got = dec.decode(&mut ctxs[ctx]);
        let mark = if got == expect { ' ' } else { '*' };
        eprintln!("sym{i:2} ctx={ctx:2} st={st:?} got={got} ours={expect} {mark}");
    }
}

#[test]
fn qe_table_state_0_has_uniform_probability() {
    assert_eq!(QE_TABLE[0].qe, 0x5601, "state 0 Qe = 0x5601");
    assert!(QE_TABLE[0].switch_flag, "state 0 has SWITCH = true");
}

#[test]
fn qe_table_state_45_is_nearly_certain() {
    assert_eq!(
        QE_TABLE[45].qe, 0x0001,
        "state 45 Qe = 0x0001 (lowest LPS probability)"
    );
}

#[test]
fn initial_contexts_uni_is_state_46() {
    let ctx = initial_contexts();
    assert_eq!(ctx[17].state, 46, "UNI context initialised to state 46");
}
