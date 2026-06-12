//! MQ arithmetic coder – encoder and decoder (ISO 15444-1 Annex C).
//!
//! # Specification (ISO 15444-1 §C.2–§C.3)
//! The MQ-coder is a binary adaptive arithmetic coder with 47 probability states
//! per context and 19 context labels.  State transitions follow Table C.2.
//!
//! The probability state table defines for each state:
//! - `Qe`: probability estimate of the Less-Probable Symbol (LPS).
//! - `NLPS` / `NMPS`: next state when an LPS / MPS is decoded.
//! - `SWITCH`: when `true`, the MPS sense is flipped on the LPS transition.
//!
//! # Decoder register representation
//! Following OpenJPEG's implementation, the code register `c` is kept as a
//! 32-bit integer where bits 16–31 hold the "Chigh" value compared against Qe.
//!
//! # Encoder register representation
//! The encoder accumulates the coded interval in `c` (32-bit, with overflow bits
//! flowing into output bytes via `byteout`).  The interval `a` starts at `0x8000`
//! and is kept in the range [Qe_min, 0xFFFF] after each renormalisation.

// ── Probability state table (ISO 15444-1 Table C.2) ─────────────────────────

/// One entry of the MQ probability state table.
///
/// Fields: `(Qe, NLPS, NMPS, SWITCH)`.
#[derive(Clone, Copy)]
struct QeEntry {
    qe: u16,
    nlps: u8,
    nmps: u8,
    switch_flag: bool,
}

#[rustfmt::skip]
const QE_TABLE: [QeEntry; 47] = [
    // idx  Qe       NMPS  NLPS  SWITCH (ISO 15444-1 Table C.2)
    QeEntry { qe: 0x5601, nmps:  1, nlps:  1, switch_flag: true  }, //  0
    QeEntry { qe: 0x3401, nmps:  2, nlps:  6, switch_flag: false }, //  1
    QeEntry { qe: 0x1801, nmps:  3, nlps:  9, switch_flag: false }, //  2
    QeEntry { qe: 0x0AC1, nmps:  4, nlps: 12, switch_flag: false }, //  3
    QeEntry { qe: 0x0521, nmps:  5, nlps: 29, switch_flag: false }, //  4
    QeEntry { qe: 0x0221, nmps: 38, nlps: 33, switch_flag: false }, //  5
    QeEntry { qe: 0x5601, nmps:  7, nlps:  6, switch_flag: true  }, //  6
    QeEntry { qe: 0x5401, nmps:  8, nlps: 14, switch_flag: false }, //  7
    QeEntry { qe: 0x4801, nmps:  9, nlps: 14, switch_flag: false }, //  8
    QeEntry { qe: 0x3801, nmps: 10, nlps: 14, switch_flag: false }, //  9
    QeEntry { qe: 0x3001, nmps: 11, nlps: 17, switch_flag: false }, // 10
    QeEntry { qe: 0x2401, nmps: 12, nlps: 18, switch_flag: false }, // 11
    QeEntry { qe: 0x1C01, nmps: 13, nlps: 20, switch_flag: false }, // 12
    QeEntry { qe: 0x1601, nmps: 29, nlps: 21, switch_flag: false }, // 13
    QeEntry { qe: 0x5601, nmps: 15, nlps: 14, switch_flag: true  }, // 14
    QeEntry { qe: 0x5401, nmps: 16, nlps: 14, switch_flag: false }, // 15
    QeEntry { qe: 0x5101, nmps: 17, nlps: 15, switch_flag: false }, // 16
    QeEntry { qe: 0x4801, nmps: 18, nlps: 16, switch_flag: false }, // 17
    QeEntry { qe: 0x3801, nmps: 19, nlps: 17, switch_flag: false }, // 18
    QeEntry { qe: 0x3401, nmps: 20, nlps: 18, switch_flag: false }, // 19
    QeEntry { qe: 0x3001, nmps: 21, nlps: 19, switch_flag: false }, // 20
    QeEntry { qe: 0x2801, nmps: 22, nlps: 19, switch_flag: false }, // 21
    QeEntry { qe: 0x2401, nmps: 23, nlps: 20, switch_flag: false }, // 22
    QeEntry { qe: 0x2201, nmps: 24, nlps: 21, switch_flag: false }, // 23
    QeEntry { qe: 0x1C01, nmps: 25, nlps: 22, switch_flag: false }, // 24
    QeEntry { qe: 0x1801, nmps: 26, nlps: 23, switch_flag: false }, // 25
    QeEntry { qe: 0x1601, nmps: 27, nlps: 24, switch_flag: false }, // 26
    QeEntry { qe: 0x1401, nmps: 28, nlps: 25, switch_flag: false }, // 27
    QeEntry { qe: 0x1201, nmps: 29, nlps: 26, switch_flag: false }, // 28
    QeEntry { qe: 0x1101, nmps: 30, nlps: 27, switch_flag: false }, // 29
    QeEntry { qe: 0x0AC1, nmps: 31, nlps: 28, switch_flag: false }, // 30
    QeEntry { qe: 0x09C1, nmps: 32, nlps: 29, switch_flag: false }, // 31
    QeEntry { qe: 0x08A1, nmps: 33, nlps: 30, switch_flag: false }, // 32
    QeEntry { qe: 0x0521, nmps: 34, nlps: 31, switch_flag: false }, // 33
    QeEntry { qe: 0x0441, nmps: 35, nlps: 32, switch_flag: false }, // 34
    QeEntry { qe: 0x02A1, nmps: 36, nlps: 33, switch_flag: false }, // 35
    QeEntry { qe: 0x0221, nmps: 37, nlps: 34, switch_flag: false }, // 36
    QeEntry { qe: 0x0141, nmps: 38, nlps: 35, switch_flag: false }, // 37
    QeEntry { qe: 0x0111, nmps: 39, nlps: 36, switch_flag: false }, // 38
    QeEntry { qe: 0x0085, nmps: 40, nlps: 37, switch_flag: false }, // 39
    QeEntry { qe: 0x0049, nmps: 41, nlps: 38, switch_flag: false }, // 40
    QeEntry { qe: 0x0025, nmps: 42, nlps: 39, switch_flag: false }, // 41
    QeEntry { qe: 0x0015, nmps: 43, nlps: 40, switch_flag: false }, // 42
    QeEntry { qe: 0x0009, nmps: 44, nlps: 41, switch_flag: false }, // 43
    QeEntry { qe: 0x0005, nmps: 45, nlps: 42, switch_flag: false }, // 44
    QeEntry { qe: 0x0001, nmps: 45, nlps: 43, switch_flag: false }, // 45
    QeEntry { qe: 0x5601, nmps: 46, nlps: 46, switch_flag: false }, // 46
];

// ── Context state ─────────────────────────────────────────────────────────────

/// Number of MQ contexts used by EBCOT.
pub const NUM_CONTEXTS: usize = 19;

/// Per-context probability state (state-table index + current MPS value).
#[derive(Clone, Copy, Debug, Default)]
pub struct CtxState {
    /// Index into `QE_TABLE` (0–46).
    pub state: u8,
    /// Current Most-Probable Symbol (0 or 1).
    pub mps: u8,
}

/// Create the 19-context initial state array (ISO 15444-1 Table D.7).
///
/// All contexts start at state 0 except: ZC context 0 → state 4,
/// run-length (AGG) context → state 3, uniform (UNI) context → state 46.
pub fn initial_contexts() -> [CtxState; NUM_CONTEXTS] {
    let mut ctx = [CtxState::default(); NUM_CONTEXTS];
    // ZC context 0 (all-non-significant neighbourhood).
    ctx[0] = CtxState { state: 4, mps: 0 };
    // UNI context (17): near-uniform probability.
    ctx[17] = CtxState { state: 46, mps: 0 };
    // AGG / RLC context (18).
    ctx[18] = CtxState { state: 3, mps: 0 };
    ctx
}

// ── MQ Decoder ────────────────────────────────────────────────────────────────

/// MQ arithmetic decoder (ISO 15444-1 §C.3).
///
/// The `c` register stores the residual code fraction in bits 16–31 (Chigh),
/// matching the representation used by OpenJPEG.
pub struct MqDecoder<'a> {
    src: &'a [u8],
    /// Next byte position in `src`.
    pos: usize,
    /// Last byte consumed (used by the byte-stuffing logic in `bytein`).
    prev: u8,
    /// Interval register (lower 16 bits significant).
    a: u32,
    /// Code register (bits 16–31 = Chigh).
    c: u32,
    /// Remaining bit-shifts available before the next `bytein`.
    ct: u32,
}

impl<'a> MqDecoder<'a> {
    /// Initialise the MQ decoder over `src` (ISO 15444-1 §C.3.2 INITDEC).
    ///
    /// `C = B0 << 16; BYTEIN; C <<= 7; CT -= 7; A = 0x8000` — matching
    /// OpenJPEG `opj_mqc_init_dec`.
    pub fn new(src: &'a [u8]) -> Self {
        let b0 = src.first().copied().unwrap_or(0xFF);
        let mut d = Self {
            src,
            pos: usize::from(!src.is_empty()),
            prev: b0,
            a: 0,
            c: u32::from(b0) << 16,
            ct: 0,
        };
        d.bytein();
        d.c <<= 7;
        d.ct -= 7;
        d.a = 0x8000;
        d
    }

    /// Decode one binary symbol using the given context (ISO 15444-1 §C.3.3 DECODE).
    ///
    /// Returns 0 or 1.
    #[inline]
    pub fn decode(&mut self, ctx: &mut CtxState) -> u32 {
        let qe = u32::from(QE_TABLE[ctx.state as usize].qe);
        self.a -= qe;
        if (self.c >> 16) < qe {
            // Code fell in the LPS interval.
            let d = self.lps_exchange(ctx, qe);
            self.renormd();
            d
        } else {
            self.c -= qe << 16;
            if self.a & 0x8000 != 0 {
                // Already normalised; MPS decoded without renorm.
                ctx.state = QE_TABLE[ctx.state as usize].nmps;
                u32::from(ctx.mps)
            } else {
                // MPS but renorm needed (possible exchange).
                let d = self.mps_exchange(ctx, qe);
                self.renormd();
                d
            }
        }
    }

    /// Test-only register introspection: `(a, c, ct)`.
    #[cfg(test)]
    pub(crate) fn registers(&self) -> (u32, u32, u32) {
        (self.a, self.c, self.ct)
    }

    // ── private helpers ──────────────────────────────────────────────────────

    #[inline]
    fn lps_exchange(&mut self, ctx: &mut CtxState, qe: u32) -> u32 {
        let entry = QE_TABLE[ctx.state as usize];
        let d;
        if self.a < qe {
            // Exchange: MPS interval < Qe → MPS takes the smaller interval.
            // The decoded symbol is returned as MPS (exchange condition).
            d = u32::from(ctx.mps);
            ctx.state = entry.nmps;
        } else {
            // Normal LPS.
            d = u32::from(1 - ctx.mps);
            if entry.switch_flag {
                ctx.mps = 1 - ctx.mps;
            }
            ctx.state = entry.nlps;
        }
        // ISO 15444-1 §C.3.3: A = Qe[I] where I is the ORIGINAL context index
        // (before state transition) — matching OpenJPEG's `mqc->a = st->qeval`.
        self.a = qe;
        d
    }

    #[inline]
    fn mps_exchange(&mut self, ctx: &mut CtxState, qe: u32) -> u32 {
        let entry = QE_TABLE[ctx.state as usize];
        let d;
        if self.a < qe {
            // Exchange: MPS interval < Qe.
            d = u32::from(1 - ctx.mps);
            if entry.switch_flag {
                ctx.mps = 1 - ctx.mps;
            }
            ctx.state = entry.nlps;
        } else {
            d = u32::from(ctx.mps);
            ctx.state = entry.nmps;
        }
        // MPSEXCHANGE (ISO 15444-1 Figure C.18) does not modify A.
        d
    }

    #[inline]
    fn renormd(&mut self) {
        loop {
            if self.ct == 0 {
                self.bytein();
            }
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }

    /// Read the next byte from `src` into the code register (ISO 15444-1 §C.3.4 BYTEIN).
    ///
    /// Byte stuffing: 0xFF followed by a byte T ≤ 0x8F → T is stuffed (7 bits effective).
    /// Marker escape: 0xFF followed by T > 0x8F → treat the byte as a fill value of 0xFF.
    #[inline]
    fn bytein(&mut self) {
        if self.prev == 0xFF {
            let t = if self.pos < self.src.len() {
                self.src[self.pos]
            } else {
                0xFF
            };
            if t > 0x8F {
                // Marker or end-of-data: add 0xFF00 without advancing.
                self.c += 0xFF00;
                self.ct = 8;
            } else {
                // Stuffed byte: advance, load with 7-bit effective contribution.
                self.pos += 1;
                self.prev = t;
                self.c += u32::from(t) << 9;
                self.ct = 7;
            }
        } else {
            let b = if self.pos < self.src.len() {
                self.src[self.pos]
            } else {
                0xFF
            };
            self.pos += 1;
            self.prev = b;
            self.c += u32::from(b) << 8;
            self.ct = 8;
        }
    }
}

// ── MQ Encoder ────────────────────────────────────────────────────────────────

/// MQ arithmetic encoder (ISO 15444-1 §C.2).
pub struct MqEncoder {
    /// Output byte buffer.
    out: Vec<u8>,
    /// Interval register.
    a: u32,
    /// Code register (overflow propagates into `out`).
    c: u32,
    /// Bit-shifts available before the next `byteout`.
    ct: u32,
    /// Pending output byte (not yet written to `out`).
    b: u32,
    /// `true` until the first `byteout` commit: the initial pending byte is a
    /// dummy (OpenJPEG's `bp = start - 1` slot) and is discarded, not emitted.
    first: bool,
}

impl Default for MqEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl MqEncoder {
    /// Create a new MQ encoder (ISO 15444-1 §C.2.7 INITENC).
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            a: 0x8000,
            c: 0,
            ct: 12,
            b: 0,
            first: true,
        }
    }

    /// Encode one binary symbol `d` (0 or 1) using the given context.
    pub fn encode(&mut self, d: u32, ctx: &mut CtxState) {
        if u32::from(ctx.mps) == d {
            self.mps_encode(ctx);
        } else {
            self.lps_encode(ctx);
        }
    }

    /// Flush the remaining bits to `out` and return the encoded bytes
    /// (ISO 15444-1 §C.2.9 FLUSH).
    pub fn finish(mut self) -> Vec<u8> {
        // ISO 15444-1 §C.2.9 FLUSH (= OpenJPEG `opj_mqc_flush`):
        // SETBITS; C <<= CT; BYTEOUT; C <<= CT; BYTEOUT — the second shift uses
        // the CT set by the first BYTEOUT (7 after a stuffed byte, else 8); a
        // fixed 8 misaligns the final bits. The final pending byte is included
        // only when it is not 0xFF (OpenJPEG `if (*bp != 0xff) bp++`); pushing
        // it unconditionally breaks tail decode (regression test:
        // ebcot_1x7_tail_refinement_round_trip).
        self.set_bits();
        self.c <<= self.ct;
        self.byteout();
        self.c <<= self.ct;
        self.byteout();
        if self.b != 0xFF {
            self.commit();
        }
        self.out
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// CODEMPS (ISO 15444-1 Figure C.7).
    ///
    /// The MPS sub-interval sits above the LPS sub-interval: coding an MPS
    /// normally adds `Qe` to `C`. When renormalisation is required, the
    /// conditional-exchange test `A < Qe` decides which sub-interval the MPS
    /// takes.
    #[inline]
    fn mps_encode(&mut self, ctx: &mut CtxState) {
        let entry = QE_TABLE[ctx.state as usize];
        let qe = u32::from(entry.qe);
        self.a -= qe;
        if self.a & 0x8000 != 0 {
            // Still normalised: MPS takes the upper interval.
            self.c += qe;
            ctx.state = entry.nmps;
        } else {
            if self.a < qe {
                // Conditional exchange: MPS takes the lower (Qe) interval.
                self.a = qe;
            } else {
                self.c += qe;
            }
            ctx.state = entry.nmps;
            self.renorme();
        }
    }

    /// CODELPS (ISO 15444-1 Figure C.8).
    ///
    /// Exchange (A < Qe): C += Qe so the code lands in the MPS region;
    /// the DECODER routes via `mps_exchange` exchange → returns 1−mps. ✓
    /// No-exchange (A ≥ Qe): C unchanged; code stays in the LPS region;
    /// the DECODER routes via `lps_exchange` no-exchange → returns 1−mps. ✓
    /// Both paths: A = Qe (synchronises renorm depth with the decoder), state → NLPS,
    /// flip MPS when SWITCH is set.
    #[inline]
    fn lps_encode(&mut self, ctx: &mut CtxState) {
        let entry = QE_TABLE[ctx.state as usize];
        let qe = u32::from(entry.qe);
        self.a -= qe;
        if self.a < qe {
            // Conditional exchange (Figure C.8): LPS takes the upper interval;
            // A keeps its reduced value A − Qe. Setting A = Qe here desyncs the
            // decoder (regression test: ebcot_1x7_tail_refinement_round_trip).
            self.c += qe;
        } else {
            self.a = qe;
        }
        ctx.state = entry.nlps;
        if entry.switch_flag {
            ctx.mps = 1 - ctx.mps;
        }
        self.renorme();
    }

    #[inline]
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

    /// Test-only register introspection: `(a, c, ct, b)`.
    #[cfg(test)]
    pub(crate) fn enc_registers(&self) -> (u32, u32, u32, u32) {
        (self.a, self.c, self.ct, self.b)
    }

    /// Commit the pending byte to `out`; the initial dummy byte is discarded
    /// (OpenJPEG's write slot at `start - 1`).
    #[inline]
    fn commit(&mut self) {
        if self.first {
            self.first = false;
        } else {
            self.out.push(self.b as u8);
        }
    }

    /// Output a compressed byte to `out`, handling carry propagation and
    /// byte-stuffing of 0xFF (ISO 15444-1 §C.2.4 BYTEOUT, = OpenJPEG
    /// `opj_mqc_byteout`).
    #[inline]
    fn byteout(&mut self) {
        if self.b == 0xFF {
            // Previous byte is 0xFF: stuffed byte carries only 7 bits.
            self.commit();
            self.b = self.c >> 20;
            self.c &= 0x000F_FFFF;
            self.ct = 7;
        } else if self.c & 0x0800_0000 == 0 {
            // No carry.
            self.commit();
            self.b = (self.c >> 19) & 0xFF;
            self.c &= 0x0007_FFFF;
            self.ct = 8;
        } else {
            // Carry: increment the pending byte before committing it.
            self.b += 1;
            if self.b == 0xFF {
                self.c &= 0x07FF_FFFF;
                self.commit();
                self.b = self.c >> 20;
                self.c &= 0x000F_FFFF;
                self.ct = 7;
            } else {
                self.commit();
                self.b = (self.c >> 19) & 0xFF;
                self.c &= 0x0007_FFFF;
                self.ct = 8;
            }
        }
    }

    /// Prepare the interval end-bits before flushing (ISO 15444-1 §C.2.9 SETBITS).
    fn set_bits(&mut self) {
        let temp_length = self.a + self.c;
        self.c |= 0x0000_FFFF;
        if self.c >= temp_length {
            self.c -= 0x8000;
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
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
        // 2.5.2 (8×8, prec 8, numres 1).
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
        out: Vec<u8>,
    }

    impl RefMq {
        fn new() -> Self {
            Self {
                a: 0x8000,
                c: 0,
                ct: 12,
                b: 0,
                first: true,
                out: Vec::new(),
            }
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
                    self.c += qe;
                    ctx.state = entry.nmps;
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
}
