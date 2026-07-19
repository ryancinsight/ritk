use super::{CtxState, QE_TABLE};

/// MQ arithmetic encoder (ISO 15444-1 Â§C.2).
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
    /// Create a new MQ encoder (ISO 15444-1 Â§C.2.7 INITENC).
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
    /// (ISO 15444-1 Â§C.2.9 FLUSH).
    pub fn finish(mut self) -> Vec<u8> {
        // ISO 15444-1 Â§C.2.9 FLUSH (= OpenJPEG `opj_mqc_flush`):
        // SETBITS; C <<= CT; BYTEOUT; C <<= CT; BYTEOUT â€” the second shift uses
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

    // â”€â”€ private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            // Still normalised: MPS takes the upper interval. No state
            // transition â€” ISO 15444-1 Figure C.7 advances I(CX) only on the
            // renormalisation path (probability estimation is renorm-driven,
            // Â§C.2.6); matching OpenJPEG `opj_mqc_codemps`.
            self.c += qe;
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
    /// the DECODER routes via `mps_exchange` exchange â†’ returns 1âˆ’mps. âœ“
    /// No-exchange (A â‰¥ Qe): C unchanged; code stays in the LPS region;
    /// the DECODER routes via `lps_exchange` no-exchange â†’ returns 1âˆ’mps. âœ“
    /// Both paths: A = Qe (synchronises renorm depth with the decoder), state â†’ NLPS,
    /// flip MPS when SWITCH is set.
    #[inline]
    fn lps_encode(&mut self, ctx: &mut CtxState) {
        let entry = QE_TABLE[ctx.state as usize];
        let qe = u32::from(entry.qe);
        self.a -= qe;
        if self.a < qe {
            // Conditional exchange (Figure C.8): LPS takes the upper interval;
            // A keeps its reduced value A âˆ’ Qe. Setting A = Qe here desyncs the
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
    /// byte-stuffing of 0xFF (ISO 15444-1 Â§C.2.4 BYTEOUT, = OpenJPEG
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

    /// Prepare the interval end-bits before flushing (ISO 15444-1 Â§C.2.9 SETBITS).
    fn set_bits(&mut self) {
        let temp_length = self.a + self.c;
        self.c |= 0x0000_FFFF;
        if self.c >= temp_length {
            self.c -= 0x8000;
        }
    }
}
