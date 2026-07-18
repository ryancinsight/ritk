use super::{CtxState, QE_TABLE};

/// MQ arithmetic decoder (ISO 15444-1 Â§C.3).
///
/// The `c` register stores the residual code fraction in bits 16â€“31 (Chigh),
/// matching the representation used by OpenJPEG.
pub struct MqDecoder<'a> {
    src: &'a [u8],
    /// Next byte position in `src`.
    pos: usize,
    /// Last byte consumed (used by the byte-stuffing logic in `bytein`).
    prev: u8,
    /// Interval register (lower 16 bits significant).
    a: u32,
    /// Code register (bits 16â€“31 = Chigh).
    c: u32,
    /// Remaining bit-shifts available before the next `bytein`.
    ct: u32 }

impl<'a> MqDecoder<'a> {
    /// Initialise the MQ decoder over `src` (ISO 15444-1 Â§C.3.2 INITDEC).
    ///
    /// `C = B0 << 16; BYTEIN; C <<= 7; CT -= 7; A = 0x8000` â€” matching
    /// OpenJPEG `opj_mqc_init_dec`.
    pub fn new(src: &'a [u8]) -> Self {
        let b0 = src.first().copied().unwrap_or(0xFF);
        let mut d = Self {
            src,
            pos: usize::from(!src.is_empty()),
            prev: b0,
            a: 0,
            c: u32::from(b0) << 16,
            ct: 0 };
        d.bytein();
        d.c <<= 7;
        d.ct -= 7;
        d.a = 0x8000;
        d
    }

    /// Decode one binary symbol using the given context (ISO 15444-1 Â§C.3.3 DECODE).
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
                // Already normalised; MPS decoded without renorm. No state
                // transition â€” ISO 15444-1 Figure C.15 advances I(CX) only on
                // the renormalisation path (mirrors the encoder's CODEMPS).
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

    // â”€â”€ private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[inline]
    fn lps_exchange(&mut self, ctx: &mut CtxState, qe: u32) -> u32 {
        let entry = QE_TABLE[ctx.state as usize];
        let d;
        if self.a < qe {
            // Exchange: MPS interval < Qe â†’ MPS takes the smaller interval.
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
        // ISO 15444-1 Â§C.3.3: A = Qe[I] where I is the ORIGINAL context index
        // (before state transition) â€” matching OpenJPEG's `mqc->a = st->qeval`.
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

    /// Read the next byte from `src` into the code register (ISO 15444-1 Â§C.3.4 BYTEIN).
    ///
    /// Byte stuffing: 0xFF followed by a byte T â‰¤ 0x8F â†’ T is stuffed (7 bits effective).
    /// Marker escape: 0xFF followed by T > 0x8F â†’ treat the byte as a fill value of 0xFF.
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
