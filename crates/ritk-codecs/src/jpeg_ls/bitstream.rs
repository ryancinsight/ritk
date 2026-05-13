//! Bit-level reader for JPEG-LS compressed scan data.
//!
//! # JPEG-LS Bit Stuffing (ISO 14495-1 §C.2.1)
//! In a JPEG-LS scan, an encoded `0xFF` data byte is followed by one stuffed
//! zero bit. The decoder preserves all eight `0xFF` data bits, discards that
//! single zero bit, and then consumes the remaining seven bits of the following
//! byte as entropy data. Marker prefixes (`0xFF` followed by a byte with the
//! high bit set) terminate the scan.

/// Bit-level reader for JPEG-LS compressed scan data.
///
/// Maintains a 32-bit buffer with lazy refill. Handles JPEG-LS bit stuffing.
pub(super) struct BitReader<'a> {
    data: &'a [u8],
    /// Current byte position in `data`.
    pos: usize,
    /// Bit accumulator (MSB aligned).
    buf: u32,
    /// Number of valid bits in `buf`.
    bits: u32,
}

impl<'a> BitReader<'a> {
    /// Construct and prime the buffer.
    pub(super) fn new(data: &'a [u8]) -> Self {
        let mut r = Self {
            data,
            pos: 0,
            buf: 0,
            bits: 0,
        };
        r.refill();
        r
    }

    #[inline(always)]
    fn push_bits(&mut self, value: u32, count: u32) {
        self.buf = (self.buf << count) | (value & ((1 << count) - 1));
        self.bits += count;
    }

    /// Refill `buf` with entropy bits, removing JPEG-LS stuffed zero bits.
    fn refill(&mut self) {
        while self.bits <= 16 && self.pos < self.data.len() {
            let b = self.data[self.pos];
            if b == 0xFF && self.pos + 1 < self.data.len() && (self.data[self.pos + 1] & 0x80) != 0
            {
                self.pos = self.data.len();
                break;
            }

            self.pos += 1;
            self.push_bits(b as u32, 8);

            if b == 0xFF && self.pos < self.data.len() {
                let stuffed = self.data[self.pos];
                if (stuffed & 0x80) == 0 {
                    self.pos += 1;
                    self.push_bits((stuffed & 0x7F) as u32, 7);
                }
            }
        }
    }

    /// Read `n` bits (n ≤ 24). Returns 0 on underflow (stream exhausted).
    #[inline(always)]
    pub(super) fn read_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.bits < n {
            self.refill();
        }
        if self.bits < n {
            return 0; // underflow: stream is exhausted
        }
        self.bits -= n;
        (self.buf >> self.bits) & ((1 << n) - 1)
    }

    /// Read 1 bit.
    #[inline(always)]
    pub(super) fn read_bit(&mut self) -> u32 {
        self.read_bits(1)
    }

    /// Decode a Golomb-Rice code with ISO 14495-1 LIMIT guard (§A.3).
    ///
    /// # Arguments
    /// * `k`    — Golomb-Rice order (0 ≤ k ≤ qbpp)
    /// * `limit` — Maximum total code length; value = `2*(qbpp + max(2, bpp))`
    /// * `qbpp`  — Bits required to represent RANGE; equals `bpp` for lossless
    ///
    /// # Returns
    /// The decoded non-negative MErrval.
    pub(super) fn read_golomb(&mut self, k: u32, limit: u32, qbpp: u32) -> u32 {
        // Count leading zeros (unary quotient), stopping at limit−qbpp−1
        let max_zeros = limit - qbpp - 1;
        let mut q = 0u32;
        loop {
            if self.read_bit() == 1 {
                break;
            }
            q += 1;
            if q == max_zeros {
                let _ = self.read_bit();
                break;
            }
        }
        if q < max_zeros {
            // Normal Golomb-Rice: MErrval = (q << k) | read_bits(k)
            let rem = self.read_bits(k);
            (q << k) | rem
        } else {
            // Limited-length code: encoded low bits store mapped_error − 1.
            let raw_val = self.read_bits(qbpp);
            raw_val + 1
        }
    }

    /// Return remaining byte count (approximate; does not account for buffered bits).
    #[allow(dead_code)]
    pub(super) fn remaining_bytes(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_basic_bit_sequence() {
        // 0b10110000 = bits 1,0,1,1,0,0,0,0 from MSB
        let data = [0b10110000u8, 0b11001100u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bit(), 1);
        assert_eq!(r.read_bit(), 0);
        assert_eq!(r.read_bit(), 1);
        assert_eq!(r.read_bit(), 1);
        assert_eq!(r.read_bit(), 0);
        assert_eq!(r.read_bit(), 0);
    }

    #[test]
    fn read_bits_3_from_0b101() {
        // 0b10110000: top 3 bits = 0b101 = 5
        let data = [0b10110000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(3), 5);
    }

    #[test]
    fn stuffed_zero_bit_discarded() {
        // 0xFF 0x00 0b10000000 -> 0xFF, seven data zeros, then the next byte.
        let data = [0xFF, 0x00, 0b10000000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(8), 0xFF);
        assert_eq!(r.read_bits(7), 0);
        assert_eq!(r.read_bit(), 1);
    }

    #[test]
    fn marker_terminates_scan_data() {
        let data = [0b10100000u8, 0xFF, 0xD9, 0xFF];
        let mut r = BitReader::new(&data);

        assert_eq!(r.read_bits(8), 0b10100000);
        assert_eq!(r.read_bits(8), 0);
        assert_eq!(r.remaining_bytes(), 0);
    }

    #[test]
    fn golomb_rice_k0_meval_0() {
        // MErrval=0 with k=0: Golomb code is '1' (0 zeros, then 1, then 0 bits)
        // bit sequence: 1 → 1 bit set = 0b10000000
        let data = [0b10000000u8];
        let mut r = BitReader::new(&data);
        let limit = 32u32;
        let qbpp = 8u32;
        let val = r.read_golomb(0, limit, qbpp);
        // q=0 (read '1' first bit, no leading zeros), rem=0 → MErrval=0
        assert_eq!(val, 0);
    }

    #[test]
    fn golomb_rice_k0_meval_2() {
        // MErrval=2 with k=0: q=2, code = '001' (2 zeros, then 1, then 0 bits)
        // bit sequence: 0 0 1 → top bits of 0b00100000
        let data = [0b00100000u8];
        let mut r = BitReader::new(&data);
        let limit = 32u32;
        let qbpp = 8u32;
        let val = r.read_golomb(0, limit, qbpp);
        assert_eq!(val, 2);
    }

    #[test]
    fn golomb_rice_k1_meval_5() {
        // k=1, MErrval=5: q = 5>>1 = 2, rem = 5 & 1 = 1
        // code = '001' + '1' = bits: 0,0,1,1 → 0b00110000
        let data = [0b00110000u8];
        let mut r = BitReader::new(&data);
        let limit = 32u32;
        let qbpp = 8u32;
        let val = r.read_golomb(1, limit, qbpp);
        assert_eq!(val, 5);
    }

    #[test]
    fn golomb_limited_code_decodes_mapped_error_plus_one() {
        let data = [0x00, 0x00, 0x01, 0xFE];
        let mut r = BitReader::new(&data);
        let val = r.read_golomb(2, 32, 8);

        assert_eq!(val, 255);
    }
}
