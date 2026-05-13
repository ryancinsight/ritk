//! JPEG Huffman table construction and entropy-coded bit reader.
//!
//! # Specification
//! ITU-T T.81 §C.1: Canonical Huffman tables are derived from BITS[1..16]
//! (code counts per length) and HUFFVAL (symbols in canonical order).
//! Codes are assigned MSB-first: shorter codes are lexicographically smaller.
//!
//! # Bit reader
//! JPEG entropy data uses byte stuffing: after every 0xFF data byte the encoder
//! inserts a 0x00 pad byte. The decoder discards it, producing 0xFF entropy.
//! Restart markers (0xFF 0xD0–0xD7) delimit restart intervals.

use anyhow::{bail, Result};

// ─── Huffman Table ────────────────────────────────────────────────────────────

/// Canonical Huffman decode table (T.81 §C.1).
#[derive(Debug, Clone)]
pub(crate) struct HuffmanTable {
    /// Largest canonical code of length `i+1`; −1 if no codes of that length.
    maxcode: [i32; 16],
    /// Smallest canonical code of length `i+1`.
    mincode: [i32; 16],
    /// Index into `huffval` where length-`i+1` codes start.
    valptr: [usize; 16],
    /// Decoded symbols in canonical order (by ascending length, then by code).
    huffval: [u8; 256],
    pub(crate) num_symbols: usize,
}

impl HuffmanTable {
    /// Build a canonical Huffman table from `bits[0..15]` (one-indexed lengths
    /// 1–16) and `huffval` (symbols in canonical order).
    pub(crate) fn from_bits_huffval(bits: &[u8; 16], huffval: &[u8]) -> Result<Self> {
        let num_symbols = huffval.len();
        if num_symbols > 256 {
            bail!("Huffman table has too many symbols: {}", num_symbols);
        }
        let total: usize = bits.iter().map(|&b| b as usize).sum();
        if total != num_symbols {
            bail!("BITS sum {} != HUFFVAL length {}", total, num_symbols);
        }

        let mut hv = [0u8; 256];
        hv[..num_symbols].copy_from_slice(huffval);

        let mut maxcode = [-1i32; 16];
        let mut mincode = [-1i32; 16];
        let mut valptr = [0usize; 16];

        // Assign canonical codes per T.81 Figure C.1
        let mut code: i32 = 0;
        let mut idx: usize = 0;
        for len in 0..16 {
            let count = bits[len] as usize;
            if count == 0 {
                code <<= 1;
                continue;
            }
            mincode[len] = code;
            valptr[len] = idx;
            code += count as i32 - 1;
            maxcode[len] = code;
            idx += count;
            code = (code + 1) << 1;
        }

        Ok(HuffmanTable {
            maxcode,
            mincode,
            valptr,
            huffval: hv,
            num_symbols,
        })
    }

    /// Decode one symbol using the given `BitReader`.
    #[inline]
    pub(crate) fn decode(&self, reader: &mut BitReader<'_>) -> Result<u8> {
        let mut code: i32 = 0;
        for len in 0..16 {
            code = (code << 1) | (reader.read_bit()? as i32);
            if self.maxcode[len] >= 0 && code <= self.maxcode[len] {
                let idx = self.valptr[len] + (code - self.mincode[len]) as usize;
                return Ok(self.huffval[idx]);
            }
        }
        bail!("invalid Huffman code (no matching entry after 16 bits)")
    }
}

// ─── Bit Reader ───────────────────────────────────────────────────────────────

/// Bit-level reader over JPEG entropy data with byte-stuffing removal.
///
/// JPEG entropy data: 0xFF followed by 0x00 is a stuffed byte → use 0xFF.
/// 0xFF followed by 0xD0–0xD7 is a restart marker → reset DC predictors.
/// 0xFF 0xD9 is EOI → end of data.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u32,
    avail: u8,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            buf: 0,
            avail: 0,
        }
    }

    /// Read the next raw byte with byte-stuffing removal.
    /// Returns `None` when the entropy stream ends (marker boundary or EOF).
    fn next_entropy_byte(&mut self) -> Option<u8> {
        if self.pos >= self.data.len() {
            return None;
        }
        let b = self.data[self.pos];
        self.pos += 1;
        if b == 0xFF {
            if self.pos >= self.data.len() {
                return None;
            }
            let next = self.data[self.pos];
            if next == 0x00 {
                // Stuffed zero — discard it, use 0xFF as the entropy byte.
                self.pos += 1;
            } else if (0xD0..=0xD7).contains(&next) {
                // Restart marker — skip it, caller resets DC predictors.
                self.pos += 1;
                return Some(0xFF); // Treat the restart as a boundary fill byte.
            } else {
                // Non-stuffed marker (e.g. EOI) — back up.
                self.pos -= 1;
                return None;
            }
        }
        Some(b)
    }

    /// Ensure `avail >= 1` bit in `buf`.
    #[inline]
    fn fill(&mut self) -> Result<()> {
        while self.avail < 8 {
            match self.next_entropy_byte() {
                Some(byte) => {
                    self.buf = (self.buf << 8) | u32::from(byte);
                    self.avail += 8;
                }
                None => {
                    // Pad with 1-bits (T.81 §F.1.2.3) when stream ends early.
                    self.buf = (self.buf << 8) | 0xFF;
                    self.avail += 8;
                }
            }
        }
        Ok(())
    }

    /// Read one bit (0 or 1) from the MSB of the buffer.
    #[inline]
    pub(crate) fn read_bit(&mut self) -> Result<u8> {
        if self.avail == 0 {
            self.fill()?;
        }
        self.avail -= 1;
        Ok(((self.buf >> self.avail) & 1) as u8)
    }

    /// Read `n` bits (MSB first) and return them as a `u32`.
    /// `n` must be ≤ 16.
    pub(crate) fn read_bits(&mut self, n: u8) -> Result<u32> {
        debug_assert!(n <= 16);
        if n == 0 {
            return Ok(0);
        }
        while self.avail < n {
            self.fill()?;
        }
        self.avail -= n;
        Ok((self.buf >> self.avail) & ((1u32 << n) - 1))
    }
}

// ─── Receive and Extend ───────────────────────────────────────────────────────

/// JPEG "RECEIVE and EXTEND" (T.81 §F.2.2.1): read `n` bits and sign-extend.
///
/// For `n == 0` returns 0 (zero-bit difference = no change).
/// For `n > 0`: if the leading bit is 1, value is positive; if 0, negative.
pub(crate) fn receive_and_extend(reader: &mut BitReader<'_>, n: u8) -> Result<i32> {
    if n == 0 {
        return Ok(0);
    }
    let raw = reader.read_bits(n)? as i32;
    if raw < (1 << (n - 1)) {
        // Negative value: subtract bias
        Ok(raw - (1 << n) + 1)
    } else {
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn huffman_table_single_length_1_code() {
        // BITS: one code of length 1 → code 0 maps to symbol 0
        let bits = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let table = HuffmanTable::from_bits_huffval(&bits, &[0u8]).unwrap();
        assert_eq!(table.maxcode[0], 0);
        assert_eq!(table.mincode[0], 0);
        assert_eq!(table.valptr[0], 0);
        assert_eq!(table.huffval[0], 0);
    }

    #[test]
    fn huffman_decode_category_zero_from_zero_bit() {
        // One code of length 1 mapping to category 0; entropy byte 0x7F = 0111...
        let bits = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let table = HuffmanTable::from_bits_huffval(&bits, &[0u8]).unwrap();
        let data = [0x7Fu8]; // first bit is 0
        let mut reader = BitReader::new(&data);
        let sym = table.decode(&mut reader).unwrap();
        assert_eq!(sym, 0);
    }

    #[test]
    fn huffman_decode_category_15_from_zero_bit() {
        // HUFFVAL = [15]: one code of length 1 maps to category 15
        let bits = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let table = HuffmanTable::from_bits_huffval(&bits, &[15u8]).unwrap();
        let data = [0x12u8, 0x33u8]; // first bit is 0
        let mut reader = BitReader::new(&data);
        let sym = table.decode(&mut reader).unwrap();
        assert_eq!(sym, 15);
    }

    #[test]
    fn receive_and_extend_zero_bits_returns_zero() {
        let data = [0u8];
        let mut reader = BitReader::new(&data);
        assert_eq!(receive_and_extend(&mut reader, 0).unwrap(), 0);
    }

    #[test]
    fn receive_and_extend_15_bits_negative() {
        // After 1-bit Huffman code consumed from 0x12, remaining 15 bits:
        // 0x12 = 0001_0010, 0x33 = 0011_0011
        // After consuming 1 bit (the 0 for the Huffman decode), we have:
        // remaining from 0x12: bits[1..7] = 001_0010 (7 bits)
        // all of 0x33: 0011_0011 (8 bits)
        // Combined 15 bits = 001_0010_0011_0011 = 0x1233 = 4659
        // receive_and_extend(15): 4659 < 2^14=16384 → negative → 4659 - 32767 = -28108
        let data = [0x12u8, 0x33u8];
        let mut reader = BitReader::new(&data);
        // Consume 1 bit (the Huffman code bit = 0)
        let _ = reader.read_bit().unwrap();
        let diff = receive_and_extend(&mut reader, 15).unwrap();
        assert_eq!(diff, -28108);
    }

    #[test]
    fn byte_stuffing_removes_pad_zero_after_ff() {
        // 0xFF 0x00 → entropy byte 0xFF
        let data = [0xFF, 0x00, 0x80]; // 0xFF (stuffed), 0x80
        let mut reader = BitReader::new(&data);
        let b0 = reader.read_bits(8).unwrap();
        let b1 = reader.read_bits(8).unwrap();
        assert_eq!(b0, 0xFF);
        assert_eq!(b1, 0x80);
    }
}
