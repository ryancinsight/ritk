//! Bit-stuffed JPEG 2000 packet headers and coding-pass counts.
use anyhow::{bail, Result};

/// Read individual bits (MSB first) from a borrowed §B.10.1 bit-stuffed
/// header: after a 0xFF byte, the following byte contributes only its low
/// 7 bits (= OpenJPEG `opj_bio_bytein`).
pub struct BitReader<'a> {
    bytes: &'a [u8],
    /// Next unread byte index.
    pos: usize,
    /// 16-bit sliding window (high byte = previously consumed byte).
    buf: u32,
    /// Unread bits remaining in the low byte of `buf`.
    ct: u8,
}

impl<'a> BitReader<'a> {
    /// Create a `BitReader` over a raw packet-header byte slice.
    pub fn new(raw: &'a [u8]) -> Self {
        Self {
            bytes: raw,
            pos: 0,
            buf: 0,
            ct: 0,
        }
    }

    fn bytein(&mut self) -> Result<()> {
        self.buf = (self.buf << 8) & 0xFFFF;
        self.ct = if self.buf == 0xFF00 { 7 } else { 8 };
        if self.pos < self.bytes.len() {
            self.buf |= u32::from(self.bytes[self.pos]);
            self.pos += 1;
            Ok(())
        } else {
            bail!(
                "J2K: truncated packet header while reading byte {}",
                self.pos
            )
        }
    }

    #[inline]
    pub fn read_bit(&mut self) -> Result<u32> {
        if self.ct == 0 {
            self.bytein()?;
        }
        self.ct -= 1;
        Ok((self.buf >> self.ct) & 1)
    }

    /// Read at most 32 packet-header bits, rejecting truncated input.
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        if u32::from(n) > u32::BITS {
            bail!("J2K: packet-header field width {n} exceeds 32-bit decode limit");
        }
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read_bit()?;
        }
        Ok(value)
    }

    /// Align past the header to the packet body and return its RAW byte
    /// offset (= OpenJPEG `opj_bio_inalign`): if the last consumed header
    /// byte is 0xFF, the mandatory stuffed follow byte is skipped too.
    pub fn byte_pos(&mut self) -> Result<usize> {
        if self.buf & 0xFF == 0xFF {
            self.bytein()?;
        }
        self.ct = 0;
        Ok(self.pos)
    }
}

/// Decode the number-of-passes code from a `BitReader` (ISO 15444-1 Table B.4).
pub(crate) fn read_num_passes(br: &mut BitReader<'_>) -> Result<u32> {
    if br.read_bit()? == 0 {
        return Ok(1);
    }
    if br.read_bit()? == 0 {
        return Ok(2);
    }
    let n = br.read_bits(2)?;
    if n != 3 {
        return Ok(3 + n);
    }
    let n = br.read_bits(5)?;
    if n != 31 {
        return Ok(6 + n);
    }
    Ok(37 + br.read_bits(7)?)
}
