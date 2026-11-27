//! ISO/IEC 15444-1 A.7.1 tile-part length markers, in Ztlm order.
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct TileLengths<'a>(BTreeMap<u8, &'a [u8]>);
impl<'a> TileLengths<'a> {
    pub(super) fn add(&mut self, body: &'a [u8]) -> Result<()> {
        let (&index, entries) = body.split_first().context("missing Ztlm")?;
        ensure!(
            self.0.insert(index, entries).is_none(),
            "duplicate TLM segment"
        );
        Ok(())
    }
    pub(super) fn verify(&self, actual: &[(u16, usize)], required: bool) -> Result<()> {
        ensure!(
            !required || !self.0.is_empty(),
            "HTJ2K .202 requires tile length markers"
        );
        if self.0.is_empty() {
            return Ok(());
        }
        let mut parts = actual.iter();
        for (number, (&index, segment)) in self.0.iter().enumerate() {
            ensure!(usize::from(index) == number, "missing TLM segment");
            let (&style, entries) = segment.split_first().context("missing Stlm")?;
            let index_size = usize::from((style >> 4) & 3);
            ensure!(style & 0x8f == 0 && index_size <= 2, "invalid Stlm");
            let length_size = if style & 64 == 0 { 2 } else { 4 };
            let stride = index_size + length_size;
            ensure!(
                !entries.is_empty() && entries.len().is_multiple_of(stride),
                "incomplete TLM entries"
            );
            for field in entries.chunks_exact(stride) {
                let &(tile, length) = parts.next().context("extra TLM entry")?;
                let declared_tile = match index_size {
                    1 => u16::from(field[0]),
                    2 => u16::from_be_bytes([field[0], field[1]]),
                    _ => tile,
                };
                let len = field[index_size..]
                    .iter()
                    .fold(0u32, |v, b| (v << 8) | u32::from(*b));
                ensure!(
                    tile == declared_tile && length == usize::try_from(len)?,
                    "TLM entry disagrees with tile-part length or index"
                );
            }
        }
        ensure!(parts.next().is_none(), "incomplete TLM index");
        Ok(())
    }
}
