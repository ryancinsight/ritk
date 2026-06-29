//! Generic VTK I/O reader helpers — ASCII, big-endian binary numeric readers,
//! and shared line/cell parsing utilities.

use anyhow::{bail, Context, Result};
use std::io::{BufRead, Read};

/// Upper bound, in bytes, on speculative buffer allocation for a length field
/// that has not yet been validated against the actual input stream.
///
/// VTK count/size fields (`POINTS n`, `CELLS n total`, …) are attacker- and
/// corruption-controlled: a header may name a multi-gigabyte `count` whose bytes
/// are not present in the file. Allocating the full claimed size up front turns
/// such a header into an out-of-memory abort. The bounded readers below grow
/// their buffers by at most this increment per *confirmed* chunk, so a lying
/// length fails fast with a truncation error after a bounded allocation while
/// legitimate large payloads still read in full.
const MAX_EAGER_BYTES: usize = 16 * 1024 * 1024; // 16 MiB

/// Capacity to reserve for a collection of `count` elements of `elem_size` bytes
/// when `count` comes from an unvalidated length field.
///
/// Caps the reservation at [`MAX_EAGER_BYTES`] so a hostile header cannot force a
/// huge speculative allocation; the collection still grows to its true length as
/// elements are pushed from validated input.
pub(crate) fn bounded_capacity(count: usize, elem_size: usize) -> usize {
    count.min(MAX_EAGER_BYTES / elem_size.max(1))
}

/// Read exactly `byte_count` bytes into a freshly allocated `Vec<u8>`, bounding
/// the speculative allocation to [`MAX_EAGER_BYTES`] per chunk.
///
/// Equivalent in result to `read_exact` over a `vec![0u8; byte_count]`, but the
/// buffer grows only as bytes are confirmed present. A `byte_count` larger than
/// the remaining input therefore yields a truncation `Err` after allocating at
/// most one extra chunk, never a huge up-front reservation. `ctx` describes the
/// payload for the error message.
pub(crate) fn read_exact_bounded<R: Read + ?Sized>(
    reader: &mut R,
    byte_count: usize,
    ctx: &str,
) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    let mut remaining = byte_count;
    while remaining > 0 {
        let chunk = remaining.min(MAX_EAGER_BYTES);
        let start = buf.len();
        buf.resize(start + chunk, 0);
        reader
            .read_exact(&mut buf[start..])
            .with_context(|| ctx.to_owned())?;
        remaining -= chunk;
    }
    Ok(buf)
}

/// Read `count` ASCII whitespace-delimited numeric values from a buffered reader.
///
/// Generic over any type parseable from a string (`FromStr`) whose parse error
/// implements `Error + Send + Sync` (required by `anyhow::Context`).
///
/// The output capacity is reserved up front but capped against
/// [`MAX_EAGER_BYTES`] so a hostile `count` cannot force a huge speculative
/// allocation; the vector grows as real values are parsed and the loop bails if
/// the stream ends before `count` values are read.
pub fn read_ascii<T>(reader: &mut dyn BufRead, count: usize, type_name: &str) -> Result<Vec<T>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let mut out = Vec::with_capacity(bounded_capacity(count, std::mem::size_of::<T>()));
    let mut buf = String::new();
    while out.len() < count {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        for tok in buf.split_whitespace() {
            if out.len() >= count {
                break;
            }
            let v: T = tok
                .parse()
                .with_context(|| format!("bad {type_name} token '{tok}'"))?;
            out.push(v);
        }
    }
    if out.len() != count {
        bail!("expected {count} {type_name} values, got {}", out.len());
    }
    Ok(out)
}

/// Trait for types that can be decoded from big-endian byte slices.
pub trait FromBeBytes: Sized {
    /// Number of bytes per element.
    const SIZE: usize;
    /// Decode one value from a big-endian byte slice of exactly `SIZE` bytes.
    fn from_be_slice(bytes: &[u8]) -> Self;
}

impl FromBeBytes for f32 {
    const SIZE: usize = 4;
    fn from_be_slice(bytes: &[u8]) -> Self {
        f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl FromBeBytes for f64 {
    const SIZE: usize = 8;
    fn from_be_slice(bytes: &[u8]) -> Self {
        f64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

impl FromBeBytes for i32 {
    const SIZE: usize = 4;
    fn from_be_slice(bytes: &[u8]) -> Self {
        i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl FromBeBytes for u8 {
    const SIZE: usize = 1;
    fn from_be_slice(bytes: &[u8]) -> Self {
        bytes[0]
    }
}

/// Read the next non-blank line from a buffered reader.
///
/// Strips leading/trailing whitespace.  Returns `Ok(None)` at EOF, `Err` on
/// I/O failure, and `Ok(Some(line))` for the first non-empty line.
pub(crate) fn read_line(reader: &mut dyn BufRead) -> Result<Option<String>> {
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = buf.trim();
        if !trimmed.is_empty() {
            return Ok(Some(trimmed.to_owned()));
        }
    }
}

/// Reconstruct a `Vec<Vec<u32>>` from a flat VTK i32 cell buffer.
///
/// Buffer layout: `[n0, i0_0, …, i0_{n0-1}, n1, i1_0, …]` where `n_k` is the
/// vertex count of cell k.  Errors on truncated data or overrun.
pub(crate) fn parse_cells_from_ints(raw: &[i32], n_cells: usize) -> Result<Vec<Vec<u32>>> {
    let mut cells = Vec::with_capacity(n_cells);
    let mut pos = 0;
    for _ in 0..n_cells {
        if pos >= raw.len() {
            bail!("truncated cell data");
        }
        let count = raw[pos] as usize;
        pos += 1;
        if pos + count > raw.len() {
            bail!("cell overruns data buffer");
        }
        let cell: Vec<u32> = raw[pos..pos + count].iter().map(|&i| i as u32).collect();
        cells.push(cell);
        pos += count;
    }
    Ok(cells)
}

/// Read `count` big-endian binary numeric values from a reader.
///
/// The intermediate byte buffer (`count * T::SIZE` bytes) is read through
/// [`read_exact_bounded`], so a corrupt or hostile `count` cannot force a huge
/// up-front allocation: the buffer grows by at most [`MAX_EAGER_BYTES`] per
/// confirmed chunk and a count exceeding the available input yields a truncation
/// error. The `count * T::SIZE` product is checked for overflow.
pub fn read_binary_be<T: FromBeBytes>(
    reader: &mut dyn Read,
    count: usize,
    type_name: &str,
) -> Result<Vec<T>> {
    let byte_count = count
        .checked_mul(T::SIZE)
        .with_context(|| format!("binary {type_name} length overflow ({count} values)"))?;
    let buf = read_exact_bounded(
        reader,
        byte_count,
        &format!("truncated binary {type_name} (need {count} values)"),
    )?;
    Ok(buf
        .chunks_exact(T::SIZE)
        .map(|c| T::from_be_slice(c))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_binary_be_roundtrips_values() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&3.5f32.to_be_bytes());
        payload.extend_from_slice(&(-1.25f32).to_be_bytes());
        let mut cur = Cursor::new(payload);
        let out = read_binary_be::<f32>(&mut cur, 2, "f32").expect("read");
        assert_eq!(out, vec![3.5f32, -1.25f32]);
    }

    #[test]
    fn read_binary_be_rejects_hostile_count_without_oom() {
        // Header claims a billion f32 values (4 GiB) but only 8 bytes follow.
        // The bounded reader must allocate at most one chunk and fail with a
        // truncation error rather than reserving 4 GiB up front.
        let mut cur = Cursor::new(vec![0u8; 8]);
        let err = read_binary_be::<f32>(&mut cur, 1_000_000_000, "f32")
            .expect_err("hostile count must error");
        assert!(
            err.to_string().contains("truncated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn read_binary_be_rejects_length_overflow() {
        // count * SIZE overflows usize: must error before any allocation.
        let mut cur = Cursor::new(Vec::<u8>::new());
        let err =
            read_binary_be::<f64>(&mut cur, usize::MAX, "f64").expect_err("overflow must error");
        assert!(
            err.to_string().contains("overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn read_ascii_roundtrips_values() {
        let mut cur = Cursor::new(b"3.5 -1.25\n".to_vec());
        let out = read_ascii::<f32>(&mut cur, 2, "f32").expect("read");
        assert_eq!(out, vec![3.5f32, -1.25f32]);
    }

    #[test]
    fn read_ascii_rejects_hostile_count_without_oom() {
        // Capacity is capped at MAX_EAGER_BYTES regardless of the billion-value
        // claim; the truncated stream then triggers a count-mismatch error.
        let mut cur = Cursor::new(b"1.0 2.0".to_vec());
        let err = read_ascii::<f32>(&mut cur, 1_000_000_000, "f32")
            .expect_err("hostile count must error");
        assert!(
            err.to_string().contains("expected"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn read_exact_bounded_reads_full_payload() {
        let mut cur = Cursor::new(vec![1u8, 2, 3, 4, 5]);
        let out = read_exact_bounded(&mut cur, 5, "ctx").expect("read");
        assert_eq!(out, vec![1u8, 2, 3, 4, 5]);
    }

    #[test]
    fn read_exact_bounded_truncation_errors() {
        let mut cur = Cursor::new(vec![1u8, 2, 3]);
        let err = read_exact_bounded(&mut cur, 5, "payload ctx").expect_err("must error");
        assert!(err.to_string().contains("payload ctx"), "unexpected: {err}");
    }
}
