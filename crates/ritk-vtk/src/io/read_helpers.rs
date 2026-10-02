//! Generic VTK I/O reader helpers — ASCII, big-endian binary numeric readers,
//! and shared line/cell parsing utilities.

use anyhow::{bail, Context, Result};
use consus_core::{ByteOrder, EndianScalar};
use consus_io::{bounded_capacity, read_exact_bounded};
use ritk_codecs::sample::{SampleBuffer, SampleType};
use std::io::{BufRead, Read};

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

/// Read `count` ASCII values of `sample_type` into a [`SampleBuffer`].
///
/// Each token parses in the stored type itself, so a 64-bit integer or an `f64`
/// is never rounded through a narrower type, and a token that is not a value of
/// that type (`300` as `u8`, `1.5` as `i32`) is an error naming it.
pub(crate) fn read_ascii_samples(
    reader: &mut dyn BufRead,
    sample_type: SampleType,
    count: usize,
) -> Result<SampleBuffer> {
    let name = sample_type.name();
    Ok(match sample_type {
        SampleType::U8 => SampleBuffer::U8(read_ascii(reader, count, name)?),
        SampleType::I8 => SampleBuffer::I8(read_ascii(reader, count, name)?),
        SampleType::U16 => SampleBuffer::U16(read_ascii(reader, count, name)?),
        SampleType::I16 => SampleBuffer::I16(read_ascii(reader, count, name)?),
        SampleType::U32 => SampleBuffer::U32(read_ascii(reader, count, name)?),
        SampleType::I32 => SampleBuffer::I32(read_ascii(reader, count, name)?),
        SampleType::U64 => SampleBuffer::U64(read_ascii(reader, count, name)?),
        SampleType::I64 => SampleBuffer::I64(read_ascii(reader, count, name)?),
        SampleType::F32 => SampleBuffer::F32(read_ascii(reader, count, name)?),
        SampleType::F64 => SampleBuffer::F64(read_ascii(reader, count, name)?),
    })
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
/// The intermediate byte buffer (`count * T::BYTE_WIDTH` bytes) is read through
/// [`read_exact_bounded`], so a corrupt or hostile `count` cannot force a huge
/// up-front allocation: the buffer grows by at most [`MAX_EAGER_BYTES`] per
/// confirmed chunk and a count exceeding the available input yields a truncation
/// error. The `count * T::BYTE_WIDTH` product is checked for overflow.
pub fn read_binary_be<T: EndianScalar>(
    reader: &mut dyn Read,
    count: usize,
    type_name: &str,
) -> Result<Vec<T>> {
    let byte_count = count
        .checked_mul(T::BYTE_WIDTH)
        .with_context(|| format!("binary {type_name} length overflow ({count} values)"))?;
    let buf = read_exact_bounded(reader, byte_count)
        .with_context(|| format!("truncated binary {type_name} (need {count} values)"))?;
    Ok(buf
        .chunks_exact(T::BYTE_WIDTH)
        .map(|c| {
            T::from_bytes(c, ByteOrder::BigEndian)
                .expect("invariant: chunks_exact yields BYTE_WIDTH bytes")
        })
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
}
