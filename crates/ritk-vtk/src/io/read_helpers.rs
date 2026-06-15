//! Generic VTK I/O reader helpers — ASCII, big-endian binary numeric readers,
//! and shared line/cell parsing utilities.

use anyhow::{bail, Context, Result};
use std::io::{BufRead, Read};

/// Read `count` ASCII whitespace-delimited numeric values from a buffered reader.
///
/// Generic over any type parseable from a string (`FromStr`) whose parse error
/// implements `Error + Send + Sync` (required by `anyhow::Context`).
/// Pre-allocates the output vector to `count`.
pub fn read_ascii<T>(reader: &mut dyn BufRead, count: usize, type_name: &str) -> Result<Vec<T>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let mut out = Vec::with_capacity(count);
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
/// Pre-allocates an intermediate byte buffer of `count * T::SIZE` bytes.
pub fn read_binary_be<T: FromBeBytes>(
    reader: &mut dyn Read,
    count: usize,
    type_name: &str,
) -> Result<Vec<T>> {
    let byte_count = count * T::SIZE;
    let mut buf = vec![0u8; byte_count];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("truncated binary {type_name} (need {count} values)"))?;
    Ok(buf
        .chunks_exact(T::SIZE)
        .map(|c| T::from_be_slice(c))
        .collect())
}
