//! Generic VTK I/O reader helpers — ASCII and big-endian binary numeric readers.

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
