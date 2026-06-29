//! Endian-aware byte-field primitives for NIfTI header (de)serialization.
//!
//! Pure little/big-endian scalar reads and little-endian writes over byte
//! slices, with bounds-checked reads. No NIfTI semantics live here — only the
//! byte-layer codec the NIfTI-1/2 header parser and encoder build on.

use super::convert::f64_to_f32;
use anyhow::{anyhow, Result};

/// Byte order of a parsed NIfTI header, selected from the `sizeof_hdr` field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Endian {
    Little,
    Big,
}

pub(super) fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N]> {
    bytes
        .get(offset..offset + N)
        .ok_or_else(|| anyhow!("NIfTI header truncated at byte {offset}"))?
        .try_into()
        .map_err(|_| anyhow!("NIfTI header field width mismatch at byte {offset}"))
}

pub(super) fn read_u16(bytes: &[u8], offset: usize, endian: Endian) -> Result<u16> {
    let raw = read_array::<2>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => u16::from_le_bytes(raw),
        Endian::Big => u16::from_be_bytes(raw),
    })
}

pub(super) fn read_i16(bytes: &[u8], offset: usize, endian: Endian) -> Result<i16> {
    let raw = read_array::<2>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => i16::from_le_bytes(raw),
        Endian::Big => i16::from_be_bytes(raw),
    })
}

pub(super) fn read_i32(bytes: &[u8], offset: usize, endian: Endian) -> Result<i32> {
    let raw = read_array::<4>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => i32::from_le_bytes(raw),
        Endian::Big => i32::from_be_bytes(raw),
    })
}

pub(super) fn read_i64(bytes: &[u8], offset: usize, endian: Endian) -> Result<i64> {
    let raw = read_array::<8>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => i64::from_le_bytes(raw),
        Endian::Big => i64::from_be_bytes(raw),
    })
}

pub(super) fn read_f32(bytes: &[u8], offset: usize, endian: Endian) -> Result<f32> {
    let raw = read_array::<4>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => f32::from_le_bytes(raw),
        Endian::Big => f32::from_be_bytes(raw),
    })
}

pub(super) fn read_f64(bytes: &[u8], offset: usize, endian: Endian) -> Result<f64> {
    let raw = read_array::<8>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => f64::from_le_bytes(raw),
        Endian::Big => f64::from_be_bytes(raw),
    })
}

pub(super) fn read_f32x4_as_f64(bytes: &[u8], offset: usize, endian: Endian) -> Result<[f64; 4]> {
    Ok([
        f64::from(read_f32(bytes, offset, endian)?),
        f64::from(read_f32(bytes, offset + 4, endian)?),
        f64::from(read_f32(bytes, offset + 8, endian)?),
        f64::from(read_f32(bytes, offset + 12, endian)?),
    ])
}

pub(super) fn read_f64x4(bytes: &[u8], offset: usize, endian: Endian) -> Result<[f64; 4]> {
    Ok([
        read_f64(bytes, offset, endian)?,
        read_f64(bytes, offset + 8, endian)?,
        read_f64(bytes, offset + 16, endian)?,
        read_f64(bytes, offset + 24, endian)?,
    ])
}

pub(super) fn write_i32(out: &mut [u8], offset: usize, value: i32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn write_i64(out: &mut [u8], offset: usize, value: i64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn write_u16(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn write_i16(out: &mut [u8], offset: usize, value: i16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn write_f32(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn write_f64(out: &mut [u8], offset: usize, value: f64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

pub(super) fn write_f32x4(out: &mut [u8], offset: usize, values: [f64; 4]) {
    for (index, value) in values.into_iter().enumerate() {
        write_f32(out, offset + index * 4, f64_to_f32(value, "srow"));
    }
}

pub(super) fn write_f64x4(out: &mut [u8], offset: usize, values: [f64; 4]) {
    for (index, value) in values.into_iter().enumerate() {
        write_f64(out, offset + index * 8, value);
    }
}
