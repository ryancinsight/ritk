//! Byte-order field primitives for NIfTI header (de)serialization.
//!
//! Bounds-checked scalar reads in either byte order and little-endian writes
//! over byte slices, through `consus_core`'s `EndianScalar`. No NIfTI
//! semantics live here, only the byte layer the NIfTI-1/2 header parser and
//! encoder build on.

use super::convert::f64_to_f32;
use anyhow::{anyhow, Result};
use consus_core::{read_integer, write_integer, ByteOrder, EndianScalar};

pub(super) fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N]> {
    bytes
        .get(offset..offset + N)
        .ok_or_else(|| anyhow!("NIfTI header truncated at byte {offset}"))?
        .try_into()
        .map_err(|_| anyhow!("NIfTI header field width mismatch at byte {offset}"))
}

/// Read the scalar field starting at `offset` in `order`.
pub(super) fn read_field<T: EndianScalar>(
    bytes: &[u8],
    offset: usize,
    order: ByteOrder,
) -> Result<T> {
    bytes
        .get(offset..)
        .and_then(|field| read_integer(field, order))
        .ok_or_else(|| anyhow!("NIfTI header truncated at byte {offset}"))
}

/// Write `value` little-endian into the field starting at `offset`.
///
/// # Panics
///
/// When `out` ends before the field; the encoder sizes the header buffer for
/// every field it writes.
pub(super) fn write_field<T: EndianScalar>(out: &mut [u8], offset: usize, value: T) {
    write_integer(&mut out[offset..], value, ByteOrder::LittleEndian)
        .expect("invariant: the header buffer holds every field the encoder writes");
}

pub(super) fn read_f32x4_as_f64(bytes: &[u8], offset: usize, order: ByteOrder) -> Result<[f64; 4]> {
    Ok([
        f64::from(read_field::<f32>(bytes, offset, order)?),
        f64::from(read_field::<f32>(bytes, offset + 4, order)?),
        f64::from(read_field::<f32>(bytes, offset + 8, order)?),
        f64::from(read_field::<f32>(bytes, offset + 12, order)?),
    ])
}

pub(super) fn read_f64x4(bytes: &[u8], offset: usize, order: ByteOrder) -> Result<[f64; 4]> {
    Ok([
        read_field::<f64>(bytes, offset, order)?,
        read_field::<f64>(bytes, offset + 8, order)?,
        read_field::<f64>(bytes, offset + 16, order)?,
        read_field::<f64>(bytes, offset + 24, order)?,
    ])
}

pub(super) fn write_f32x4(out: &mut [u8], offset: usize, values: [f64; 4]) {
    for (index, value) in values.into_iter().enumerate() {
        write_field(out, offset + index * 4, f64_to_f32(value, "srow"));
    }
}

pub(super) fn write_f64x4(out: &mut [u8], offset: usize, values: [f64; 4]) {
    for (index, value) in values.into_iter().enumerate() {
        write_field(out, offset + index * 8, value);
    }
}
