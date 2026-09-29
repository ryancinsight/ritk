//! Big-endian wire helpers specific to the binary FreeSurfer formats.
//!
//! Binary FreeSurfer surface-family files store `i32` and `f32` big-endian,
//! read and written through `consus_core::{read_from, write_to}`. This module
//! keeps what is FreeSurfer's own: the three-byte magic (`fread3`/`fwrite3`)
//! and the bounded element counts.

use consus_core::{ByteOrder, read_from, write_to};
use std::io::{self, Read, Write};

use super::{FreeSurferError, FreeSurferFormat};

/// Element count up to which a reader reserves storage before data arrives.
///
/// A count field cannot be checked against the length of an arbitrary reader,
/// so trusting it for `with_capacity` lets a forged header demand gigabytes
/// before a single element is read. Beyond this many elements the vector grows
/// only as real input backs it, which keeps the allocation proportional to the
/// bytes actually supplied plus this constant.
const RESERVE_LIMIT: usize = 1 << 16;

/// Capacity to reserve for `count` elements announced by a header.
pub(super) fn reserve_for(count: usize) -> usize {
    count.min(RESERVE_LIMIT)
}

/// Read a three-byte big-endian unsigned integer (FreeSurfer `fread3`).
pub(super) fn read_u24(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 3];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]))
}

/// Write the low three bytes of `value` big-endian (FreeSurfer `fwrite3`).
pub(super) fn write_u24(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_be_bytes()[1..])
}

/// Read an `i32` count and accept it only within `0..=max`.
pub(super) fn read_count(
    reader: &mut impl Read,
    format: FreeSurferFormat,
    field: &'static str,
    max: usize,
) -> Result<usize, FreeSurferError> {
    bounded_count(
        read_from::<i32, _>(reader, ByteOrder::BigEndian)?,
        format,
        field,
        max,
    )
}

/// Accept a count read from a header only within `0..=max`.
pub(super) fn bounded_count(
    count: i32,
    format: FreeSurferFormat,
    field: &'static str,
    max: usize,
) -> Result<usize, FreeSurferError> {
    usize::try_from(count)
        .ok()
        .filter(|count| *count <= max)
        .ok_or(FreeSurferError::InvalidCount {
            format,
            field,
            count: i64::from(count),
            max: i64::try_from(max).unwrap_or(i64::MAX),
        })
}

/// Write a count that the caller has bounded to the `i32` range.
pub(super) fn write_count(
    writer: &mut impl Write,
    format: FreeSurferFormat,
    field: &'static str,
    count: usize,
) -> Result<(), FreeSurferError> {
    let value = i32::try_from(count).map_err(|_| FreeSurferError::InvalidCount {
        format,
        field,
        count: i64::try_from(count).unwrap_or(i64::MAX),
        max: i64::from(i32::MAX),
    })?;
    Ok(write_to(writer, value, ByteOrder::BigEndian)?)
}
