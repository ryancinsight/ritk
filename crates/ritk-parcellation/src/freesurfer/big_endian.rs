//! Big-endian wire primitives shared by the binary FreeSurfer formats.
//!
//! Every binary FreeSurfer surface-family file stores `i32` and `f32` in
//! big-endian order, plus a three-byte big-endian magic (`fread3` in
//! FreeSurfer's MATLAB tools). `ritk-mgh` carries an equivalent trait, but it is
//! crate-private, reports through `anyhow`, and depending on it would pull an
//! image stack into this vocabulary crate for four-byte reads.

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

/// A value with a fixed-width big-endian wire representation.
pub(super) trait BigEndian: Sized {
    /// The on-disk bytes.
    type Bytes: AsRef<[u8]> + AsMut<[u8]> + Default;
    /// Decode from big-endian bytes.
    fn from_be(bytes: Self::Bytes) -> Self;
    /// Encode to big-endian bytes.
    fn to_be(self) -> Self::Bytes;
}

impl BigEndian for i32 {
    type Bytes = [u8; 4];
    fn from_be(bytes: Self::Bytes) -> Self {
        Self::from_be_bytes(bytes)
    }
    fn to_be(self) -> Self::Bytes {
        self.to_be_bytes()
    }
}

impl BigEndian for f32 {
    type Bytes = [u8; 4];
    fn from_be(bytes: Self::Bytes) -> Self {
        Self::from_be_bytes(bytes)
    }
    fn to_be(self) -> Self::Bytes {
        self.to_be_bytes()
    }
}

/// Read one big-endian `T`.
pub(super) fn read_be<T: BigEndian>(reader: &mut impl Read) -> io::Result<T> {
    let mut bytes = T::Bytes::default();
    reader.read_exact(bytes.as_mut())?;
    Ok(T::from_be(bytes))
}

/// Write one big-endian `T`.
pub(super) fn write_be<T: BigEndian>(writer: &mut impl Write, value: T) -> io::Result<()> {
    writer.write_all(value.to_be().as_ref())
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
    bounded_count(read_be::<i32>(reader)?, format, field, max)
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
    Ok(write_be(writer, value)?)
}
