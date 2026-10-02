//! Shared byte-codec helpers for Analyze 7.5 header serialization.
//!
//! Centralizes the `DT_*` datatype codes and their sample types, header
//! constants, and the little-endian read/write primitives shared between
//! `reader.rs` and `writer.rs`.

use anyhow::{bail, Result};
use consus_core::{read_integer, write_integer, ByteOrder, EndianScalar};
use ritk_codecs::sample::SampleType;

/// `DT_UNSIGNED_CHAR`: `u8` samples.
pub const DT_UNSIGNED_CHAR: i16 = 2;
/// `DT_SIGNED_SHORT`: `i16` samples.
pub const DT_SIGNED_SHORT: i16 = 4;
/// `DT_SIGNED_INT`: `i32` samples.
pub const DT_SIGNED_INT: i16 = 8;
/// `DT_FLOAT`: `f32` samples.
pub const DT_FLOAT: i16 = 16;
/// `DT_DOUBLE`: `f64` samples.
pub const DT_DOUBLE: i16 = 64;

/// The five sample types Analyze 7.5 stores, with their `datatype` codes
/// (`dbh.h`). NIfTI-1 later reused these codes and added the others.
const DATATYPES: [(i16, SampleType); 5] = [
    (DT_UNSIGNED_CHAR, SampleType::U8),
    (DT_SIGNED_SHORT, SampleType::I16),
    (DT_SIGNED_INT, SampleType::I32),
    (DT_FLOAT, SampleType::F32),
    (DT_DOUBLE, SampleType::F64),
];

/// The sample type a `datatype` code names.
///
/// # Errors
///
/// Returns an error for a code outside the five Analyze sample types.
pub(crate) fn sample_type_from_code(code: i16) -> Result<SampleType> {
    match DATATYPES.iter().find(|&&(known, _)| known == code) {
        Some(&(_, sample_type)) => Ok(sample_type),
        None => bail!(
            "Unsupported Analyze datatype {code}. Supported codes: 2 (u8), 4 (i16), 8 (i32), 16 (f32), 64 (f64)."
        ),
    }
}

/// The `datatype` code that stores `sample_type`.
///
/// # Errors
///
/// Returns an error for a sample type Analyze 7.5 cannot store.
pub(crate) fn code_for(sample_type: SampleType) -> Result<i16> {
    match DATATYPES.iter().find(|&&(_, known)| known == sample_type) {
        Some(&(code, _)) => Ok(code),
        None => bail!(
            "Analyze cannot store {sample_type} samples; it stores u8, i16, i32, f32, and f64 — convert the image to one of them first"
        ),
    }
}

/// The `bitpix` field beside `sample_type`: its width in bits.
pub(crate) const fn bitpix(sample_type: SampleType) -> i16 {
    // The widest sample is 8 bytes, so the bit count is at most 64.
    (sample_type.byte_width() * 8) as i16
}

/// Size of the Analyze 7.5 header block in bytes (§3.1: `sizeof_hdr` must equal this).
pub(crate) const HDR_SIZE: usize = 348;

/// Required value of the Analyze 7.5 `extents` field for valid files.
pub(crate) const EXTENTS: i32 = 16_384;

/// Read the little-endian field `T` at byte offset `off` of a header block.
///
/// # Panics
///
/// When `buf` ends before the field; callers read from a full
/// [`HDR_SIZE`]-byte header block.
#[inline]
pub(crate) fn read_le<T: EndianScalar>(buf: &[u8], off: usize) -> T {
    read_integer(&buf[off..], ByteOrder::LittleEndian)
        .expect("invariant: the header block holds every field")
}

/// Write `val` little-endian into the field at byte offset `off` of a header
/// block.
///
/// # Panics
///
/// When `buf` ends before the field; callers write into a full
/// [`HDR_SIZE`]-byte header block.
#[inline]
pub(crate) fn write_le<T: EndianScalar>(buf: &mut [u8], off: usize, val: T) {
    write_integer(&mut buf[off..], val, ByteOrder::LittleEndian)
        .expect("invariant: the header block holds every field");
}
