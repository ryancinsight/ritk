//! Shared byte-codec helpers for Analyze 7.5 header serialization.
//!
//! Centralizes the `DT_*` datatype constants, header constants, and the
//! little-endian read/write primitives shared between `reader.rs` and
//! `writer.rs`.

use consus_core::{read_integer, write_integer, ByteOrder, EndianScalar};

// ── Datatype constants ────────────────────────────────────────────────────────

pub const DT_UNSIGNED_CHAR: i16 = 2;
pub const DT_SIGNED_SHORT: i16 = 4;
pub const DT_SIGNED_INT: i16 = 8;
pub const DT_FLOAT: i16 = 16;
pub const DT_DOUBLE: i16 = 64;

// ── Header constants ──────────────────────────────────────────────────────────

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
