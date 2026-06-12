//! Shared byte-codec helpers for Analyze 7.5 header serialization.
//!
//! Centralizes the `DT_*` datatype constants and the little-endian read/write
//! primitives shared between `reader.rs` and `writer.rs`.

// ── Datatype constants ────────────────────────────────────────────────────────

pub const DT_UNSIGNED_CHAR: i16 = 2;
pub const DT_SIGNED_SHORT: i16 = 4;
pub const DT_SIGNED_INT: i16 = 8;
pub const DT_FLOAT: i16 = 16;
pub const DT_DOUBLE: i16 = 64;

// ── Read helpers ──────────────────────────────────────────────────────────────

/// Read a little-endian `i16` from `buf` at byte offset `off`.
#[inline]
pub(crate) fn read_i16(buf: &[u8], off: usize) -> i16 {
    i16::from_le_bytes([buf[off], buf[off + 1]])
}

/// Read a little-endian `i32` from `buf` at byte offset `off`.
#[inline]
pub(crate) fn read_i32(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Read a little-endian `f32` from `buf` at byte offset `off`.
#[inline]
pub(crate) fn read_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

// ── Write helpers ─────────────────────────────────────────────────────────────

/// Write a little-endian `i16` at byte offset `off` in `buf`.
#[inline]
pub(crate) fn write_i16(buf: &mut [u8], off: usize, val: i16) {
    buf[off..off + 2].copy_from_slice(&val.to_le_bytes());
}

/// Write a little-endian `i32` at byte offset `off` in `buf`.
#[inline]
pub(crate) fn write_i32(buf: &mut [u8], off: usize, val: i32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

/// Write a little-endian `f32` at byte offset `off` in `buf`.
#[inline]
pub(crate) fn write_f32(buf: &mut [u8], off: usize, val: f32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}
