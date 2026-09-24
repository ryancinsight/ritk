//! Shared byte-codec helpers for Analyze 7.5 header serialization.
//!
//! Centralizes the `DT_*` datatype constants, header constants, and the
//! little-endian read/write primitives shared between `reader.rs` and
//! `writer.rs`.

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

// ── Sealed trait for little-endian primitive codec ─────────────────────────

mod sealed {
    /// Sealed marker: types that can be encoded/decoded as little-endian bytes.
    /// Sealed to prevent external implementations.
    pub trait LeBytes: Sized {
        fn le_decode(buf: &[u8], off: usize) -> Self;
        fn le_encode(self, buf: &mut [u8], off: usize);
    }

    macro_rules! impl_le_bytes {
        ($($ty:ty => $width:expr),+ $(,)?) => {
            $(
                impl LeBytes for $ty {
                    fn le_decode(buf: &[u8], off: usize) -> Self {
                        <$ty>::from_le_bytes(
                            buf[off..off + $width]
                                .try_into()
                                .expect("slice width matches primitive width"),
                        )
                    }
                    fn le_encode(self, buf: &mut [u8], off: usize) {
                        buf[off..off + $width].copy_from_slice(&self.to_le_bytes());
                    }
                }
            )+
        };
    }

    impl_le_bytes!(i16 => 2, i32 => 4, f32 => 4);
}

/// Read a little-endian value of type `T` from `buf` at byte offset `off`.
#[inline]
pub(crate) fn read_le<T: sealed::LeBytes>(buf: &[u8], off: usize) -> T {
    T::le_decode(buf, off)
}

/// Write a little-endian value of type `T` into `buf` at byte offset `off`.
#[inline]
pub(crate) fn write_le<T: sealed::LeBytes>(buf: &mut [u8], off: usize, val: T) {
    val.le_encode(buf, off);
}
