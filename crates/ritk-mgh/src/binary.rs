//! Big-endian primitive I/O for the MGH wire format.
//!
//! Every fixed-width MGH header field is one of `i16`, `i32`, or `f32` in
//! big-endian byte order; the field's on-disk width is a property of the
//! numeric type being transferred, not a distinct operation per field. One
//! generic entry point ([`read_be`], [`write_be`]) spans every field type
//! the format uses, monomorphized at each call site by the target type.

use anyhow::{Context, Result};
use std::io::{Read, Write};

/// A value with a fixed-width big-endian wire representation.
pub(crate) trait BigEndian: Sized {
    fn read_be_bytes<R: Read>(reader: &mut R) -> Result<Self>;
    fn write_be_bytes<W: Write>(self, writer: &mut W) -> Result<()>;
}

macro_rules! impl_big_endian {
    ($($ty:ty => $width:expr),+ $(,)?) => {
        $(
            impl BigEndian for $ty {
                fn read_be_bytes<R: Read>(reader: &mut R) -> Result<Self> {
                    let mut buf = [0u8; $width];
                    reader
                        .read_exact(&mut buf)
                        .context(concat!("Failed to read ", stringify!($ty), " BE"))?;
                    Ok(<$ty>::from_be_bytes(buf))
                }

                fn write_be_bytes<W: Write>(self, writer: &mut W) -> Result<()> {
                    writer
                        .write_all(&self.to_be_bytes())
                        .context(concat!("Failed to write ", stringify!($ty), " BE"))
                }
            }
        )+
    };
}

impl_big_endian!(i16 => 2, i32 => 4, f32 => 4);

/// Read one big-endian `T` from `reader`.
pub(crate) fn read_be<T: BigEndian, R: Read>(reader: &mut R) -> Result<T> {
    T::read_be_bytes(reader)
}

/// Write one big-endian `T` to `writer`.
pub(crate) fn write_be<T: BigEndian, W: Write>(writer: &mut W, value: T) -> Result<()> {
    value.write_be_bytes(writer)
}
