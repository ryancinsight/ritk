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

impl BigEndian for i16 {
    fn read_be_bytes<R: Read>(reader: &mut R) -> Result<Self> {
        let mut buf = [0u8; 2];
        reader
            .read_exact(&mut buf)
            .context("Failed to read i16 BE")?;
        Ok(i16::from_be_bytes(buf))
    }

    fn write_be_bytes<W: Write>(self, writer: &mut W) -> Result<()> {
        writer
            .write_all(&self.to_be_bytes())
            .context("Failed to write i16 BE")
    }
}

impl BigEndian for i32 {
    fn read_be_bytes<R: Read>(reader: &mut R) -> Result<Self> {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .context("Failed to read i32 BE")?;
        Ok(i32::from_be_bytes(buf))
    }

    fn write_be_bytes<W: Write>(self, writer: &mut W) -> Result<()> {
        writer
            .write_all(&self.to_be_bytes())
            .context("Failed to write i32 BE")
    }
}

impl BigEndian for f32 {
    fn read_be_bytes<R: Read>(reader: &mut R) -> Result<Self> {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .context("Failed to read f32 BE")?;
        Ok(f32::from_be_bytes(buf))
    }

    fn write_be_bytes<W: Write>(self, writer: &mut W) -> Result<()> {
        writer
            .write_all(&self.to_be_bytes())
            .context("Failed to write f32 BE")
    }
}

/// Read one big-endian `T` from `reader`.
pub(crate) fn read_be<T: BigEndian, R: Read>(reader: &mut R) -> Result<T> {
    T::read_be_bytes(reader)
}

/// Write one big-endian `T` to `writer`.
pub(crate) fn write_be<T: BigEndian, W: Write>(writer: &mut W, value: T) -> Result<()> {
    value.write_be_bytes(writer)
}
