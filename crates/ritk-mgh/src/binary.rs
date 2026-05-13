//! Big-endian primitive I/O helpers for the MGH wire format.

use anyhow::{Context, Result};
use std::io::{Read, Write};

pub(crate) fn read_i32_be<R: Read>(reader: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .context("Failed to read i32 BE")?;
    Ok(i32::from_be_bytes(buf))
}

pub(crate) fn read_i16_be<R: Read>(reader: &mut R) -> Result<i16> {
    let mut buf = [0u8; 2];
    reader
        .read_exact(&mut buf)
        .context("Failed to read i16 BE")?;
    Ok(i16::from_be_bytes(buf))
}

pub(crate) fn read_f32_be<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .context("Failed to read f32 BE")?;
    Ok(f32::from_be_bytes(buf))
}

pub(crate) fn write_i32_be<W: Write>(writer: &mut W, value: i32) -> Result<()> {
    writer
        .write_all(&value.to_be_bytes())
        .context("Failed to write i32 BE")
}

pub(crate) fn write_i16_be<W: Write>(writer: &mut W, value: i16) -> Result<()> {
    writer
        .write_all(&value.to_be_bytes())
        .context("Failed to write i16 BE")
}

pub(crate) fn write_f32_be<W: Write>(writer: &mut W, value: f32) -> Result<()> {
    writer
        .write_all(&value.to_be_bytes())
        .context("Failed to write f32 BE")
}
