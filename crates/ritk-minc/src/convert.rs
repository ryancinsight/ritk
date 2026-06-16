//! Raw byte to `f32` conversion for MINC2 voxel data.
//!
//! Converts a byte slice from HDF5 contiguous storage into a `Vec<f32>`
//! based on the HDF5 `Datatype` metadata. All integer types are cast
//! to `f32`; floating-point 64-bit values are narrowed from `f64` to
//! `f32` (lossy, but within RITK tensor precision contract).

use anyhow::{bail, Result};
use consus_core::Datatype;

/// Decode raw bytes to `Vec<f32>` based on the HDF5 datatype.
///
/// # Supported Types
///
/// | HDF5 Datatype                | Conversion                    |
/// |------------------------------|-------------------------------|
/// | `Integer { 8, unsigned }`    | `u8 as f32`                   |
/// | `Integer { 8, signed }`      | `i8 as f32`                   |
/// | `Integer { 16, LE, signed }` | `i16::from_le_bytes as f32`   |
/// | `Integer { 16, LE, unsigned}`| `u16::from_le_bytes as f32`   |
/// | `Integer { 32, LE, signed }` | `i32::from_le_bytes as f32`   |
/// | `Integer { 32, LE, unsigned}`| `u32::from_le_bytes as f32`   |
/// | `Float { 32, LE }`           | `f32::from_le_bytes`          |
/// | `Float { 64, LE }`           | `f64::from_le_bytes as f32`   |
/// | Big-endian variants          | analogous with `from_be_bytes`|
///
/// # Errors
///
/// Returns `Err` for unsupported or variable-length data types.
pub fn decode_raw_bytes(raw: &[u8], dtype: &Datatype) -> Result<Vec<f32>> {
    use consus_core::ByteOrder;

    match dtype {
        Datatype::Float { bits, byte_order } => {
            let bw = bits.get();
            match (bw, byte_order) {
                (32, ByteOrder::LittleEndian) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()),
                (32, ByteOrder::BigEndian) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()),
                (64, ByteOrder::LittleEndian) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                (64, ByteOrder::BigEndian) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                _ => bail!("Unsupported float bit width: {}", bw),
            }
        }

        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => {
            let bw = bits.get();
            match (bw, byte_order, signed) {
                // 8-bit
                (8, _, false) => Ok(raw.iter().map(|&b| b as f32).collect()),
                (8, _, true) => Ok(raw.iter().map(|&b| (b as i8) as f32).collect()),

                // 16-bit little-endian
                (16, ByteOrder::LittleEndian, true) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
                    .collect()),
                (16, ByteOrder::LittleEndian, false) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
                    .collect()),

                // 16-bit big-endian
                (16, ByteOrder::BigEndian, true) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| i16::from_be_bytes([c[0], c[1]]) as f32)
                    .collect()),
                (16, ByteOrder::BigEndian, false) => Ok(raw
                    .chunks_exact(2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]]) as f32)
                    .collect()),

                // 32-bit little-endian
                (32, ByteOrder::LittleEndian, true) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()),
                (32, ByteOrder::LittleEndian, false) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()),

                // 32-bit big-endian
                (32, ByteOrder::BigEndian, true) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()),
                (32, ByteOrder::BigEndian, false) => Ok(raw
                    .chunks_exact(4)
                    .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()),

                // 64-bit (lossy cast to f32)
                (64, ByteOrder::LittleEndian, true) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                (64, ByteOrder::LittleEndian, false) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                (64, ByteOrder::BigEndian, true) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),
                (64, ByteOrder::BigEndian, false) => Ok(raw
                    .chunks_exact(8)
                    .map(|c| {
                        u64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()),

                _ => bail!("Unsupported integer type: {} bits, signed={}", bw, signed),
            }
        }

        Datatype::Boolean => Ok(raw
            .iter()
            .map(|&b| if b != 0 { 1.0f32 } else { 0.0f32 })
            .collect()),

        other => bail!("Unsupported MINC2 voxel datatype: {:?}", other),
    }
}

#[cfg(test)]
#[path = "tests_convert.rs"]
mod tests;
