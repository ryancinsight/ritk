//! Raw byte to `f32` conversion for MINC2 voxel data.
//!
//! Converts a byte slice from HDF5 contiguous storage into a `Vec<f32>`
//! based on the HDF5 `Datatype` metadata. All integer types are cast
//! to `f32`; floating-point 64-bit values are narrowed from `f64` to
//! `f32` (lossy, but within RITK tensor precision contract).

use anyhow::{bail, Result};
use consus_core::Datatype;

/// Convert raw bytes to `Vec<f32>` based on the HDF5 datatype.
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
pub fn convert_to_f32(raw: &[u8], dtype: &Datatype) -> Result<Vec<f32>> {
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
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn convert_f32_le_round_trips() {
        let val: f32 = 3.14;
        let raw = val.to_le_bytes().to_vec();
        let dtype = Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.14).abs() < 1e-5);
    }

    #[test]
    fn convert_f32_be_round_trips() {
        let val: f32 = 2.71;
        let raw = val.to_be_bytes().to_vec();
        let dtype = Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::BigEndian,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.71).abs() < 1e-5);
    }

    #[test]
    fn convert_f64_le_narrows_to_f32() {
        let val: f64 = 1.23456789;
        let raw = val.to_le_bytes().to_vec();
        let dtype = Datatype::Float {
            bits: NonZeroUsize::new(64).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.23456789f32).abs() < 1e-5);
    }

    #[test]
    fn convert_u8_identity() {
        let raw = vec![0u8, 128, 255];
        let dtype = Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: false,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result, vec![0.0, 128.0, 255.0]);
    }

    #[test]
    fn convert_i8_signed_range() {
        let raw = vec![0u8, 127, 0x80]; // 0, 127, -128
        let dtype = Datatype::Integer {
            bits: NonZeroUsize::new(8).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: true,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result, vec![0.0, 127.0, -128.0]);
    }

    #[test]
    fn convert_i16_le_boundary_values() {
        let raw: Vec<u8> = vec![
            0x00, 0x00, // 0
            0xFF, 0x7F, // 32767
            0x00, 0x80, // -32768
        ];
        let dtype = Datatype::Integer {
            bits: NonZeroUsize::new(16).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
            signed: true,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result, vec![0.0, 32767.0, -32768.0]);
    }

    #[test]
    fn convert_boolean_maps_nonzero_to_one() {
        let raw = vec![0u8, 1, 0, 255];
        let result = convert_to_f32(&raw, &Datatype::Boolean).unwrap();
        assert_eq!(result, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn convert_unsupported_type_errors() {
        let dtype = Datatype::VariableString {
            encoding: consus_core::StringEncoding::Utf8,
        };
        assert!(convert_to_f32(&[], &dtype).is_err());
    }

    #[test]
    fn convert_multiple_f32_le_values() {
        let values = [0.0f32, -1.0, 1.0, f32::MAX, f32::MIN_POSITIVE];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dtype = Datatype::Float {
            bits: NonZeroUsize::new(32).unwrap(),
            byte_order: consus_core::ByteOrder::LittleEndian,
        };
        let result = convert_to_f32(&raw, &dtype).unwrap();
        assert_eq!(result.len(), 5);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(
                result[i].to_bits(),
                expected.to_bits(),
                "mismatch at index {}",
                i
            );
        }
    }
}
