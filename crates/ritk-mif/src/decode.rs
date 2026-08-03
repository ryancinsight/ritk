//! Byte decoding for MRtrix `.mif` binary payloads.

use anyhow::{anyhow, Result};

/// Decode raw bytes into `f32` values.
///
/// `elem_size` is the byte size of each element (e.g. 4 for float32, 8 for
/// float64).  `is_big_endian` selects the byte-order for multi-byte types.
///
/// Supported element sizes: 1 (uint8/int8 → f32), 2 (int16/uint16 → f32),
/// 4 (float32/int32/uint32 → f32), 8 (float64 → f32).
pub(crate) fn decode_bytes(
    raw: &[u8],
    elem_size: usize,
    expected_count: usize,
    is_big_endian: bool,
) -> Result<Vec<f32>> {
    if raw.len() != expected_count * elem_size {
        return Err(anyhow!(
            ".mif payload size mismatch: expected {} elements ({} bytes), got {} bytes",
            expected_count,
            expected_count * elem_size,
            raw.len()
        ));
    }

    let mut out = Vec::with_capacity(expected_count);

    match elem_size {
        1 => {
            for &b in raw {
                out.push(b as f32);
            }
        }
        2 => {
            for chunk in raw.chunks_exact(2) {
                let arr: [u8; 2] = chunk.try_into().unwrap();
                let val = if is_big_endian {
                    i16::from_be_bytes(arr) as f32
                } else {
                    i16::from_le_bytes(arr) as f32
                };
                out.push(val);
            }
        }
        4 => {
            for chunk in raw.chunks_exact(4) {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                let val = if is_big_endian {
                    f32::from_bits(u32::from_be_bytes(arr))
                } else {
                    f32::from_bits(u32::from_le_bytes(arr))
                };
                out.push(val);
            }
        }
        8 => {
            for chunk in raw.chunks_exact(8) {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                let val = if is_big_endian {
                    f64::from_be_bytes(arr) as f32
                } else {
                    f64::from_le_bytes(arr) as f32
                };
                out.push(val);
            }
        }
        _ => {
            return Err(anyhow!(
                "Unsupported .mif element size {} (expected 1, 2, 4, or 8)",
                elem_size
            ));
        }
    }

    Ok(out)
}
