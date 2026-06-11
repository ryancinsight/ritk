//! NRRD header parsing and byte decoding helpers.

use anyhow::{anyhow, Context, Result};
use ritk_spatial::Point;

/// Byte order for multi-byte pixel data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ByteOrder {
    MostSignificantByteFirst,
    LeastSignificantByteFirst,
}

/// Parse a `space directions` field into three NRRD file-axis vectors.
///
/// Each direction vector `v_i` encodes the physical displacement per voxel
/// step along image axis `i`:
/// ```text
/// v_i = Direction[:, i] * spacing[i]
/// spacing[i] = |v_i|
/// Direction[:, i] = v_i / |v_i|
/// ```
pub(super) fn parse_space_directions(s: &str) -> Result<[[f64; 3]; 3]> {
    let vecs = parse_parenthesized_vectors(s)?;
    if vecs.len() != 3 {
        return Err(anyhow!(
            "'space directions' must contain 3 vectors, found {}",
            vecs.len()
        ));
    }
    Ok([vecs[0], vecs[1], vecs[2]])
}

/// Parse a `space origin` field into a `Point<3>`.
///
/// The field value must contain exactly one `(v0,v1,v2)` group.
pub(super) fn parse_nrrd_point(s: &str) -> Result<Point<3>> {
    let vecs = parse_parenthesized_vectors(s)?;
    if vecs.is_empty() {
        return Err(anyhow!(
            "Invalid 'space origin' format: no parenthesised vector found in '{}'",
            s
        ));
    }
    Ok(Point::new([vecs[0][0], vecs[0][1], vecs[0][2]]))
}

/// Extract all `(v0,v1,v2)` groups from `s` and return them as `Vec<[f64;3]>`.
///
/// Handles spaces inside or between components; stops at any malformed group.
pub(super) fn parse_parenthesized_vectors(s: &str) -> Result<Vec<[f64; 3]>> {
    let mut vecs: Vec<[f64; 3]> = Vec::new();
    let mut rest = s.trim();
    while let Some(start) = rest.find('(') {
        rest = &rest[start + 1..];
        if let Some(end) = rest.find(')') {
            let inner = &rest[..end];
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() != 3 {
                return Err(anyhow!(
                    "Expected 3 components in vector '({})'; got {}",
                    inner,
                    parts.len()
                ));
            }
            let v = [
                parts[0]
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("Cannot parse '{}' as f64", parts[0].trim()))?,
                parts[1]
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("Cannot parse '{}' as f64", parts[1].trim()))?,
                parts[2]
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("Cannot parse '{}' as f64", parts[2].trim()))?,
            ];
            vecs.push(v);
            rest = &rest[end + 1..];
        } else {
            break;
        }
    }
    Ok(vecs)
}

/// Decode a raw byte buffer into `Vec<f32>` according to the NRRD `type`.
///
/// Precisely `count` elements are decoded; surplus trailing bytes are ignored.
pub(super) fn decode_bytes_to_f32(
    bytes: &[u8],
    element_type: &str,
    count: usize,
    byte_order: ByteOrder,
) -> Result<Vec<f32>> {
    match element_type.to_lowercase().as_str() {
        "uchar" | "unsigned char" | "uint8" => {
            require_bytes(bytes.len(), count, 1, element_type)?;
            Ok(bytes[..count].iter().map(|&b| b as f32).collect())
        }
        "char" | "signed char" | "int8" => {
            require_bytes(bytes.len(), count, 1, element_type)?;
            Ok(bytes[..count].iter().map(|&b| (b as i8) as f32).collect())
        }
        "short" | "int16" | "signed short" | "int 16" => {
            require_bytes(bytes.len(), count, 2, element_type)?;
            Ok(bytes
                .chunks_exact(2)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1]];
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
                        i16::from_be_bytes(b)
                    } else {
                        i16::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        "unsigned short" | "uint16" | "ushort" | "unsigned short int" => {
            require_bytes(bytes.len(), count, 2, element_type)?;
            Ok(bytes
                .chunks_exact(2)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1]];
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
                        u16::from_be_bytes(b)
                    } else {
                        u16::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        "int" | "int32" | "signed int" | "int 32" => {
            require_bytes(bytes.len(), count, 4, element_type)?;
            Ok(bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3]];
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
                        i32::from_be_bytes(b)
                    } else {
                        i32::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        "unsigned int" | "uint32" | "uint" | "unsigned int 32" => {
            require_bytes(bytes.len(), count, 4, element_type)?;
            Ok(bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3]];
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
                        u32::from_be_bytes(b)
                    } else {
                        u32::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        "float" => {
            require_bytes(bytes.len(), count, 4, element_type)?;
            Ok(bytes
                .chunks_exact(4)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3]];
                    if byte_order == ByteOrder::MostSignificantByteFirst {
                        f32::from_be_bytes(b)
                    } else {
                        f32::from_le_bytes(b)
                    }
                })
                .collect())
        }
        "double" => {
            require_bytes(bytes.len(), count, 8, element_type)?;
            Ok(bytes
                .chunks_exact(8)
                .take(count)
                .map(|c| {
                    let b = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                    (if byte_order == ByteOrder::MostSignificantByteFirst {
                        f64::from_be_bytes(b)
                    } else {
                        f64::from_le_bytes(b)
                    }) as f32
                })
                .collect())
        }
        other => Err(anyhow!("Unsupported NRRD type: '{}'", other)),
    }
}

/// Return an error when `have` bytes are fewer than `count * elem_size`.
pub(super) fn require_bytes(
    have: usize,
    count: usize,
    elem_size: usize,
    type_name: &str,
) -> Result<()> {
    let need = count * elem_size;
    if have < need {
        Err(anyhow!(
            "Insufficient data for type '{}': need {} bytes, got {}",
            type_name,
            need,
            have
        ))
    } else {
        Ok(())
    }
}

/// Parse a whitespace-separated list of exactly `expected` `f64` values.
pub(super) fn parse_f64_vec(s: &str, field: &str, expected: usize) -> Result<Vec<f64>> {
    let vals: Vec<f64> = s
        .split_whitespace()
        .map(|t| {
            t.parse::<f64>()
                .with_context(|| format!("Invalid float in '{}': '{}'", field, t))
        })
        .collect::<Result<Vec<_>>>()?;
    if vals.len() != expected {
        return Err(anyhow!(
            "'{}' must have {} components, got {}",
            field,
            expected,
            vals.len()
        ));
    }
    Ok(vals)
}

/// Parse a whitespace-separated list of exactly `expected` `usize` values.
pub(super) fn parse_usize_vec(s: &str, field: &str, expected: usize) -> Result<Vec<usize>> {
    let vals: Vec<usize> = s
        .split_whitespace()
        .map(|t| {
            t.parse::<usize>()
                .with_context(|| format!("Invalid integer in '{}': '{}'", field, t))
        })
        .collect::<Result<Vec<_>>>()?;
    if vals.len() != expected {
        return Err(anyhow!(
            "'{}' must have {} components, got {}",
            field,
            expected,
            vals.len()
        ));
    }
    Ok(vals)
}
