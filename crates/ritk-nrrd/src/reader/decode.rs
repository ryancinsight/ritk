//! NRRD header parsing and byte decoding helpers.

use anyhow::{anyhow, Context, Result};
use ritk_codecs::{decode_bytes_to_f32, ByteOrder};
use ritk_spatial::Point;

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

/// Parse a 2-D `space directions` field "(a,b) (c,d)" and promote it to a 3-D
/// row-major direction matrix `[[a,b,0],[c,d,0],[0,0,1]]` — the in-plane axes
/// keep their cosines and an identity through-plane z-axis is appended (the
/// 2-D-as-z=1 convention).
pub(super) fn parse_space_directions_planar(s: &str) -> Result<[[f64; 3]; 3]> {
    let vecs = parse_vectors(s, 2)?;
    if vecs.len() != 2 {
        return Err(anyhow!(
            "2-D 'space directions' must contain 2 vectors, found {}",
            vecs.len()
        ));
    }
    Ok([
        [vecs[0][0], vecs[0][1], 0.0],
        [vecs[1][0], vecs[1][1], 0.0],
        [0.0, 0.0, 1.0],
    ])
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

/// Parse a 2-D `space origin` "(x,y)" and promote it to the 3-D point `[x,y,0]`.
pub(super) fn parse_nrrd_point_planar(s: &str) -> Result<Point<3>> {
    let vecs = parse_vectors(s, 2)?;
    if vecs.is_empty() {
        return Err(anyhow!(
            "Invalid 2-D 'space origin' format: no parenthesised vector found in '{}'",
            s
        ));
    }
    Ok(Point::new([vecs[0][0], vecs[0][1], 0.0]))
}

/// Extract all `(v0,v1,v2)` groups from `s` as `Vec<[f64;3]>`.
pub(super) fn parse_parenthesized_vectors(s: &str) -> Result<Vec<[f64; 3]>> {
    parse_vectors(s, 3).map(|vs| {
        vs.into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect::<Vec<_>>()
    })
}

/// Extract all parenthesised groups of exactly `n` comma-separated f64
/// components from `s`.  Handles spaces inside or between components; stops at
/// any unterminated group.
fn parse_vectors(s: &str, n: usize) -> Result<Vec<Vec<f64>>> {
    let mut vecs: Vec<Vec<f64>> = Vec::new();
    let mut rest = s.trim();
    while let Some(start) = rest.find('(') {
        rest = &rest[start + 1..];
        let Some(end) = rest.find(')') else { break };
        let inner = &rest[..end];
        let parts: Vec<&str> = inner.split(',').collect();
        if parts.len() != n {
            return Err(anyhow!(
                "Expected {} components in vector '({})'; got {}",
                n,
                inner,
                parts.len()
            ));
        }
        let mut v = Vec::with_capacity(n);
        for p in parts {
            v.push(
                p.trim()
                    .parse::<f64>()
                    .with_context(|| format!("Cannot parse '{}' as f64", p.trim()))?,
            );
        }
        vecs.push(v);
        rest = &rest[end + 1..];
    }
    Ok(vecs)
}

/// Decode a raw byte buffer into `Vec<f32>` according to the NRRD `type`.
///
/// Translates the NRRD type-name string to a (size, signed, is_float) triple
/// and delegates to [`ritk_codecs::decode_bytes_to_f32`].
pub(super) fn decode_element_bytes(
    bytes: &[u8],
    element_type: &str,
    count: usize,
    byte_order: ByteOrder,
) -> Result<Vec<f32>> {
    let normalised = element_type.to_lowercase();
    let (elem_size, signed, is_float) = match normalised.as_str() {
        "uchar" | "unsigned char" | "uint8" => (1_usize, false, false),
        "char" | "signed char" | "int8" => (1, true, false),
        "short" | "int16" | "signed short" | "int 16" => (2, true, false),
        "unsigned short" | "uint16" | "ushort" | "unsigned short int" => (2, false, false),
        "int" | "int32" | "signed int" | "int 32" => (4, true, false),
        "unsigned int" | "uint32" | "uint" | "unsigned int 32" => (4, false, false),
        "float" => (4, false, true),
        "double" => (8, false, true),
        other => return Err(anyhow!("Unsupported NRRD type: '{}'", other)),
    };
    decode_bytes_to_f32(
        bytes,
        elem_size,
        signed,
        is_float,
        byte_order,
        count,
        element_type,
    )
}
