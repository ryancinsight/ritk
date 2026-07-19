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
/// row-major direction matrix `[[a,b,0],[c,d,0],[0,0,1]]` â€” the in-plane axes
/// keep their cosines and an identity through-plane z-axis is appended (the
/// 2-D-as-z=1 convention).
pub(super) fn parse_space_directions_planar(s: &str) -> Result<[[f64; 3]; 3]> {
    let vecs = parse_vectors::<2>(s)?;
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
    if vecs.len() != 1 {
        return Err(anyhow!(
            "'space origin' must contain exactly 1 vector, found {}",
            vecs.len()
        ));
    }
    Ok(Point::new([vecs[0][0], vecs[0][1], vecs[0][2]]))
}

/// Parse a 2-D `space origin` "(x,y)" and promote it to the 3-D point `[x,y,0]`.
pub(super) fn parse_nrrd_point_planar(s: &str) -> Result<Point<3>> {
    let vecs = parse_vectors::<2>(s)?;
    if vecs.len() != 1 {
        return Err(anyhow!(
            "2-D 'space origin' must contain exactly 1 vector, found {}",
            vecs.len()
        ));
    }
    Ok(Point::new([vecs[0][0], vecs[0][1], 0.0]))
}

/// Extract all `(v0,v1,v2)` groups from `s` as `Vec<[f64;3]>`.
pub(super) fn parse_parenthesized_vectors(s: &str) -> Result<Vec<[f64; 3]>> {
    parse_vectors::<3>(s)
}

/// Extract all parenthesised groups of exactly `N` comma-separated f64
/// components from `s`. Handles spaces inside or between components and
/// rejects non-whitespace text outside the vector groups.
fn parse_vectors<const N: usize>(s: &str) -> Result<Vec<[f64; N]>> {
    let mut vecs: Vec<[f64; N]> = Vec::new();
    let mut rest = s.trim();
    while !rest.is_empty() {
        let Some(after_open) = rest.strip_prefix('(') else {
            return Err(anyhow!(
                "Unexpected text outside vector group in '{}': '{}'",
                s,
                rest
            ));
        };
        rest = after_open;
        let Some(end) = rest.find(')') else {
            return Err(anyhow!("Unterminated vector group in '{}'", s));
        };
        let inner = &rest[..end];
        vecs.push(parse_vector_components::<N>(inner)?);
        rest = rest[end + 1..].trim_start();
    }
    Ok(vecs)
}

fn parse_vector_components<const N: usize>(inner: &str) -> Result<[f64; N]> {
    let mut values = [0.0_f64; N];
    let mut count = 0usize;
    for part in inner.split(',') {
        if count == N {
            return Err(anyhow!(
                "Expected {} components in vector '({})'; got more than {}",
                N,
                inner,
                N
            ));
        }
        let trimmed = part.trim();
        values[count] = trimmed
            .parse::<f64>()
            .with_context(|| format!("Cannot parse '{}' as f64", trimmed))?;
        count += 1;
    }
    if count != N {
        return Err(anyhow!(
            "Expected {} components in vector '({})'; got {}",
            N,
            inner,
            count
        ));
    }
    Ok(values)
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
    let (elem_size, signed, is_float) = element_type_spec(element_type)?;
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

pub(super) fn element_type_spec(element_type: &str) -> Result<(usize, bool, bool)> {
    let normalised = element_type.to_lowercase();
    let spec = match normalised.as_str() {
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
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::{
        parse_nrrd_point, parse_nrrd_point_planar, parse_parenthesized_vectors,
        parse_space_directions, parse_space_directions_planar,
    };

    #[test]
    fn parse_space_directions_returns_fixed_vectors() {
        let directions =
            parse_space_directions("(1, 0, 0) (0, 2, 0) (0, 0, 3)").expect("valid vectors");

        assert_eq!(
            directions,
            [[1.0_f64, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            "3-D space directions must preserve vector component order"
        );
    }

    #[test]
    fn parse_planar_directions_promotes_z_axis() {
        let directions =
            parse_space_directions_planar("(0.5, 0) (0, 0.25)").expect("valid planar vectors");

        assert_eq!(
            directions,
            [[0.5_f64, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]],
            "2-D directions must promote to identity through-plane z-axis"
        );
    }

    #[test]
    fn parse_planar_origin_promotes_zero_z() {
        let origin = parse_nrrd_point_planar("(4.5, -2.0)").expect("valid planar origin");

        assert_eq!(
            origin.to_array(),
            [4.5_f64, -2.0, 0.0],
            "2-D origin must promote to z = 0"
        );
    }

    #[test]
    fn parse_parenthesized_vectors_rejects_wrong_component_count() {
        let err = parse_parenthesized_vectors("(1, 2) (3, 4, 5)")
            .expect_err("first vector has only two components");

        assert!(
            err.to_string()
                .contains("Expected 3 components in vector '(1, 2)'; got 2"),
            "wrong component count error must name the violated vector contract, got {err}"
        );
    }

    #[test]
    fn parse_parenthesized_vectors_rejects_unterminated_group() {
        let err = parse_parenthesized_vectors("(1, 0, 0) (0, 1, 0")
            .expect_err("second vector is unterminated");

        assert!(
            err.to_string()
                .contains("Unterminated vector group in '(1, 0, 0) (0, 1, 0'"),
            "unterminated vector error must name the rejected field value, got {err}"
        );
    }

    #[test]
    fn parse_parenthesized_vectors_rejects_trailing_tokens() {
        let err = parse_parenthesized_vectors("(1, 0, 0) junk")
            .expect_err("trailing token is not part of a vector list");

        assert!(
            err.to_string()
                .contains("Unexpected text outside vector group in '(1, 0, 0) junk': 'junk'"),
            "trailing token error must name the rejected suffix, got {err}"
        );
    }

    #[test]
    fn parse_space_origin_rejects_multiple_vectors() {
        let err = parse_nrrd_point("(1, 2, 3) (4, 5, 6)")
            .expect_err("space origin must contain one vector");

        assert!(
            err.to_string()
                .contains("'space origin' must contain exactly 1 vector, found 2"),
            "space origin error must name the vector-count contract, got {err}"
        );
    }
}
