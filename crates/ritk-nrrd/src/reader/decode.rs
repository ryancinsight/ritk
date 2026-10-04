//! NRRD header parsing and byte decoding helpers.

use anyhow::{anyhow, Context, Result};
use ritk_codecs::{decode_bytes_to_f32, ByteOrder, SampleType};
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
///
/// Non-spatial `none` slots are skipped, so a 4-D acquisition file and a 3-D
/// volume both yield exactly three vectors.
pub(super) fn parse_space_directions(s: &str) -> Result<[[f64; 3]; 3]> {
    let vecs = spatial_space_directions(s)?;
    if vecs.len() != 3 {
        return Err(anyhow!(
            "'space directions' must contain 3 spatial vectors, found {}",
            vecs.len()
        ));
    }
    Ok([vecs[0], vecs[1], vecs[2]])
}

/// Parse `space directions` into one slot per axis, `None` for a non-spatial
/// axis.
///
/// NRRD marks an axis that spans no physical direction with the bare token
/// `none` — the diffusion gradient axis of a DWI file, for example. Every other
/// slot is a parenthesised direction vector.
pub(super) fn parse_space_direction_slots(s: &str) -> Result<Vec<Option<[f64; 3]>>> {
    let mut slots = Vec::new();
    let mut rest = s.trim();

    while !rest.is_empty() {
        if let Some(after_none) = strip_none_token(rest) {
            slots.push(None);
            rest = after_none.trim_start();
            continue;
        }

        let Some(after_open) = rest.strip_prefix('(') else {
            return Err(anyhow!(
                "Unexpected text outside vector group in '{}': '{}'",
                s,
                rest
            ));
        };
        let Some(end) = after_open.find(')') else {
            return Err(anyhow!("Unterminated vector group in '{}'", s));
        };
        slots.push(Some(parse_vector_components::<3>(&after_open[..end])?));
        rest = after_open[end + 1..].trim_start();
    }

    Ok(slots)
}

/// The direction vectors of the spatial axes, dropping any `none` slots.
pub(super) fn spatial_space_directions(s: &str) -> Result<Vec<[f64; 3]>> {
    Ok(parse_space_direction_slots(s)?
        .into_iter()
        .flatten()
        .collect())
}

/// Strip a leading case-insensitive `none` token, if the text starts with one.
///
/// The token must be followed by whitespace or end of input so a hypothetical
/// future field value beginning with those letters is not silently consumed.
fn strip_none_token(rest: &str) -> Option<&str> {
    let candidate = rest.get(..4)?;
    if !candidate.eq_ignore_ascii_case("none") {
        return None;
    }
    let after = &rest[4..];
    (after.is_empty() || after.starts_with(char::is_whitespace)).then_some(after)
}

/// Parse a 2-D `space directions` field "(a,b) (c,d)" and promote it to a 3-D
/// row-major direction matrix `[[a,b,0],[c,d,0],[0,0,1]]` — the in-plane axes
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

/// Parse two rank-2 image axes expressed in a three-dimensional world space.
pub(super) fn parse_space_directions_planar_world(s: &str) -> Result<[[f64; 3]; 2]> {
    let vectors = parse_vectors::<3>(s)?;
    if vectors.len() != 2 {
        return Err(anyhow!(
            "rank-2 'space directions' in a 3-D world must contain 2 vectors, found {}",
            vectors.len()
        ));
    }
    Ok([vectors[0], vectors[1]])
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
    let point = [vecs[0][0], vecs[0][1], vecs[0][2]];
    if point.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!("'space origin' contains a non-finite coordinate"));
    }
    Ok(Point::new(point))
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
    let point = [vecs[0][0], vecs[0][1], 0.0];
    if point.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!(
            "2-D 'space origin' contains a non-finite coordinate"
        ));
    }
    Ok(Point::new(point))
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
    let stored_type = sample_type(element_type)?;
    let signed = matches!(
        stored_type,
        SampleType::I8 | SampleType::I16 | SampleType::I32 | SampleType::I64
    );
    let is_float = matches!(stored_type, SampleType::F32 | SampleType::F64);
    Ok((stored_type.byte_width(), signed, is_float))
}

pub(super) fn sample_type(element_type: &str) -> Result<SampleType> {
    let normalised = element_type.trim().to_ascii_lowercase();
    let stored_type = match normalised.as_str() {
        "uchar" | "unsigned char" | "uint8" | "uint8_t" => SampleType::U8,
        "char" | "signed char" | "int8" | "int8_t" => SampleType::I8,
        "short" | "short int" | "signed short" | "signed short int" | "int16" | "int16_t"
        | "int 16" => SampleType::I16,
        "unsigned short" | "unsigned short int" | "uint16" | "uint16_t" | "ushort" => {
            SampleType::U16
        }
        "int" | "signed int" | "int32" | "int32_t" | "int 32" => SampleType::I32,
        "unsigned int" | "uint32" | "uint32_t" | "uint" | "unsigned int 32" => SampleType::U32,
        "long long"
        | "long long int"
        | "signed long long"
        | "signed long long int"
        | "longlong"
        | "int64"
        | "int64_t" => SampleType::I64,
        "unsigned long long" | "unsigned long long int" | "ulonglong" | "uint64" | "uint64_t" => {
            SampleType::U64
        }
        "float" => SampleType::F32,
        "double" => SampleType::F64,
        other => return Err(anyhow!("Unsupported NRRD type: '{other}'")),
    };
    Ok(stored_type)
}

#[cfg(test)]
mod tests {
    use super::{
        parse_nrrd_point, parse_nrrd_point_planar, parse_parenthesized_vectors,
        parse_space_direction_slots, parse_space_directions, parse_space_directions_planar,
        sample_type,
    };
    use ritk_codecs::SampleType;

    #[test]
    fn nrrd_sample_names_cover_every_fixed_width_codec_type() {
        let names = [
            ("unsigned char", SampleType::U8),
            ("uint8", SampleType::U8),
            ("uint8_t", SampleType::U8),
            ("signed char", SampleType::I8),
            ("int8", SampleType::I8),
            ("int8_t", SampleType::I8),
            ("unsigned short", SampleType::U16),
            ("unsigned short int", SampleType::U16),
            ("uint16", SampleType::U16),
            ("uint16_t", SampleType::U16),
            ("short", SampleType::I16),
            ("short int", SampleType::I16),
            ("signed short", SampleType::I16),
            ("signed short int", SampleType::I16),
            ("int16", SampleType::I16),
            ("int16_t", SampleType::I16),
            ("unsigned int", SampleType::U32),
            ("uint", SampleType::U32),
            ("uint32", SampleType::U32),
            ("uint32_t", SampleType::U32),
            ("int", SampleType::I32),
            ("signed int", SampleType::I32),
            ("int32", SampleType::I32),
            ("int32_t", SampleType::I32),
            ("unsigned long long", SampleType::U64),
            ("unsigned long long int", SampleType::U64),
            ("ulonglong", SampleType::U64),
            ("uint64", SampleType::U64),
            ("uint64_t", SampleType::U64),
            ("long long", SampleType::I64),
            ("long long int", SampleType::I64),
            ("signed long long", SampleType::I64),
            ("signed long long int", SampleType::I64),
            ("longlong", SampleType::I64),
            ("int64", SampleType::I64),
            ("int64_t", SampleType::I64),
            ("float", SampleType::F32),
            ("double", SampleType::F64),
            ("char", SampleType::I8),
            ("uchar", SampleType::U8),
            ("ushort", SampleType::U16),
        ];
        for (name, expected) in names {
            assert_eq!(sample_type(name).expect("supported NRRD type"), expected);
        }
        assert_eq!(
            sample_type(" uint64_t ").expect("trimmed alias"),
            SampleType::U64
        );
        let error = sample_type("long double").expect_err("unsupported type is rejected");
        assert!(error.to_string().contains("long double"));
    }

    #[test]
    fn parse_space_directions_skips_none_axes() {
        // DWI-style 4-axis header: the gradient axis spans no physical
        // direction and is marked `none`; the 3 spatial vectors remain.
        let directions = parse_space_directions("none (1, 0, 0) (0, 2, 0) (0, 0, 3)")
            .expect("none slot plus 3 spatial vectors");
        assert_eq!(
            directions,
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        );
    }

    #[test]
    fn parse_space_direction_slots_preserves_axis_positions() {
        let slots = parse_space_direction_slots("(1, 0, 0) NONE (0, 0, 3)")
            .expect("mixed vector/none slots");
        assert_eq!(
            slots,
            vec![Some([1.0, 0.0, 0.0]), None, Some([0.0, 0.0, 3.0])]
        );
    }

    #[test]
    fn parse_space_direction_slots_rejects_none_prefixed_token() {
        let err = parse_space_direction_slots("nonesuch (1, 0, 0)")
            .expect_err("a none-prefixed word is not the none token");
        assert!(
            err.to_string().contains("outside vector group"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn parse_space_directions_rejects_wrong_spatial_count_after_none() {
        let err = parse_space_directions("none (1, 0, 0) (0, 1, 0)")
            .expect_err("2 spatial vectors are not enough");
        assert!(
            err.to_string().contains("must contain 3 spatial vectors"),
            "unexpected: {err}"
        );
    }

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
