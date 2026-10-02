//! Typed scalar I/O of the legacy structured-points reader and writer
//! (ADR 0053): the stored type survives the round trip, conversions follow the
//! caller's policy, and damaged payloads are refused.

use super::*;
use anyhow::Result;
use coeus_core::SequentialBackend;
use consus_core::ByteOrder;
use ritk_codecs::sample::{write_samples, Cast, Exact};
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::{Debug, Display};
use tempfile::tempdir;

const FIXTURE_NAME: &str = "samples.vtk";

/// A 2x2x2 image of `values` on a grid with distinguishable axes.
fn image_of<T: Sample>(values: &[T; 8]) -> Result<Image<T, SequentialBackend, 3>> {
    Image::from_flat_on(
        values.to_vec(),
        [2, 2, 2],
        Point::new([1.0, -2.0, 3.5]),
        Spacing::new([0.5, 0.75, 1.25]),
        Direction::identity(),
        &SequentialBackend,
    )
}

/// `values` packed big-endian, to compare samples bit for bit.
fn packed<T: Sample>(values: &[T]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    write_samples(values, ByteOrder::BigEndian, &mut bytes)?;
    Ok(bytes)
}

/// The header of a 2x2x2 structured-points file holding `name` scalars.
fn header(encoding: &str, name: &str, components: &str) -> Vec<u8> {
    format!(
        "# vtk DataFile Version 3.0\nfixture\n{encoding}\nDATASET STRUCTURED_POINTS\n\
         DIMENSIONS 2 2 2\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA 8\n\
         SCALARS s {name}{components}\nLOOKUP_TABLE default\n"
    )
    .into_bytes()
}

/// Write `contents` into a fresh temporary directory and return both.
fn fixture(contents: &[u8]) -> Result<(tempfile::TempDir, std::path::PathBuf)> {
    let directory = tempdir()?;
    let path = directory.path().join(FIXTURE_NAME);
    std::fs::write(&path, contents)?;
    Ok((directory, path))
}

/// Writing `values` stores them in `T`, names `name` on the `SCALARS` line, and
/// reading them back in `T` returns every sample bit for bit.
fn round_trips_in_the_stored_type<T: Sample + Debug>(name: &str, values: [T; 8]) -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join(FIXTURE_NAME);
    let image = image_of(&values)?;

    VtkWriter::new(SequentialBackend).write(&path, &image)?;

    let bytes = std::fs::read(&path)?;
    let scalars_line = format!("SCALARS scalars {name} 1\n");
    let text_end = bytes.len() - 8 * T::TYPE.byte_width();
    assert!(
        String::from_utf8_lossy(&bytes[..text_end]).contains(&scalars_line),
        "{}: header names {name}",
        T::TYPE
    );
    assert_eq!(&bytes[text_end..], packed(&values)?, "{} payload", T::TYPE);

    let loaded: Image<T, _, 3> = VtkReader::new(SequentialBackend).read(&path, Exact)?;
    assert_eq!(loaded.shape(), [2, 2, 2]);
    assert_eq!(
        packed(loaded.data_slice()?)?,
        packed(&values)?,
        "{}",
        T::TYPE
    );
    assert_eq!(*loaded.origin(), Point::new([1.0, -2.0, 3.5]));
    assert_eq!(*loaded.spacing(), Spacing::new([0.5, 0.75, 1.25]));
    Ok(())
}

#[test]
fn every_sample_type_round_trips_bit_for_bit_under_its_legacy_name() -> Result<()> {
    round_trips_in_the_stored_type("unsigned_char", [0_u8, 1, 2, 127, 128, 200, 254, 255])?;
    round_trips_in_the_stored_type("char", [i8::MIN, -100, -1, 0, 1, 2, 100, i8::MAX])?;
    round_trips_in_the_stored_type(
        "unsigned_short",
        [0_u16, 1, 255, 256, 32_768, 40_000, 65_534, u16::MAX],
    )?;
    round_trips_in_the_stored_type("short", [i16::MIN, -1024, -1, 0, 1, 3071, 30_000, i16::MAX])?;
    round_trips_in_the_stored_type(
        "unsigned_int",
        [
            0_u32,
            1,
            16_777_216,
            16_777_217,
            70_000,
            3_000_000_000,
            4_294_967_294,
            u32::MAX,
        ],
    )?;
    round_trips_in_the_stored_type(
        "int",
        [
            i32::MIN,
            -16_777_217,
            -1,
            0,
            1,
            16_777_217,
            70_000,
            i32::MAX,
        ],
    )?;
    round_trips_in_the_stored_type(
        "vtktypeuint64",
        [
            0_u64,
            1,
            16_777_217,
            (1 << 53) + 1,
            1 << 63,
            u64::MAX - 1,
            u64::MAX,
            42,
        ],
    )?;
    round_trips_in_the_stored_type(
        "vtktypeint64",
        [
            i64::MIN,
            -(1 << 53) - 1,
            -1,
            0,
            1,
            16_777_217,
            (1 << 53) + 1,
            i64::MAX,
        ],
    )?;
    round_trips_in_the_stored_type(
        "float",
        [
            -0.0_f32,
            0.0,
            f32::MIN_POSITIVE,
            1.0 / 7.0,
            -123_456.79,
            f32::MAX,
            f32::MIN,
            f32::EPSILON,
        ],
    )?;
    round_trips_in_the_stored_type(
        "double",
        [
            0.1_f64,
            -0.0,
            f64::MIN_POSITIVE,
            1.0 / 7.0,
            -123_456.789_012_345,
            f64::MAX,
            f64::MIN,
            f64::EPSILON,
        ],
    )
}

/// An ASCII file holding `values` as decimal text reads back in `T` exactly.
fn ascii_decodes_in_the_stored_type<T: Sample + Display + Debug>(
    name: &str,
    values: [T; 8],
) -> Result<()> {
    let mut contents = header("ASCII", name, "");
    for row in values.chunks(4) {
        let line: Vec<String> = row.iter().map(ToString::to_string).collect();
        contents.extend_from_slice(format!("{}\n", line.join(" ")).as_bytes());
    }
    let (_directory, path) = fixture(&contents)?;

    let loaded: Image<T, _, 3> = read_vtk(&path, &SequentialBackend, Exact)?;

    assert_eq!(
        packed(loaded.data_slice()?)?,
        packed(&values)?,
        "{}",
        T::TYPE
    );
    Ok(())
}

#[test]
fn ascii_scalars_decode_in_the_stored_type() -> Result<()> {
    ascii_decodes_in_the_stored_type("unsigned_char", [0_u8, 1, 2, 127, 128, 200, 254, 255])?;
    ascii_decodes_in_the_stored_type("signed_char", [i8::MIN, -100, -1, 0, 1, 2, 100, i8::MAX])?;
    ascii_decodes_in_the_stored_type("unsigned_short", [0_u16, 1, 2, 3, 4, 5, 6, u16::MAX])?;
    ascii_decodes_in_the_stored_type("short", [i16::MIN, -1, 0, 1, 2, 3, 4, i16::MAX])?;
    ascii_decodes_in_the_stored_type("unsigned_int", [0_u32, 1, 2, 3, 16_777_217, 5, 6, u32::MAX])?;
    ascii_decodes_in_the_stored_type(
        "int",
        [i32::MIN, -16_777_217, -1, 0, 1, 16_777_217, 2, i32::MAX],
    )?;
    ascii_decodes_in_the_stored_type(
        "vtktypeuint64",
        [0_u64, 1, 2, 3, (1 << 53) + 1, 5, 6, u64::MAX],
    )?;
    ascii_decodes_in_the_stored_type(
        "vtktypeint64",
        [
            i64::MIN,
            -(1 << 53) - 1,
            -1,
            0,
            1,
            2,
            (1 << 53) + 1,
            i64::MAX,
        ],
    )?;
    ascii_decodes_in_the_stored_type(
        "float",
        [
            -0.0_f32,
            0.1,
            1.0 / 7.0,
            3.5,
            -4.25,
            5.0,
            f32::MAX,
            f32::MIN,
        ],
    )?;
    ascii_decodes_in_the_stored_type(
        "double",
        [
            0.1_f64,
            -0.0,
            1.0 / 7.0,
            3.5,
            -4.25,
            5.0,
            f64::MAX,
            f64::MIN,
        ],
    )
}

/// A binary payload is big-endian: the legacy bytes `01 02` are 258 as a short.
#[test]
fn binary_payload_is_big_endian_in_the_declared_width() -> Result<()> {
    let mut contents = header("BINARY", "short", " 1");
    for value in [258_i16, -2, 0, 1, 2, 3, 4, 5] {
        contents.extend_from_slice(&value.to_be_bytes());
    }
    let (_directory, path) = fixture(&contents)?;

    let loaded: Image<i16, _, 3> = read_vtk(&path, &SequentialBackend, Exact)?;

    assert_eq!(loaded.data_slice()?, [258_i16, -2, 0, 1, 2, 3, 4, 5]);
    Ok(())
}

#[test]
fn exact_widens_and_refuses_what_it_cannot_hold() -> Result<()> {
    let mut contents = header("BINARY", "int", "");
    for value in [16_777_217_i32, 0, 1, 2, 3, 4, 5, -16_777_217] {
        contents.extend_from_slice(&value.to_be_bytes());
    }
    let (_directory, path) = fixture(&contents)?;

    let refused = read_vtk::<f32, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("i32 does not widen to f32");
    assert!(
        format!("{refused:#}").contains("i32 samples do not all have exact f32 values"),
        "{refused:#}"
    );
    let widened: Image<f64, _, 3> = read_vtk(&path, &SequentialBackend, Exact)?;
    assert_eq!(widened.data_slice()?[0], 16_777_217.0_f64);
    assert_eq!(widened.data_slice()?[7], -16_777_217.0_f64);
    Ok(())
}

#[test]
fn cast_rounds_to_the_requested_type() -> Result<()> {
    let mut contents = header("BINARY", "double", "");
    for value in [0.1_f64, 16_777_217.0, 1.5, 2.0, 3.0, 4.0, 5.0, -0.1] {
        contents.extend_from_slice(&value.to_be_bytes());
    }
    let (_directory, path) = fixture(&contents)?;

    let rounded: Image<f32, _, 3> = read_vtk(&path, &SequentialBackend, Cast)?;
    assert_eq!(
        rounded.data_slice()?,
        [0.1_f32, 16_777_216.0, 1.5, 2.0, 3.0, 4.0, 5.0, -0.1]
    );
    let truncated: Image<i32, _, 3> = read_vtk(&path, &SequentialBackend, Cast)?;
    assert_eq!(truncated.data_slice()?, [0, 16_777_217, 1, 2, 3, 4, 5, 0]);
    Ok(())
}

#[test]
fn truncated_binary_payload_is_refused_at_the_missing_sample() -> Result<()> {
    let mut contents = header("BINARY", "int", "");
    for value in 0..7_i32 {
        contents.extend_from_slice(&value.to_be_bytes());
    }
    contents.extend_from_slice(&[0, 0]);
    let (_directory, path) = fixture(&contents)?;

    let refused = read_vtk::<i32, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("seven and a half samples of eight");

    assert!(
        format!("{refused:#}").contains("stream ended at i32 sample 7 of 8"),
        "{refused:#}"
    );
    Ok(())
}

#[test]
fn truncated_ascii_payload_is_refused_with_the_count() -> Result<()> {
    let mut contents = header("ASCII", "int", "");
    contents.extend_from_slice(b"1 2 3\n4 5 6 7\n");
    let (_directory, path) = fixture(&contents)?;

    let refused = read_vtk::<i32, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("seven values of eight");

    assert!(
        format!("{refused:#}").contains("expected 8 i32 values, got 7"),
        "{refused:#}"
    );
    Ok(())
}

#[test]
fn ascii_token_outside_the_stored_type_is_refused() -> Result<()> {
    let mut contents = header("ASCII", "unsigned_char", "");
    contents.extend_from_slice(b"1 2 3 4 5 6 7 256\n");
    let (_directory, path) = fixture(&contents)?;

    let refused = read_vtk::<u8, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("256 is not an unsigned_char");

    assert!(
        format!("{refused:#}").contains("bad u8 token '256'"),
        "{refused:#}"
    );
    Ok(())
}

#[test]
fn types_without_a_sample_type_are_refused_by_name() -> Result<()> {
    for (name, needle) in [
        ("bit", "'bit'"),
        ("long", "'long'"),
        ("unsigned_long", "'unsigned_long'"),
    ] {
        let mut contents = header("BINARY", name, "");
        contents.extend_from_slice(&[0; 64]);
        let (_directory, path) = fixture(&contents)?;

        let refused =
            read_vtk::<f32, _, _, _>(&path, &SequentialBackend, Cast).expect_err("no sample type");

        assert!(
            format!("{refused:#}").contains(needle),
            "{name}: {refused:#}"
        );
    }
    Ok(())
}

/// `vtkDataReader` reads `vtkidtype` as 4-byte big-endian `int` data, so the
/// samples are those of the 4-byte signed integer, in either encoding.
#[test]
fn id_type_scalars_read_as_four_byte_integers_in_both_encodings() -> Result<()> {
    let values = [
        i32::MIN,
        -16_777_217,
        -1,
        0,
        1,
        16_777_217,
        70_000,
        i32::MAX,
    ];
    ascii_decodes_in_the_stored_type("vtkidtype", values)?;
    ascii_decodes_in_the_stored_type("VTKIDTYPE", values)?;

    let mut contents = header("BINARY", "vtkidtype", "");
    contents.extend_from_slice(&packed(&values)?);
    let (_directory, path) = fixture(&contents)?;

    let loaded: Image<i32, _, 3> = read_vtk(&path, &SequentialBackend, Exact)?;

    assert_eq!(packed(loaded.data_slice()?)?, packed(&values)?);
    Ok(())
}

#[test]
fn multi_component_scalars_are_refused() -> Result<()> {
    let mut contents = header("BINARY", "float", " 3");
    contents.extend_from_slice(&[0; 96]);
    let (_directory, path) = fixture(&contents)?;

    let refused = read_vtk::<f32, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("three components per point");

    assert!(
        format!("{refused:#}")
            .contains("has 3 components; structured-points reading supports exactly one"),
        "{refused:#}"
    );
    Ok(())
}

#[test]
fn hostile_dimensions_do_not_reserve_the_declared_payload() -> Result<()> {
    let mut contents = header("BINARY", "double", "");
    contents.extend_from_slice(&[0; 16]);
    let text = String::from_utf8(contents)?
        .replace("DIMENSIONS 2 2 2", "DIMENSIONS 1000000 1000000 1000")
        .replace("POINT_DATA 8", "POINT_DATA 1000000000000000");
    let (_directory, path) = fixture(text.as_bytes())?;

    let refused = read_vtk::<f64, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("payload is sixteen bytes");

    assert!(
        format!("{refused:#}").contains("stream ended at f64 sample 2 of"),
        "{refused:#}"
    );
    Ok(())
}

#[test]
fn flat_encoder_rejects_a_mismatched_length_and_overflowing_dimensions() {
    let mut sink = Vec::new();
    let short = encode_vtk_flat(&mut sink, &[1_u16, 2, 3], [1, 2, 2], [0.0; 3], [1.0; 3])
        .expect_err("three samples for four voxels");
    assert!(
        short.to_string().contains("3 elements but expected 4"),
        "{short}"
    );

    let overflow = encode_vtk_flat(
        &mut sink,
        &[0_u8; 0],
        [usize::MAX, 2, 1],
        [0.0; 3],
        [1.0; 3],
    )
    .expect_err("dimension product overflows");
    assert!(
        overflow.to_string().contains("overflows usize"),
        "{overflow}"
    );
}

#[test]
fn flat_reader_returns_stored_values_and_header_geometry() -> Result<()> {
    let image = image_of(&[1_u16, 2, 3, 4, 5, 6, 7, 40_000])?;
    let directory = tempdir()?;
    let path = directory.path().join(FIXTURE_NAME);
    VtkWriter::new(SequentialBackend).write(&path, &image)?;

    let (data, dims, origin, spacing) = read_vtk_flat::<u16, _, _>(&path, Exact)?;

    assert_eq!(data, [1, 2, 3, 4, 5, 6, 7, 40_000]);
    assert_eq!(dims, [2, 2, 2]);
    assert_eq!(origin, [1.0, -2.0, 3.5]);
    assert_eq!(spacing, [0.5, 0.75, 1.25]);
    Ok(())
}
