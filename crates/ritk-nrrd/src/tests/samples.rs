//! Stored sample types: every NRRD numeric type reads and writes as itself.
//!
//! Each case is generic over the sample type and instantiated for all ten by
//! [`for_every_sample_type!`], so a type added to the format inherits the
//! suite. Payloads are encoded here with the codec's own writer, never with the
//! crate's NRRD writer, wherever the case is about reading.

use super::probe::Probe;
use crate::{
    read_nrrd, read_nrrd_series, write_nrrd, write_nrrd_series, write_nrrd_with_data, NrrdReader,
    NrrdWriter,
};
use anyhow::Result;
use coeus_core::SequentialBackend;
use consus_core::ByteOrder;
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_codecs::sample::{write_samples, Cast, Exact, Sample};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::Write;
use std::path::Path;
use tempfile::tempdir;

type TestBackend = SequentialBackend;

const BYTE_ORDERS: [ByteOrder; 2] = [ByteOrder::BigEndian, ByteOrder::LittleEndian];

/// Run the generic case `$case` for each of the ten sample types.
macro_rules! for_every_sample_type {
    ($case:ident) => {{
        $case::<u8>()?;
        $case::<i8>()?;
        $case::<u16>()?;
        $case::<i16>()?;
        $case::<u32>()?;
        $case::<i32>()?;
        $case::<u64>()?;
        $case::<i64>()?;
        $case::<f32>()?;
        $case::<f64>()?;
        Ok(())
    }};
}

/// The packed little-endian bytes of `values`, which compare floats bit for
/// bit and so tell `-0.0` from `0.0`.
fn bits<T: Sample>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_samples(values, ByteOrder::LittleEndian, &mut bytes).expect("write to a vector");
    bytes
}

fn image_of<T: Sample>(values: Vec<T>) -> Result<Image<T, TestBackend, 3>> {
    Image::from_flat_on(
        values,
        [2, 2, 2],
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &TestBackend::default(),
    )
}

fn voxels_of<T: Sample>(image: &Image<T, TestBackend, 3>) -> Vec<T> {
    image.data_slice().expect("contiguous host voxels").to_vec()
}

/// `values` packed in `order`, gzip-compressed when `gzip`.
fn encode_payload<T: Sample>(values: &[T], order: ByteOrder, gzip: bool) -> Vec<u8> {
    let mut packed = Vec::new();
    write_samples(values, order, &mut packed).expect("write to a vector");
    if !gzip {
        return packed;
    }
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&packed).expect("compress to a vector");
    encoder.finish().expect("finish gzip stream")
}

fn endian_name(order: ByteOrder) -> &'static str {
    match order {
        ByteOrder::BigEndian => "big",
        ByteOrder::LittleEndian => "little",
    }
}

/// Write an inline NRRD whose header is `header` (one field per line, ended by
/// the blank line) followed by `payload`.
fn write_file(path: &Path, header: &str, payload: &[u8]) -> Result<()> {
    let mut bytes = format!("NRRD0004\n{header}\n\n").into_bytes();
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes)?;
    Ok(())
}

/// The header of a 2x2x2 volume of `type_name` samples.
fn volume_header(type_name: &str, encoding: &str, endian: &str) -> String {
    format!("type: {type_name}\ndimension: 3\nsizes: 2 2 2\nencoding: {encoding}\nendian: {endian}")
}

fn encoding_name(gzip: bool) -> &'static str {
    if gzip {
        "gzip"
    } else {
        "raw"
    }
}

fn reads_every_encoding_byte_order_and_alias<T: Probe>() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("typed.nrrd");
    for gzip in [false, true] {
        for order in BYTE_ORDERS {
            for name in [T::ALIAS, T::CANONICAL] {
                let header = volume_header(name, encoding_name(gzip), endian_name(order));
                write_file(&path, &header, &encode_payload(&T::VALUES, order, gzip))?;
                let image = read_nrrd::<T, _, _, _>(&path, &backend, Exact)?;
                assert_eq!(
                    bits(&voxels_of(&image)),
                    bits(&T::VALUES),
                    "{} {name} {order:?} gzip={gzip}",
                    T::TYPE
                );
            }
        }
    }
    Ok(())
}

#[test]
fn every_sample_type_reads_in_every_encoding_and_byte_order() -> Result<()> {
    for_every_sample_type!(reads_every_encoding_byte_order_and_alias)
}

fn writes_its_own_type_and_reads_back_bit_for_bit<T: Probe>() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("written.nrrd");
    let image = image_of(T::VALUES.to_vec())?;
    write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains(&format!("type: {}\n", T::CANONICAL)),
        "{} header: {text}",
        T::TYPE
    );
    assert!(text.contains("endian: little\n"));
    let payload_start = bytes.len() - 8 * T::TYPE.byte_width();
    assert_eq!(&bytes[payload_start..], bits(&T::VALUES), "{}", T::TYPE);

    let loaded = read_nrrd::<T, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(bits(&voxels_of(&loaded)), bits(&T::VALUES), "{}", T::TYPE);
    assert_eq!(loaded.origin(), image.origin());
    Ok(())
}

#[test]
fn every_sample_type_writes_itself_and_round_trips_bit_for_bit() -> Result<()> {
    for_every_sample_type!(writes_its_own_type_and_reads_back_bit_for_bit)
}

fn series_keeps_every_volume_in_its_type<T: Probe>() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("series.nrrd");
    let volumes: Vec<Vec<T>> = (0..3)
        .map(|shift| {
            let mut values = T::VALUES.to_vec();
            values.rotate_left(shift);
            values
        })
        .collect();
    let images = volumes
        .iter()
        .map(|values| image_of(values.clone()))
        .collect::<Result<Vec<_>>>()?;
    write_nrrd_series(&path, &images, &backend)?;
    let text = String::from_utf8_lossy(&std::fs::read(&path)?).into_owned();
    assert!(
        text.contains(&format!("type: {}\n", T::CANONICAL)),
        "{text}"
    );
    assert!(text.contains("endian: little\n"), "{text}");

    let loaded = read_nrrd_series::<T, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(loaded.len(), volumes.len());
    for (position, (got, want)) in loaded.iter().zip(&volumes).enumerate() {
        assert_eq!(
            bits(&voxels_of(got)),
            bits(want),
            "{} volume {position}",
            T::TYPE
        );
    }
    Ok(())
}

#[test]
fn every_sample_type_round_trips_as_a_series() -> Result<()> {
    for_every_sample_type!(series_keeps_every_volume_in_its_type)
}

/// A trailing acquisition axis lays the volumes out contiguously; the payload
/// here is big-endian and gzip-compressed, built by the test.
fn trailing_axis_series_reads_in_every_type<T: Probe>() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("trailing.nrrd");
    let mut second = T::VALUES.to_vec();
    second.reverse();
    let payload: Vec<T> = T::VALUES.iter().copied().chain(second.clone()).collect();
    let header = format!(
        "type: {}\ndimension: 4\nsizes: 2 2 2 2\nkinds: domain domain domain list\n\
         space directions: (1,0,0) (0,1,0) (0,0,1) none\nencoding: gzip\nendian: big",
        T::ALIAS
    );
    write_file(
        &path,
        &header,
        &encode_payload(&payload, ByteOrder::BigEndian, true),
    )?;
    let loaded = read_nrrd_series::<T, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(loaded.len(), 2);
    assert_eq!(
        bits(&voxels_of(&loaded[0])),
        bits(&T::VALUES),
        "{}",
        T::TYPE
    );
    assert_eq!(bits(&voxels_of(&loaded[1])), bits(&second), "{}", T::TYPE);
    Ok(())
}

#[test]
fn every_sample_type_reads_a_trailing_acquisition_axis() -> Result<()> {
    for_every_sample_type!(trailing_axis_series_reads_in_every_type)
}

#[test]
fn exact_refuses_a_narrowing_read_and_cast_rounds() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("wide.nrrd");
    write_nrrd(&path, &image_of(i32::VALUES.to_vec())?, &backend)?;

    let err =
        read_nrrd::<f32, _, _, _>(&path, &backend, Exact).expect_err("i32 does not widen to f32");
    assert!(
        format!("{err:#}").contains("i32 samples do not all have exact f32 values"),
        "{err:#}"
    );
    let cast = read_nrrd::<f32, _, _, _>(&path, &backend, Cast)?;
    // Rounded to nearest even: 16_777_217 lies between 16_777_216 and
    // 16_777_218 and ties to the even mantissa; i32::MAX rounds up to 2^31.
    assert_eq!(
        voxels_of(&cast),
        [
            -2_147_483_648.0_f32,
            -16_777_216.0,
            -1.0,
            0.0,
            1.0,
            16_777_216.0,
            70_000.0,
            2_147_483_648.0
        ]
    );

    write_nrrd(&path, &image_of(f64::VALUES.to_vec())?, &backend)?;
    let err =
        read_nrrd::<f32, _, _, _>(&path, &backend, Exact).expect_err("f64 does not widen to f32");
    assert!(
        format!("{err:#}").contains("f64 samples do not all have exact f32 values"),
        "{err:#}"
    );
    let cast = read_nrrd::<f32, _, _, _>(&path, &backend, Cast)?;
    assert_eq!(voxels_of(&cast)[0], 0.1_f32);
    Ok(())
}

#[test]
fn exact_widens_a_stored_type_whose_values_all_survive() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("narrow.nrrd");
    write_nrrd(&path, &image_of(u8::VALUES.to_vec())?, &backend)?;
    let widened = read_nrrd::<f64, _, _, _>(&path, &backend, Exact)?;
    let expected: Vec<f64> = u8::VALUES.iter().map(|&value| f64::from(value)).collect();
    assert_eq!(voxels_of(&widened), expected);
    let stored = read_nrrd_series::<i64, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(stored.len(), 1);
    assert_eq!(
        voxels_of(&stored[0]),
        u8::VALUES.iter().map(|&v| i64::from(v)).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn an_unknown_endian_value_is_refused() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("endian.nrrd");
    for value in ["middle", "", "msbfirst", "bigger"] {
        let header = volume_header("short", "raw", value);
        write_file(
            &path,
            &header,
            &encode_payload(&i16::VALUES, ByteOrder::BigEndian, false),
        )?;
        let err = read_nrrd::<i16, _, _, _>(&path, &backend, Exact)
            .expect_err("an unknown endian must not read as a guessed order");
        assert!(
            err.to_string()
                .contains(&format!("Unsupported NRRD endian '{value}'")),
            "{value:?}: {err}"
        );
        let err = read_nrrd_series::<i16, _, _, _>(&path, &backend, Exact)
            .expect_err("the series reader refuses it too");
        assert!(err.to_string().contains("Unsupported NRRD endian"), "{err}");
    }
    Ok(())
}

#[test]
fn endian_is_case_insensitive() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("endian.nrrd");
    write_file(
        &path,
        &volume_header("short", "raw", "BIG"),
        &encode_payload(&i16::VALUES, ByteOrder::BigEndian, false),
    )?;
    let image = read_nrrd::<i16, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(voxels_of(&image), i16::VALUES);
    Ok(())
}

/// A header without `endian`: the specification requires the field exactly
/// when the type is wider than one byte (format.html section 5, `endian`), and
/// Teem `formatNRRD.c` refuses such a file with "require endian info".
fn missing_endian_is_refused_exactly_for_multi_byte_types<T: Probe>() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("no_endian.nrrd");
    for gzip in [false, true] {
        let header = format!(
            "type: {}\ndimension: 3\nsizes: 2 2 2\nencoding: {}",
            T::CANONICAL,
            encoding_name(gzip)
        );
        write_file(
            &path,
            &header,
            &encode_payload(&T::VALUES, ByteOrder::LittleEndian, gzip),
        )?;
        if T::TYPE.byte_width() == 1 {
            let image = read_nrrd::<T, _, _, _>(&path, &backend, Exact)?;
            assert_eq!(
                bits(&voxels_of(&image)),
                bits(&T::VALUES),
                "{} gzip={gzip}",
                T::TYPE
            );
            let series = read_nrrd_series::<T, _, _, _>(&path, &backend, Exact)?;
            assert_eq!(series.len(), 1, "{} gzip={gzip}", T::TYPE);
        } else {
            let err = read_nrrd::<T, _, _, _>(&path, &backend, Exact)
                .expect_err("a multi-byte payload without 'endian' has no defined order");
            assert!(
                err.to_string().contains("'endian' field is required"),
                "{} gzip={gzip}: {err}",
                T::TYPE
            );
            let err = read_nrrd_series::<T, _, _, _>(&path, &backend, Exact)
                .expect_err("the series reader refuses it too");
            assert!(
                err.to_string().contains("'endian' field is required"),
                "{} gzip={gzip}: {err}",
                T::TYPE
            );
        }
    }
    Ok(())
}

#[test]
fn a_missing_endian_is_refused_for_multi_byte_types_and_accepted_for_one_byte() -> Result<()> {
    for_every_sample_type!(missing_endian_is_refused_exactly_for_multi_byte_types)
}

#[test]
fn block_and_unspecified_types_are_refused() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("block.nrrd");
    for (name, needle) in [
        ("block", "'block'"),
        ("long double", "Unsupported NRRD type: 'long double'"),
    ] {
        write_file(&path, &volume_header(name, "raw", "little"), &[0; 128])?;
        let err = read_nrrd::<f32, _, _, _>(&path, &backend, Exact).expect_err(name);
        assert!(err.to_string().contains(needle), "{name}: {err}");
    }
    Ok(())
}

#[test]
fn a_payload_shorter_than_the_header_declares_is_refused_in_every_encoding() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("short.nrrd");
    for gzip in [false, true] {
        let mut payload = encode_payload(&u64::VALUES, ByteOrder::LittleEndian, gzip);
        if gzip {
            payload.truncate(payload.len() - 12);
        } else {
            payload.truncate(payload.len() - 1);
        }
        write_file(
            &path,
            &volume_header("uint64", encoding_name(gzip), "little"),
            &payload,
        )?;
        let err = read_nrrd::<u64, _, _, _>(&path, &backend, Exact)
            .expect_err("a short payload must not read");
        assert!(
            format!("{err:#}").contains("8 u64 samples need 64 bytes"),
            "gzip={gzip}: {err:#}"
        );
    }
    Ok(())
}

#[test]
fn a_raw_payload_with_excess_bytes_is_refused_before_decode() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("excess-raw.nrrd");
    let mut payload = encode_payload(&u64::VALUES, ByteOrder::LittleEndian, false);
    payload.extend_from_slice(&17_u64.to_le_bytes());
    write_file(&path, &volume_header("uint64", "raw", "little"), &payload)?;

    let error = read_nrrd::<u64, _, _, _>(&path, &backend, Exact)
        .expect_err("raw bytes beyond the declared sample count must be rejected");
    assert!(
        format!("{error:#}").contains("payload has 72 bytes; expected 64"),
        "{error:#}"
    );
    Ok(())
}

#[test]
fn gzip_trailer_is_verified_and_excess_decompressed_samples_are_rejected() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("gzip-integrity.nrrd");
    let header = volume_header("uint64", "gzip", "little");

    let mut invalid_crc = encode_payload(&u64::VALUES, ByteOrder::LittleEndian, true);
    let crc_offset = invalid_crc.len() - 8;
    invalid_crc[crc_offset] ^= 1;
    write_file(&path, &header, &invalid_crc)?;
    let error = read_nrrd::<u64, _, _, _>(&path, &backend, Exact)
        .expect_err("the gzip checksum must be verified");
    assert!(format!("{error:#}").contains("corrupt"), "{error:#}");

    let mut truncated_trailer = encode_payload(&u64::VALUES, ByteOrder::LittleEndian, true);
    truncated_trailer.truncate(truncated_trailer.len() - 1);
    write_file(&path, &header, &truncated_trailer)?;
    assert!(read_nrrd::<u64, _, _, _>(&path, &backend, Exact).is_err());

    let mut excess_values = u64::VALUES.to_vec();
    excess_values.push(17);
    write_file(
        &path,
        &header,
        &encode_payload(&excess_values, ByteOrder::LittleEndian, true),
    )?;
    let error = read_nrrd::<u64, _, _, _>(&path, &backend, Exact)
        .expect_err("decompressed data past the declared sample count must be rejected");
    assert!(
        format!("{error:#}").contains("beyond the declared sample count"),
        "{error:#}"
    );
    Ok(())
}

#[test]
fn caller_supplied_values_are_stored_in_their_type() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("with_data.nrrd");
    let image = image_of(vec![0_u64; 8])?;
    write_nrrd_with_data(&path, &image, &u64::VALUES)?;
    let loaded = read_nrrd::<u64, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(voxels_of(&loaded), u64::VALUES);

    let missing = dir.path().join("missing.nrrd");
    let err = write_nrrd_with_data(&missing, &image, &[1_u64; 7])
        .expect_err("a short payload is refused");
    assert!(err.to_string().contains("requires 8"), "{err}");
    assert!(!missing.exists());
    Ok(())
}

#[test]
fn the_reader_and_writer_structs_carry_the_sample_type() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("structs.nrrd");
    let image = image_of(i16::VALUES.to_vec())?;

    NrrdWriter::new(TestBackend::default()).write(&path, &image)?;
    let loaded = NrrdReader.read::<i16, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(voxels_of(&loaded), i16::VALUES);
    Ok(())
}
