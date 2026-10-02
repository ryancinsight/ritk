//! Typed sample I/O: every stored type reads and writes in its own type.

use std::fmt::Debug;
use std::path::{Path, PathBuf};

use anyhow::Result;
use coeus_core::{ComputeBackend, CpuAddressableStorage, SequentialBackend};
use consus_core::ByteOrder;
use ritk_codecs::sample::{write_samples, Cast, Exact, Sample};
use ritk_core::rejection::assert_rejects;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::{tempdir, TempDir};

use crate::header::datatype_name;
use crate::tests::mrtrix_inline_file;
use crate::{read_mif, read_mif_series, write_mif, write_mif_series};

/// Voxels in every fixture volume: shape `[2, 3, 2]`.
const VOXELS: usize = 12;
const SHAPE: [usize; 3] = [2, 3, 2];

fn image_of<T: Sample>(values: &[T]) -> Result<Image<T, SequentialBackend, 3>> {
    Image::from_flat_on(
        values.to_vec(),
        SHAPE,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &SequentialBackend,
    )
}

/// The bytes `values` occupy stored little-endian, which is bit-exact for every
/// type including `-0.0`.
fn stored_bytes<T: Sample>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_samples(values, ByteOrder::LittleEndian, &mut bytes)
        .expect("a vector accepts every byte");
    bytes
}

/// A `.mif` holding `values` stored as `T` in `order`, built by hand so the
/// byte order is a test parameter rather than the writer's choice.
fn hand_built_file<T: Sample>(dir: &TempDir, order: ByteOrder, values: &[T]) -> PathBuf {
    let path = dir.path().join("hand_built.mif");
    let header = format!(
        "mrtrix image\ndim: 2 3 2\nvox: 1 1 1\nlayout: +0,+1,+2\ndatatype: {}\n",
        datatype_name(T::TYPE, order)
    );
    let mut payload = Vec::new();
    write_samples(values, order, &mut payload).expect("a vector accepts every byte");
    std::fs::write(&path, mrtrix_inline_file(&header, &payload)).expect("write fixture");
    path
}

fn header_text(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("read written file");
    let end = bytes
        .windows(4)
        .position(|window| window == b"END\n")
        .expect("a written file has an END line");
    String::from_utf8(bytes[..end].to_vec()).expect("a .mif header is text")
}

/// `values` written by `write_mif` and read back as `T` match bit for bit, and
/// the header names the type of `T`.
fn writer_round_trips<T>(values: [T; VOXELS], header_datatype: &str) -> Result<()>
where
    T: Sample + Debug,
    <SequentialBackend as ComputeBackend>::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    let dir = tempdir()?;
    let path = dir.path().join("round_trip.mif");
    write_mif(&path, &image_of(&values)?, &SequentialBackend)?;

    let datatype_line = format!("datatype: {header_datatype}\n");
    assert!(
        header_text(&path).contains(&datatype_line),
        "{} header is {:?}",
        T::TYPE,
        header_text(&path)
    );

    let back = read_mif::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(back.shape(), SHAPE);
    assert_eq!(back.data_slice()?, values, "{}", T::TYPE);
    assert_eq!(stored_bytes(back.data_slice()?), stored_bytes(&values));
    Ok(())
}

/// Values of each stored type, chosen to separate a correct decoder from the
/// width-only one: unsigned values past the signed range, integers past the
/// 24- and 53-bit significands, and floats with no exact narrower value.
const U8_VALUES: [u8; VOXELS] = [0, 1, 2, 10, 20, 50, 100, 127, 128, 200, 254, 255];
const I8_VALUES: [i8; VOXELS] = [i8::MIN, -100, -2, -1, 0, 1, 2, 3, 50, 100, 126, i8::MAX];
const U16_VALUES: [u16; VOXELS] = [
    0,
    1,
    255,
    256,
    1000,
    32_767,
    32_768,
    40_000,
    50_000,
    60_000,
    65_534,
    u16::MAX,
];
const I16_VALUES: [i16; VOXELS] = [
    i16::MIN,
    -1000,
    -256,
    -1,
    0,
    1,
    255,
    256,
    1000,
    3071,
    32_766,
    i16::MAX,
];
const U32_VALUES: [u32; VOXELS] = [
    0,
    1,
    255,
    65_536,
    16_777_217,
    1 << 24,
    1 << 30,
    3_000_000_000,
    4_000_000_000,
    4_294_967_294,
    7,
    u32::MAX,
];
const I32_VALUES: [i32; VOXELS] = [
    i32::MIN,
    -16_777_217,
    -100_000,
    -1,
    0,
    1,
    16_777_217,
    50_000,
    75_000,
    3,
    4,
    i32::MAX,
];
const U64_VALUES: [u64; VOXELS] = [
    0,
    1,
    1 << 24,
    (1 << 53) + 1,
    1 << 60,
    u64::MAX - 1,
    u64::MAX,
    255,
    65_536,
    4_294_967_296,
    9,
    10,
];
const I64_VALUES: [i64; VOXELS] = [
    i64::MIN,
    -(1 << 53) - 1,
    -4_294_967_297,
    -1,
    0,
    1,
    (1 << 53) + 1,
    4_294_967_297,
    9,
    10,
    11,
    i64::MAX,
];
const F32_VALUES: [f32; VOXELS] = [
    std::f32::consts::PI,
    std::f32::consts::E,
    -0.0,
    f32::MIN_POSITIVE,
    1.0 / 7.0,
    -std::f32::consts::FRAC_PI_2,
    f32::MAX,
    f32::MIN,
    1.0 / 3.0,
    0.0,
    f32::EPSILON,
    -1.0,
];
// 0.1 and 1/3 have no exact f32 value.
const F64_VALUES: [f64; VOXELS] = [
    0.1,
    1.0 / 3.0,
    -0.0,
    f64::MIN_POSITIVE,
    std::f64::consts::PI,
    f64::MAX,
    f64::MIN,
    f64::EPSILON,
    -1e-300,
    1e300,
    5e-324,
    -0.1,
];

#[test]
fn every_stored_type_round_trips_through_the_writer_in_its_own_type() -> Result<()> {
    writer_round_trips(U8_VALUES, "UInt8")?;
    writer_round_trips(I8_VALUES, "Int8")?;
    writer_round_trips(U16_VALUES, "UInt16LE")?;
    writer_round_trips(I16_VALUES, "Int16LE")?;
    writer_round_trips(U32_VALUES, "UInt32LE")?;
    writer_round_trips(I32_VALUES, "Int32LE")?;
    writer_round_trips(U64_VALUES, "UInt64LE")?;
    writer_round_trips(I64_VALUES, "Int64LE")?;
    writer_round_trips(F32_VALUES, "Float32LE")?;
    writer_round_trips(F64_VALUES, "Float64LE")
}

/// `values` stored as `T` in `order` read back as `T`, bit for bit.
fn reads_in_the_stored_byte_order<T: Sample + Debug>(
    order: ByteOrder,
    values: [T; VOXELS],
) -> Result<()> {
    let dir = tempdir()?;
    let path = hand_built_file(&dir, order, &values);
    let image = read_mif::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(image.shape(), SHAPE);
    assert_eq!(image.data_slice()?, values, "{} {order:?}", T::TYPE);
    assert_eq!(stored_bytes(image.data_slice()?), stored_bytes(&values));
    Ok(())
}

/// The width-only decoder read `U16_VALUES` as `i16`, `U32_VALUES` and
/// `I32_VALUES` as float bits, and `I8_VALUES` as `u8`.
#[test]
fn every_stored_type_reads_in_both_byte_orders() -> Result<()> {
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        reads_in_the_stored_byte_order(order, U8_VALUES)?;
        reads_in_the_stored_byte_order(order, I8_VALUES)?;
        reads_in_the_stored_byte_order(order, U16_VALUES)?;
        reads_in_the_stored_byte_order(order, I16_VALUES)?;
        reads_in_the_stored_byte_order(order, U32_VALUES)?;
        reads_in_the_stored_byte_order(order, I32_VALUES)?;
        reads_in_the_stored_byte_order(order, U64_VALUES)?;
        reads_in_the_stored_byte_order(order, I64_VALUES)?;
        reads_in_the_stored_byte_order(order, F32_VALUES)?;
        reads_in_the_stored_byte_order(order, F64_VALUES)?;
    }
    Ok(())
}

/// `u16` widens to `f32` exactly; `i32` does not, so `Exact` refuses it and
/// `Cast` rounds it.
#[test]
fn exact_widens_and_refuses_by_the_stored_type_and_cast_rounds() -> Result<()> {
    let dir = tempdir()?;
    let backend = SequentialBackend;

    let widened = read_mif::<f32, _, _, _>(
        &hand_built_file(&dir, ByteOrder::BigEndian, &U16_VALUES),
        &backend,
        Exact,
    )?;
    assert_eq!(widened.data_slice()?, U16_VALUES.map(f32::from));

    let ints = [16_777_217_i32, -3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let path = hand_built_file(&dir, ByteOrder::LittleEndian, &ints);
    assert_rejects(
        read_mif::<f32, _, _, _>(&path, &backend, Exact),
        "i32 samples do not all have exact f32 values",
    );
    let cast = read_mif::<f32, _, _, _>(&path, &backend, Cast)?;
    assert_eq!(cast.data_slice()?[..2], [16_777_216.0, -3.0]);

    let wide = read_mif::<i64, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(wide.data_slice()?[0], 16_777_217_i64);
    Ok(())
}

/// A series of `T` frames interleaved by the writer comes back frame for frame.
fn series_round_trips<T>(frames: [[T; VOXELS]; 3]) -> Result<()>
where
    T: Sample + Debug,
    <SequentialBackend as ComputeBackend>::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    let volumes = frames
        .iter()
        .map(|frame| image_of(frame))
        .collect::<Result<Vec<_>>>()?;
    let dir = tempdir()?;
    let path = dir.path().join("series.mif");
    write_mif_series(&path, &volumes, &SequentialBackend)?;
    let back = read_mif_series::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(back.len(), frames.len());
    for (volume, frame) in back.iter().zip(&frames) {
        assert_eq!(volume.data_slice()?, frame, "{}", T::TYPE);
    }
    Ok(())
}

#[test]
fn a_series_round_trips_frame_for_frame_in_its_own_type() -> Result<()> {
    series_round_trips([U16_VALUES, U16_VALUES.map(|v| !v), [7; VOXELS]])?;
    series_round_trips([I32_VALUES, I32_VALUES.map(|v| v / 2), [-7; VOXELS]])?;
    series_round_trips([U64_VALUES, U64_VALUES.map(|v| !v), [7; VOXELS]])?;
    series_round_trips([F64_VALUES, F64_VALUES.map(|v| -v), [0.5; VOXELS]])
}

/// A file whose `datatype` is not one real scalar per voxel is rejected naming
/// the reason, before any voxel is read.
#[test]
fn bit_and_complex_files_are_rejected_by_the_reader() -> Result<()> {
    let dir = tempdir()?;
    for datatype in ["Bit", "CFloat32LE", "CFloat64BE"] {
        let path = dir.path().join("unsupported.mif");
        let header = format!(
            "mrtrix image\ndim: 2 2 2\nvox: 1 1 1\nlayout: +0,+1,+2\ndatatype: {datatype}\n"
        );
        std::fs::write(&path, mrtrix_inline_file(&header, &[]))?;
        assert_rejects(
            read_mif::<f32, _, _, _>(&path, &SequentialBackend, Cast),
            "Bit and complex voxels are not one",
        );
    }
    Ok(())
}

/// A payload one sample short of `dim` fails as truncated at the narrowest and
/// widest stored widths, rather than decoding the samples it has.
#[test]
fn a_truncated_payload_is_rejected_in_every_stored_width() -> Result<()> {
    let dir = tempdir()?;
    let short_u8 = hand_built_file(&dir, ByteOrder::LittleEndian, &[1_u8; VOXELS - 1]);
    assert_rejects(
        read_mif::<u8, _, _, _>(&short_u8, &SequentialBackend, Exact),
        "payload is truncated",
    );
    let short_f64 = hand_built_file(&dir, ByteOrder::BigEndian, &[1.0_f64; VOXELS - 1]);
    assert_rejects(
        read_mif::<f64, _, _, _>(&short_f64, &SequentialBackend, Exact),
        "payload is truncated",
    );
    Ok(())
}

/// A detached payload at a non-zero offset decodes in the stored type and byte
/// order.
#[test]
fn a_detached_payload_decodes_in_the_stored_type_after_its_offset() -> Result<()> {
    let dir = tempdir()?;
    let mut payload = vec![0xEE_u8; 5];
    write_samples(&U32_VALUES, ByteOrder::BigEndian, &mut payload)?;
    std::fs::write(dir.path().join("volume.dat"), payload)?;
    let path = dir.path().join("volume.mif");
    std::fs::write(
        &path,
        "mrtrix image\ndim: 2 3 2\nvox: 1 1 1\nlayout: +0,+1,+2\ndatatype: UInt32BE\n\
         file: volume.dat 5\nEND\n",
    )?;
    let image = read_mif::<u32, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(image.data_slice()?, U32_VALUES);
    Ok(())
}

/// A file laid out byte for byte as MRtrix writes one: `file: . 96`, an offset
/// that is a multiple of four counted from the start of the file, with zero
/// padding between the `END` line and the voxels.
///
/// Read as a distance past the header, the offset 96 would skip 96 more bytes
/// and the payload would be truncated; read as a distance from the start of
/// the file but ignoring the padding, the first voxels would be zeros.
#[test]
fn a_literal_mrtrix_layout_with_alignment_padding_reads_value_equal() -> Result<()> {
    const OFFSET: usize = 96;
    let dir = tempdir()?;
    let header = "mrtrix image\ndim: 2 3 2\nvox: 1 1 1\nlayout: +0,+1,+2\n\
                  datatype: UInt32LE\nfile: . 96\nEND\n";
    assert!(
        header.len() < OFFSET,
        "the fixture needs padding to reach 96"
    );
    let mut bytes = header.as_bytes().to_vec();
    bytes.resize(OFFSET, 0);
    write_samples(&U32_VALUES, ByteOrder::LittleEndian, &mut bytes)?;
    let path = dir.path().join("mrtrix_layout.mif");
    std::fs::write(&path, bytes)?;

    let image = read_mif::<u32, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(image.data_slice()?, U32_VALUES);
    Ok(())
}

/// A written file declares an offset that is a multiple of four, at or after
/// the `END` line with less than one alignment unit of zero padding, and its
/// first voxel byte sits at that offset; the file reads back value-equal.
///
/// Two spacings give headers of different length, so the check covers more
/// than one alignment residue.
#[test]
fn a_written_file_places_its_voxels_at_the_declared_aligned_offset() -> Result<()> {
    let dir = tempdir()?;
    for (index, spacing) in [[1.0, 1.0, 1.0], [0.123_456_789, 12.5, 0.001]]
        .into_iter()
        .enumerate()
    {
        let image = Image::from_flat_on(
            U16_VALUES.to_vec(),
            SHAPE,
            Point::new([1.0, 2.0, 3.0]),
            Spacing::new(spacing),
            Direction::identity(),
            &SequentialBackend,
        )?;
        let path = dir.path().join(format!("aligned_{index}.mif"));
        write_mif(&path, &image, &SequentialBackend)?;

        let bytes = std::fs::read(&path)?;
        let header = header_text(&path);
        let offset: usize = header
            .lines()
            .find_map(|line| line.strip_prefix("file: . "))
            .expect("a written file names its inline data")
            .parse()?;
        let header_end = header.len() + "END\n".len();
        assert_eq!(offset % 4, 0, "offset {offset} is not 4-byte aligned");
        assert!(
            (header_end..header_end + 4).contains(&offset),
            "offset {offset} is not the first multiple of four at or after the END line \
             ending at {header_end}"
        );
        assert!(
            bytes[header_end..offset].iter().all(|&byte| byte == 0),
            "the bytes between END and the offset are zero padding"
        );
        assert_eq!(&bytes[offset..], stored_bytes(&U16_VALUES));

        let back = read_mif::<u16, _, _, _>(&path, &SequentialBackend, Exact)?;
        assert_eq!(back.data_slice()?, U16_VALUES);
    }
    Ok(())
}
