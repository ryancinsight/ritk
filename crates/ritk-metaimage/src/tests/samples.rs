//! Typed samples: every `ElementType` reads in its stored type, in every file
//! layout and byte order, and the writer stores the type of its image.

use anyhow::Result;
use coeus_core::SequentialBackend;
use consus_core::ByteOrder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use ritk_codecs::sample::{write_samples, Cast, Exact, Sample, SampleType};
use ritk_core::rejection::assert_rejects;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Debug;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

/// Every `ElementType` of the MetaIO specification with the sample type it
/// stores; the oracle both directions of the name mapping are checked against.
const ELEMENT_TYPES: [(SampleType, &str); 10] = [
    (SampleType::I8, "MET_CHAR"),
    (SampleType::U8, "MET_UCHAR"),
    (SampleType::I16, "MET_SHORT"),
    (SampleType::U16, "MET_USHORT"),
    (SampleType::I32, "MET_INT"),
    (SampleType::U32, "MET_UINT"),
    (SampleType::I64, "MET_LONG_LONG"),
    (SampleType::U64, "MET_ULONG_LONG"),
    (SampleType::F32, "MET_FLOAT"),
    (SampleType::F64, "MET_DOUBLE"),
];

/// `DimSize = 2 3 2`, so the image shape is `[nz, ny, nx] = [2, 3, 2]`.
const DIM_SIZE: [usize; 3] = [2, 3, 2];
const SHAPE: [usize; 3] = [2, 3, 2];

/// Where the payload of a file lives and how it is encoded.
#[derive(Clone, Copy, Debug)]
struct Encoding {
    detached: bool,
    compressed: bool,
    order: ByteOrder,
}

impl Encoding {
    /// Every combination of layout, compression, and byte order.
    fn all() -> impl Iterator<Item = Self> {
        [false, true].into_iter().flat_map(|detached| {
            [false, true].into_iter().flat_map(move |compressed| {
                [ByteOrder::LittleEndian, ByteOrder::BigEndian]
                    .into_iter()
                    .map(move |order| Self {
                        detached,
                        compressed,
                        order,
                    })
            })
        })
    }
}

fn element_type_of(sample_type: SampleType) -> &'static str {
    ELEMENT_TYPES
        .iter()
        .find(|(candidate, _)| *candidate == sample_type)
        .map(|(_, name)| *name)
        .expect("every sample type has an ElementType")
}

fn packed<T: Sample>(values: &[T], order: ByteOrder) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_samples(values, order, &mut bytes).expect("a vector accepts every byte");
    bytes
}

fn deflated(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(bytes)
        .expect("a vector accepts every byte");
    encoder.finish().expect("a vector accepts every byte")
}

/// A MetaImage of `element_type` whose payload is `payload` laid out as
/// `encoding` says; returns the header path to read.
fn file_with_payload(
    dir: &Path,
    element_type: &str,
    payload: &[u8],
    encoding: Encoding,
) -> Result<PathBuf> {
    let payload = if encoding.compressed {
        deflated(payload)
    } else {
        payload.to_vec()
    };
    let path = dir.join(if encoding.detached {
        "volume.mhd"
    } else {
        "volume.mha"
    });
    let msb = encoding.order == ByteOrder::BigEndian;
    let mut header = format!(
        "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = {msb}\n\
         CompressedData = {}\nTransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = 0 0 0\n\
         ElementSpacing = 1 1 1\nDimSize = {} {} {}\nElementType = {element_type}\n",
        encoding.compressed, DIM_SIZE[0], DIM_SIZE[1], DIM_SIZE[2],
    )
    .into_bytes();
    if encoding.detached {
        header.extend_from_slice(b"ElementDataFile = volume.raw\n");
        std::fs::write(dir.join("volume.raw"), &payload)?;
    } else {
        header.extend_from_slice(b"ElementDataFile = LOCAL\n");
        header.extend_from_slice(&payload);
    }
    std::fs::write(&path, header)?;
    Ok(path)
}

fn file_of<T: Sample>(dir: &Path, values: &[T], encoding: Encoding) -> Result<PathBuf> {
    file_with_payload(
        dir,
        element_type_of(T::TYPE),
        &packed(values, encoding.order),
        encoding,
    )
}

/// Read `values` back in their own type from every layout and byte order, bit
/// for bit.
fn reads_in_the_stored_type<T: Sample + Debug>(values: [T; 12]) -> Result<()> {
    for encoding in Encoding::all() {
        let dir = tempdir()?;
        let path = file_of(dir.path(), &values, encoding)?;
        let image = crate::read_metaimage::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
        assert_eq!(image.shape(), SHAPE, "{} {encoding:?}", T::TYPE);
        assert_eq!(
            packed(image.data_slice()?, ByteOrder::LittleEndian),
            packed(&values, ByteOrder::LittleEndian),
            "{} {encoding:?}",
            T::TYPE
        );
    }
    Ok(())
}

#[test]
fn element_type_names_map_both_ways() {
    for (sample_type, name) in ELEMENT_TYPES {
        assert_eq!(crate::element_type::element_type_name(sample_type), name);
        assert_eq!(
            crate::element_type::sample_type_from_element_type(name)
                .expect("a MetaIO name maps to a sample type"),
            sample_type,
            "{name}"
        );
    }
}

/// Write `values` through the writer and read them back in the same type, bit
/// for bit, with the header naming the type and the geometry intact.
fn writer_stores_the_image_type<T: Sample + Debug>(values: [T; 12]) -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("written.mha");
    let backend = SequentialBackend;
    let origin = Point::new([10.0, 20.0, 30.0]);
    let spacing = Spacing::new([0.9, 0.8, 1.5]);
    let image = Image::<T, _, 3>::from_flat_on(
        values.to_vec(),
        SHAPE,
        origin,
        spacing,
        Direction::identity(),
        &backend,
    )?;

    crate::write_metaimage(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;
    let header = String::from_utf8_lossy(&bytes);
    assert!(
        header.contains(&format!("ElementType = {}\n", element_type_of(T::TYPE))),
        "{}: {}",
        T::TYPE,
        &header[..header.len().min(400)]
    );
    assert!(header.contains("BinaryDataByteOrderMSB = False\n"));
    assert!(
        bytes.ends_with(&packed(&values, ByteOrder::LittleEndian)),
        "{}: the payload is the packed little-endian samples",
        T::TYPE
    );

    let loaded = crate::read_metaimage::<T, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(loaded.shape(), SHAPE);
    assert_eq!(*loaded.origin(), origin);
    assert_eq!(*loaded.spacing(), spacing);
    assert_eq!(
        packed(loaded.data_slice()?, ByteOrder::LittleEndian),
        packed(&values, ByteOrder::LittleEndian),
        "{}",
        T::TYPE
    );
    Ok(())
}

/// A property checked against twelve samples of one stored type.
trait SampleProperty {
    fn check<T: Sample + Debug>(&self, values: [T; 12]) -> Result<()>;
}

/// Run `check` over twelve samples of each of the ten stored types, chosen to
/// include values the narrower float types cannot hold.
fn for_every_sample_type(check: &impl SampleProperty) -> Result<()> {
    check.check([0_u8, 1, 2, 10, 20, 50, 100, 127, 128, 200, 254, u8::MAX])?;
    check.check([i8::MIN, -100, -10, -1, 0, 1, 2, 10, 50, 100, 126, i8::MAX])?;
    check.check([
        0_u16,
        1,
        255,
        256,
        300,
        1000,
        4095,
        32767,
        32768,
        40000,
        65534,
        u16::MAX,
    ])?;
    check.check([
        i16::MIN,
        -1000,
        -100,
        -1,
        0,
        1,
        100,
        300,
        500,
        750,
        3071,
        i16::MAX,
    ])?;
    // 2^24 + 1 is the first integer binary32 cannot hold.
    check.check([
        0_u32,
        1,
        255,
        65_536,
        16_777_216,
        16_777_217,
        100_000,
        1 << 31,
        3,
        4,
        5,
        u32::MAX,
    ])?;
    check.check([
        i32::MIN,
        -100_000,
        -16_777_217,
        -1,
        0,
        1,
        3,
        4,
        50_000,
        75_000,
        16_777_217,
        i32::MAX,
    ])?;
    // 2^53 + 1 is the first integer binary64 cannot hold.
    check.check([
        0_u64,
        1,
        255,
        1 << 32,
        (1 << 53) - 1,
        (1 << 53) + 1,
        u64::MAX - 1,
        3,
        4,
        5,
        6,
        u64::MAX,
    ])?;
    check.check([
        i64::MIN,
        -(1 << 53) - 1,
        -1,
        0,
        1,
        2,
        3,
        4,
        1 << 32,
        (1 << 53) + 1,
        i64::MAX - 1,
        i64::MAX,
    ])?;
    check.check([
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
    ])?;
    // 0.1 has no exact binary32 value.
    check.check([
        0.1_f64,
        std::f64::consts::PI,
        -0.0,
        f64::MIN_POSITIVE,
        1.0 / 7.0,
        -std::f64::consts::E,
        f64::MAX,
        f64::MIN,
        1.0 / 3.0,
        f64::EPSILON,
        1e-300,
        -1.0,
    ])
}

struct ReadsInStoredType;

impl SampleProperty for ReadsInStoredType {
    fn check<T: Sample + Debug>(&self, values: [T; 12]) -> Result<()> {
        reads_in_the_stored_type(values)
    }
}

struct WriterStoresImageType;

impl SampleProperty for WriterStoresImageType {
    fn check<T: Sample + Debug>(&self, values: [T; 12]) -> Result<()> {
        writer_stores_the_image_type(values)
    }
}

#[test]
fn every_element_type_reads_in_its_stored_type_from_every_layout() -> Result<()> {
    for_every_sample_type(&ReadsInStoredType)
}

/// MetaIO stores `MET_LONG` and `MET_ULONG` in four bytes
/// (`MET_ValueTypeSize` in its `src/metaTypes.h`), so they read as `i32` and
/// `u32` in every layout.
#[test]
fn met_long_names_read_at_their_metaio_width() -> Result<()> {
    let signed = [
        0_i32,
        1,
        -1,
        i32::MIN,
        i32::MAX,
        16_777_217,
        -16_777,
        7,
        -7,
        1 << 30,
        42,
        9,
    ];
    let unsigned = [
        0_u32,
        1,
        u32::MAX,
        3 << 30,
        16_777_217,
        2,
        3,
        1 << 31,
        7,
        8,
        9,
        10,
    ];
    for encoding in Encoding::all() {
        let dir = tempdir()?;
        let payload = packed(&signed, encoding.order);
        let path = file_with_payload(dir.path(), "MET_LONG", &payload, encoding)?;
        let image = crate::read_metaimage::<i32, _, _, _>(&path, &SequentialBackend, Exact)?;
        assert_eq!(image.data_slice()?, &signed[..], "MET_LONG {encoding:?}");
        let dir = tempdir()?;
        let payload = packed(&unsigned, encoding.order);
        let path = file_with_payload(dir.path(), "MET_ULONG", &payload, encoding)?;
        let image = crate::read_metaimage::<u32, _, _, _>(&path, &SequentialBackend, Exact)?;
        assert_eq!(image.data_slice()?, &unsigned[..], "MET_ULONG {encoding:?}");
    }
    Ok(())
}

#[test]
fn writer_round_trips_every_sample_type_bit_for_bit() -> Result<()> {
    for_every_sample_type(&WriterStoresImageType)
}

/// `i16` widens to `f32` exactly; `i32` does not, so `Exact` refuses it and
/// `Cast` rounds it, in either byte order.
#[test]
fn exact_widens_and_refuses_by_the_stored_type_and_cast_rounds() -> Result<()> {
    let backend = SequentialBackend;
    for encoding in Encoding::all() {
        let dir = tempdir()?;

        let shorts = [-1024_i16, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 3071];
        let path = file_of(dir.path(), &shorts, encoding)?;
        let widened = crate::read_metaimage::<f32, _, _, _>(&path, &backend, Exact)?;
        assert_eq!(widened.data_slice()?, shorts.map(f32::from), "{encoding:?}");

        let ints = [16_777_217_i32, -3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let path = file_of(dir.path(), &ints, encoding)?;
        let refused = crate::read_metaimage::<f32, _, _, _>(&path, &backend, Exact);
        assert_rejects(refused, "i32 samples do not all have exact f32 values");
        let cast = crate::read_metaimage::<f32, _, _, _>(&path, &backend, Cast)?;
        assert_eq!(
            cast.data_slice()?[..2],
            [16_777_216.0, -3.0],
            "{encoding:?}"
        );

        let wide = crate::read_metaimage::<i64, _, _, _>(&path, &backend, Exact)?;
        assert_eq!(wide.data_slice()?, ints.map(i64::from), "{encoding:?}");
    }
    Ok(())
}

/// A 64-bit integer survives an `i64` read untouched where a detour through a
/// float would round it, and `Exact` refuses the float read.
#[test]
fn wide_integers_never_detour_through_a_float() -> Result<()> {
    let backend = SequentialBackend;
    let values = [(1_i64 << 53) + 1; 12];
    let dir = tempdir()?;
    let path = file_of(
        dir.path(),
        &values,
        Encoding {
            detached: false,
            compressed: true,
            order: ByteOrder::BigEndian,
        },
    )?;
    let exact = crate::read_metaimage::<i64, _, _, _>(&path, &backend, Exact)?;
    assert_eq!(exact.data_slice()?, values);
    let refused = crate::read_metaimage::<f64, _, _, _>(&path, &backend, Exact);
    assert_rejects(refused, "i64 samples do not all have exact f64 values");
    Ok(())
}

/// A payload shorter or longer than `DimSize` declares is rejected in every
/// layout, naming the expected byte count.
#[test]
fn payload_length_must_match_dim_size_in_every_layout() -> Result<()> {
    let backend = SequentialBackend;
    let values = [1_i16; 12];
    for encoding in Encoding::all() {
        let dir = tempdir()?;
        let exact = packed(&values, encoding.order);

        let short =
            file_with_payload(dir.path(), "MET_SHORT", &exact[..exact.len() - 2], encoding)?;
        let result = crate::read_metaimage::<i16, _, _, _>(&short, &backend, Exact);
        assert_rejects(result, "expected 24 bytes from DimSize");

        let mut longer = exact.clone();
        longer.extend_from_slice(&[0, 0]);
        let long = file_with_payload(dir.path(), "MET_SHORT", &longer, encoding)?;
        let result = crate::read_metaimage::<i16, _, _, _>(&long, &backend, Exact);
        assert_rejects(result, "the payload continues past that length");
    }
    Ok(())
}
