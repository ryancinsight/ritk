//! MINC2 in every stored sample type (ADR 0053): the writer stores the image's
//! own type, the reader returns it unchanged, and the `valid_range` /
//! `image-min` / `image-max` conversion travels as one `RealValueMap` per slice.

use crate::datatype::stored_type;
use crate::RealValueMap;
use anyhow::Result;
use coeus_core::SequentialBackend;
use consus_core::{extend_encoded, ByteOrder, Datatype};
use eunomia::NumericElement;
use ritk_codecs::sample::{Cast, Exact, Sample, SampleType};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::fmt::Debug;
use tempfile::tempdir;

use crate::scaled_fixture::{write_fixture, ImageRangeFixture};
use crate::{read_minc, read_minc_stored, write_minc};

const SHAPE: [usize; 3] = [2, 2, 2];

fn image_of<T: Sample>(values: [T; 8]) -> Result<Image<T, SequentialBackend, 3>> {
    Image::from_flat_on(
        values.to_vec(),
        SHAPE,
        Point::new([3.0, -2.0, 1.5]),
        Spacing::new([1.5, 1.0, 0.5]),
        Direction::identity(),
        &SequentialBackend,
    )
}

/// The packed little-endian bytes of `values`, for bit-for-bit comparison.
fn bit_pattern<T: Sample>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::new();
    extend_encoded(&mut bytes, values.iter().copied(), ByteOrder::LittleEndian);
    bytes
}

/// Write `values` in `T`, read them back in `T` bit for bit, and check that the
/// file maps them by the identity.
fn writer_output_reads_back_unchanged<T: Sample + Debug>(values: [T; 8]) -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("typed.mnc");
    let image = image_of(values)?;
    write_minc(&image, &path, &SequentialBackend)?;

    let loaded = read_minc::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(
        bit_pattern(loaded.data_slice()?),
        bit_pattern(&values),
        "{}",
        T::TYPE
    );
    assert_eq!(loaded.shape(), SHAPE);
    assert_eq!(loaded.origin(), image.origin());
    assert_eq!(loaded.spacing(), image.spacing());
    assert_eq!(loaded.direction(), image.direction());

    let (stored, rescales) = read_minc_stored::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(stored.data_slice()?, values, "{}", T::TYPE);
    assert_eq!(rescales, [RealValueMap::IDENTITY; 2], "{}", T::TYPE);
    Ok(())
}

#[test]
fn every_type_minc_stores_round_trips_in_its_stored_type() -> Result<()> {
    writer_output_reads_back_unchanged([0_u8, 1, 2, 127, 128, 200, 254, 255])?;
    writer_output_reads_back_unchanged([i8::MIN, -100, -1, 0, 1, 2, 100, i8::MAX])?;
    writer_output_reads_back_unchanged([0_u16, 1, 255, 256, 4_095, 30_000, 65_000, u16::MAX])?;
    writer_output_reads_back_unchanged([i16::MIN, -1024, -1, 0, 1, 3071, 30_000, i16::MAX])?;
    // 2^24 + 1 has no exact f32: the stored 32-bit integers must survive.
    writer_output_reads_back_unchanged([
        0_u32,
        1,
        16_777_217,
        70_000,
        2_000_000_001,
        3_000_000_000,
        4_294_967_294,
        u32::MAX,
    ])?;
    writer_output_reads_back_unchanged([
        i32::MIN,
        -16_777_217,
        -1,
        0,
        1,
        16_777_217,
        70_000,
        i32::MAX,
    ])?;
    writer_output_reads_back_unchanged([
        -0.0_f32,
        0.0,
        f32::MIN_POSITIVE,
        1.0 / 7.0,
        f32::EPSILON,
        f32::MAX,
        f32::MIN,
        -1.5,
    ])?;
    // 0.1 and 1/3 have no exact f32: the stored f64 must survive.
    writer_output_reads_back_unchanged([
        -0.0_f64,
        0.1,
        1.0 / 3.0,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        f64::MAX,
        f64::MIN,
        1e300,
    ])
}

/// Author `values` in `order` with the identity map and read them back in `T`.
///
/// `values` ascend, so the first and last are the extremes: they are the
/// `valid_range` and, as `f64`, the `image-min` and `image-max` of an integer
/// type, whose map is then the identity.
fn foreign_file_reads_back_unchanged<T: Sample + Debug>(
    values: [T; 8],
    order: ByteOrder,
) -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("foreign.mnc");
    let (low, high) = (
        NumericElement::to_f64(values[0]),
        NumericElement::to_f64(values[7]),
    );
    let ranges = if T::TYPE.is_float() {
        ImageRangeFixture::Omitted
    } else {
        ImageRangeFixture::Complete {
            minima: &[low],
            maxima: &[high],
        }
    };
    write_fixture(&path, &values, SHAPE, [values[0], values[7]], order, ranges)?;

    let (stored, rescales) = read_minc_stored::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(
        bit_pattern(stored.data_slice()?),
        bit_pattern(&values),
        "{} {order:?}",
        T::TYPE
    );
    assert_eq!(
        rescales,
        [RealValueMap::IDENTITY; 2],
        "{} {order:?}",
        T::TYPE
    );
    let real = read_minc::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(real.data_slice()?, values, "{} {order:?}", T::TYPE);
    Ok(())
}

#[test]
fn foreign_files_read_in_their_stored_type_in_either_byte_order() -> Result<()> {
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        foreign_file_reads_back_unchanged([0_u8, 1, 2, 127, 128, 200, 254, 255], order)?;
        foreign_file_reads_back_unchanged([i8::MIN, -100, -1, 0, 1, 2, 100, i8::MAX], order)?;
        foreign_file_reads_back_unchanged(
            [0_u16, 1, 255, 256, 4_095, 30_000, 65_000, u16::MAX],
            order,
        )?;
        foreign_file_reads_back_unchanged(
            [i16::MIN, -1024, -1, 0, 1, 3071, 30_000, i16::MAX],
            order,
        )?;
        foreign_file_reads_back_unchanged(
            [
                0_u32,
                1,
                16_777_217,
                70_000,
                2_000_000_001,
                3_000_000_000,
                4_294_967_294,
                u32::MAX,
            ],
            order,
        )?;
        foreign_file_reads_back_unchanged(
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
            order,
        )?;
        // 2^53 + 1 has no exact f64: the stored 64-bit integers must survive.
        foreign_file_reads_back_unchanged(
            [
                0_u64,
                1,
                9_007_199_254_740_993,
                1 << 62,
                1 << 63,
                18_000_000_000_000_000_001,
                18_446_744_073_709_551_614,
                u64::MAX,
            ],
            order,
        )?;
        foreign_file_reads_back_unchanged(
            [
                i64::MIN,
                -9_007_199_254_740_993,
                -1,
                0,
                1,
                9_007_199_254_740_993,
                1 << 62,
                i64::MAX,
            ],
            order,
        )?;
        foreign_file_reads_back_unchanged(
            [
                f32::MIN,
                -1.5,
                -0.0,
                0.0,
                1.0 / 7.0,
                f32::EPSILON,
                1.0e30,
                f32::MAX,
            ],
            order,
        )?;
        foreign_file_reads_back_unchanged(
            [
                f64::MIN,
                -1e300,
                -0.0,
                0.1,
                1.0 / 3.0,
                f64::EPSILON,
                1e300,
                f64::MAX,
            ],
            order,
        )?;
    }
    Ok(())
}

#[test]
fn exact_refuses_a_narrowing_read_and_cast_converts() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("wide.mnc");
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
    write_minc(&image_of(values)?, &path, &SequentialBackend)?;

    let narrow = read_minc::<i16, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("i32 samples do not all fit i16");
    assert!(
        format!("{narrow:#}").contains("i32 samples do not all have exact i16 values"),
        "{narrow:#}"
    );
    let single = read_minc::<f32, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("i32 samples do not all have an exact f32");
    assert!(
        format!("{single:#}").contains("i32 samples do not all have exact f32 values"),
        "{single:#}"
    );

    let widened = read_minc::<f64, _, _, _>(&path, &SequentialBackend, Exact)?;
    let exact: Vec<f64> = values.iter().copied().map(f64::from).collect();
    assert_eq!(widened.data_slice()?, exact);

    // 2^24 + 1 rounds to even in f32: the cast keeps the nearest value.
    let cast = read_minc::<f32, _, _, _>(&path, &SequentialBackend, Cast)?;
    let rounded: Vec<f32> = values
        .iter()
        .copied()
        .map(|value| f32::from_signed_sample(i64::from(value)))
        .collect();
    assert_eq!(cast.data_slice()?, rounded);
    assert_eq!(cast.data_slice()?[5], 16_777_216.0_f32);

    let path = dir.path().join("double.mnc");
    write_minc(&image_of([0.1_f64; 8])?, &path, &SequentialBackend)?;
    let single = read_minc::<f32, _, _, _>(&path, &SequentialBackend, Cast)?;
    assert_eq!(single.data_slice()?, [0.1_f32; 8]);
    Ok(())
}

#[test]
fn types_minc_cannot_store_are_refused_before_a_file_is_created() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("wide-integer.mnc");
    let error = write_minc(&image_of([7_u64; 8])?, &path, &SequentialBackend)
        .expect_err("MINC2 has no 64-bit integer type");
    assert!(
        format!("{error:#}").contains("MINC2 cannot store u64 samples"),
        "{error:#}"
    );
    assert!(!path.exists());
    let error = write_minc(&image_of([-1_i64; 8])?, &path, &SequentialBackend)
        .expect_err("MINC2 has no 64-bit integer type");
    assert!(format!("{error:#}").contains("i64"), "{error:#}");
    assert!(!path.exists());
    Ok(())
}

/// A two-slice `i16` image whose slices map `[0, 100]` to `[-1000, 1000]` and
/// `[0, 200]`, authored in `order`.
fn two_slice_scaled_file(order: ByteOrder) -> Result<(tempfile::TempDir, std::path::PathBuf)> {
    let dir = tempdir()?;
    let path = dir.path().join("scaled.mnc");
    write_fixture(
        &path,
        &[0_i16, 25, 50, 100, 0, 25, 50, 100],
        SHAPE,
        [0, 100],
        order,
        ImageRangeFixture::Complete {
            minima: &[-1_000.0, 0.0],
            maxima: &[1_000.0, 200.0],
        },
    )?;
    Ok((dir, path))
}

fn mapped_slices_read_as_real_values<T: Sample + Debug>(order: ByteOrder) -> Result<()> {
    let (_dir, path) = two_slice_scaled_file(order)?;
    let real = read_minc::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
    let expected: Vec<T> = [-1_000.0, -500.0, 0.0, 1_000.0, 0.0, 50.0, 100.0, 200.0]
        .into_iter()
        .map(T::from_real_sample)
        .collect();
    assert_eq!(real.data_slice()?, expected, "{} {order:?}", T::TYPE);
    Ok(())
}

#[test]
fn scaled_files_read_real_values_in_every_float_type() -> Result<()> {
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        mapped_slices_read_as_real_values::<f32>(order)?;
        mapped_slices_read_as_real_values::<f64>(order)?;
    }
    Ok(())
}

#[test]
fn scaled_files_keep_the_stored_samples_and_one_map_per_slice() -> Result<()> {
    let (_dir, path) = two_slice_scaled_file(ByteOrder::LittleEndian)?;

    let (stored, rescales) = read_minc_stored::<i16, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(stored.data_slice()?, [0, 25, 50, 100, 0, 25, 50, 100]);
    assert_eq!(
        rescales,
        [
            RealValueMap::new(0.0, 20.0, -1_000.0)?,
            RealValueMap::new(0.0, 2.0, 0.0)?
        ]
    );

    let refused = read_minc::<i16, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("a real intensity has no faithful i16 value");
    let message = format!("{refused:#}");
    assert!(
        message.contains("MINC2 slice 0 maps stored values by (x - 0) * 20 + -1000"),
        "{message}"
    );
    assert!(message.contains("read_minc_stored"), "{message}");
    Ok(())
}

#[test]
fn stored_type_names_the_sample_and_byte_order_of_an_hdf5_datatype() {
    let signed_sixteen_bit = Datatype::Integer {
        bits: core::num::NonZeroUsize::new(16).expect("nonzero"),
        byte_order: ByteOrder::BigEndian,
        signed: true,
    };
    assert_eq!(
        stored_type(&signed_sixteen_bit).expect("MINC2 stores signed 16-bit voxels"),
        crate::datatype::StoredType {
            sample_type: SampleType::I16,
            byte_order: ByteOrder::BigEndian,
        }
    );
    let half_precision = Datatype::Float {
        bits: core::num::NonZeroUsize::new(16).expect("nonzero"),
        byte_order: ByteOrder::BigEndian,
    };
    let error = stored_type(&half_precision).expect_err("MINC2 has no half-precision voxels");
    assert!(error.to_string().contains("16 bits"), "{error}");
}

#[test]
fn stored_type_refuses_an_hdf5_boolean_datatype_by_name() {
    let error = stored_type(&Datatype::Boolean).expect_err("MINC2 has no Boolean voxels");
    assert_eq!(
        error.to_string(),
        "Unsupported MINC2 voxel datatype: Boolean"
    );
}
