//! Analyze 7.5 in every stored sample type (ADR 0053): the writer stores the
//! image's own type, the reader returns it unchanged, and the `funused1` scale
//! travels as a `Rescale`.

use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_codecs::sample::{write_samples, Cast, Exact, Rescale, Sample};
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use crate::codec::{read_le, write_le, HDR_SIZE};
use crate::{
    read_analyze, read_analyze_stored, write_analyze, DT_DOUBLE, DT_FLOAT, DT_SIGNED_INT,
    DT_SIGNED_SHORT, DT_UNSIGNED_CHAR,
};

/// A 2x2x2 image of `values` on an anisotropic grid.
fn image_of<T: Sample>(values: [T; 8]) -> Result<Image<T, SequentialBackend, 3>> {
    Image::from_flat_on(
        values.to_vec(),
        [2, 2, 2],
        Point::new([3.0, -2.0, 1.5]),
        Spacing::new([1.5, 1.0, 0.5]),
        Direction::identity(),
        &SequentialBackend,
    )
}

/// Assert `actual` holds `expected` bit for bit.
///
/// Comparing the little-endian encodings distinguishes `-0.0` from `0.0` and
/// one NaN payload from another, which `==` on floats does not.
fn assert_bits_eq<T: Sample>(actual: &[T], expected: &[T], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: sample count");
    let encode = |values: &[T]| {
        let mut bytes = Vec::new();
        write_samples(values, consus_core::ByteOrder::LittleEndian, &mut bytes)
            .expect("a vector accepts every byte");
        bytes
    };
    let (actual, expected) = (encode(actual), encode(expected));
    let width = T::TYPE.byte_width();
    if let Some(byte) = actual.iter().zip(&expected).position(|(a, e)| a != e) {
        let sample = byte / width;
        panic!(
            "{context}: {} sample {sample} differs in its bits: {:02x?} != {:02x?}",
            T::TYPE,
            &actual[sample * width..][..width],
            &expected[sample * width..][..width],
        );
    }
}

/// Write `values` in `T`, check the header's `datatype` and `bitpix`, and read
/// them back in `T` bit for bit.
fn round_trips<T: Sample>(code: i16, bits: i16, values: [T; 8]) -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("typed.hdr");
    write_analyze(&path, &image_of(values)?, &SequentialBackend)?;
    let header = std::fs::read(&path)?;
    assert_eq!(read_le::<i16>(&header, 70), code, "{}", T::TYPE);
    assert_eq!(read_le::<i16>(&header, 72), bits, "{}", T::TYPE);
    let payload = std::fs::read(path.with_extension("img"))?;
    assert_eq!(payload.len(), 8 * T::TYPE.byte_width());
    let loaded = read_analyze::<T, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_bits_eq(loaded.data_slice()?, &values, T::TYPE.name());
    assert_eq!(loaded.spacing(), &Spacing::new([1.5, 1.0, 0.5]));
    Ok(())
}

#[test]
fn every_analyze_type_round_trips_in_its_stored_type() -> Result<()> {
    round_trips(DT_UNSIGNED_CHAR, 8, [0_u8, 1, 2, 127, 128, 200, 254, 255])?;
    round_trips(
        DT_SIGNED_SHORT,
        16,
        [i16::MIN, -1024, -1, 0, 1, 3071, 30_000, i16::MAX],
    )?;
    // 2^24 + 1 has no exact f32: the stored i32 must survive.
    round_trips(
        DT_SIGNED_INT,
        32,
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
    round_trips(
        DT_FLOAT,
        32,
        [
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0 / 7.0,
            f32::EPSILON,
            f32::MAX,
            f32::MIN,
            -1.5,
        ],
    )?;
    // 0.1 and 1/3 have no exact f32: the stored f64 must survive.
    round_trips(
        DT_DOUBLE,
        64,
        [
            -0.0,
            0.1,
            1.0 / 3.0,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            f64::MAX,
            f64::MIN,
            1e300,
        ],
    )
}

/// Analyze has no code for these types; the writer refuses them before
/// creating either file.
#[test]
fn types_analyze_cannot_store_are_refused() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("unsigned.hdr");
    let err = write_analyze(&path, &image_of([7_u16; 8])?, &SequentialBackend)
        .expect_err("Analyze has no uint16 code");
    assert!(
        format!("{err:#}").contains("Analyze cannot store u16 samples"),
        "{err:#}"
    );
    assert!(!path.exists());
    assert!(!path.with_extension("img").exists());
    let err = write_analyze(&path, &image_of([-1_i64; 8])?, &SequentialBackend)
        .expect_err("Analyze has no int64 code");
    assert!(format!("{err:#}").contains("i64"), "{err:#}");
    Ok(())
}

/// An `int16` file whose `funused1` is `scale`.
fn volume_with_stored_scale(scale: f32) -> Result<(tempfile::TempDir, std::path::PathBuf)> {
    let dir = tempdir()?;
    let path = dir.path().join("scaled.hdr");
    write_analyze(
        &path,
        &image_of([-4_i16, -2, 0, 1, 3, 5, 7, 1000])?,
        &SequentialBackend,
    )?;
    let mut header = std::fs::read(&path)?;
    let mut block = [0_u8; HDR_SIZE];
    block.copy_from_slice(&header);
    write_le::<f32>(&mut block, 112, scale);
    header.copy_from_slice(&block);
    std::fs::write(&path, header)?;
    Ok((dir, path))
}

#[test]
fn scaled_files_read_physical_values_and_keep_the_stored_samples() -> Result<()> {
    let (_dir, path) = volume_with_stored_scale(0.5)?;
    let physical = [-2.0, -1.0, 0.0, 0.5, 1.5, 2.5, 3.5, 500.0];
    let single = read_analyze::<f32, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(
        single.data_slice()?,
        physical.map(|value: f64| value as f32)
    );
    let double = read_analyze::<f64, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(double.data_slice()?, physical);

    let err = read_analyze::<i16, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("a scaled file has no faithful int16 physical value");
    assert!(
        format!("{err:#}").contains("read_analyze_stored"),
        "{err:#}"
    );
    let (stored, rescale) = read_analyze_stored::<i16, _, _, _>(&path, &SequentialBackend, Exact)?;
    assert_eq!(stored.data_slice()?, [-4_i16, -2, 0, 1, 3, 5, 7, 1000]);
    assert_eq!(rescale, Rescale::new(0.5, 0.0)?);
    Ok(())
}

/// A factor of 0 or 1 is the identity, so an integer read succeeds.
#[test]
fn zero_and_unit_scales_are_the_identity() -> Result<()> {
    for scale in [0.0, 1.0] {
        let (_dir, path) = volume_with_stored_scale(scale)?;
        let image = read_analyze::<i16, _, _, _>(&path, &SequentialBackend, Exact)?;
        assert_eq!(image.data_slice()?, [-4_i16, -2, 0, 1, 3, 5, 7, 1000]);
        let (_, rescale) = read_analyze_stored::<i16, _, _, _>(&path, &SequentialBackend, Exact)?;
        assert!(rescale.is_identity(), "scale {scale}");
    }
    Ok(())
}

/// `Exact` refuses an `int32` file read as `f32`; `Cast` rounds it.
#[test]
fn exact_refuses_a_narrowing_read_and_cast_rounds() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("wide.hdr");
    write_analyze(
        &path,
        &image_of([16_777_217_i32, -3, 0, 1, 2, 3, 4, 5])?,
        &SequentialBackend,
    )?;
    let err = read_analyze::<f32, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("i32 does not widen to f32");
    assert!(
        format!("{err:#}").contains("i32 samples do not all have exact f32 values"),
        "{err:#}"
    );
    let cast = read_analyze::<f32, _, _, _>(&path, &SequentialBackend, Cast)?;
    assert_eq!(cast.data_slice()?[..2], [16_777_216.0, -3.0]);
    Ok(())
}
