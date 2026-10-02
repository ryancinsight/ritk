use super::*;
use crate::test_support::assert_bits_eq;
use ritk_codecs::sample::{Cast, Exact, Sample};

/// A 2x3x2 MGH file of `code` holding `values` big-endian.
fn file_of<T: Sample>(code: i32, values: &[T]) -> Vec<u8> {
    let mut payload = Vec::new();
    ritk_codecs::sample::write_samples(values, consus_core::ByteOrder::BigEndian, &mut payload)
        .expect("a vector accepts every byte");
    build_mgh_bytes(
        1,
        [2, 3, 2],
        SINGLE_FRAME,
        code,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &payload,
    )
}

/// Write `values` as `code`, then read them back in their own type, bit for
/// bit.
fn reads_in_the_stored_type<T: Sample>(code: i32, values: [T; 12]) -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("stored.mgh");
    std::fs::write(&path, file_of(code, &values))?;
    let image = read_mgh::<T, _, _, _>(&path, &TestBackend::default(), Exact)?;
    assert_eq!(image.shape(), [2, 3, 2]);
    assert_bits_eq(image.data_slice()?, &values, T::TYPE.name());
    Ok(())
}

#[test]
fn every_mgh_type_reads_in_its_stored_type() -> Result<()> {
    reads_in_the_stored_type(
        MRI_UCHAR,
        [0_u8, 1, 2, 10, 20, 50, 100, 127, 128, 200, 254, 255],
    )?;
    reads_in_the_stored_type(
        MRI_SHORT,
        [
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
        ],
    )?;
    // 2^24 + 1 and the extremes have no exact f32 value.
    reads_in_the_stored_type(
        MRI_INT,
        [
            i32::MIN,
            -100_000,
            -1,
            0,
            1,
            16_777_217,
            50_000,
            75_000,
            -16_777_217,
            3,
            4,
            i32::MAX,
        ],
    )?;
    reads_in_the_stored_type(
        MRI_FLOAT,
        [
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
        ],
    )
}

/// `u8` and `i16` widen to `f32` exactly; `i32` does not, so `Exact` refuses
/// it and `Cast` rounds it.
#[test]
fn exact_reads_widen_and_refuse_by_the_stored_type() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();

    let shorts = [-1024_i16, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 3071];
    let short_path = dir.path().join("i16.mgh");
    std::fs::write(&short_path, file_of(MRI_SHORT, &shorts))?;
    let widened = read_mgh::<f32, _, _, _>(&short_path, &backend, Exact)?;
    assert_eq!(widened.data_slice()?, shorts.map(f32::from));

    let ints = [16_777_217_i32, -3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let int_path = dir.path().join("i32.mgh");
    std::fs::write(&int_path, file_of(MRI_INT, &ints))?;
    let refused = read_mgh::<f32, _, _, _>(&int_path, &backend, Exact)
        .expect_err("i32 does not widen to f32");
    assert!(
        format!("{refused:#}").contains("i32 samples do not all have exact f32 values"),
        "{refused:#}"
    );
    let cast = read_mgh::<f32, _, _, _>(&int_path, &backend, Cast)?;
    assert_eq!(cast.data_slice()?[..2], [16_777_216.0, -3.0]);
    Ok(())
}
