use super::*;
use crate::test_support::assert_bits_eq;
use ritk_codecs::sample::Sample;
use ritk_image::Image;

/// A 2x2x2 image of `values` on the unit grid.
fn image_of<T: Sample>(values: [T; 8]) -> Result<Image<T, TestBackend, 3>> {
    Image::from_flat_on(
        values.to_vec(),
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
        &TestBackend::default(),
    )
}

/// The big-endian `type` field at byte 20 of an MGH header.
fn type_code(bytes: &[u8]) -> i32 {
    i32::from_be_bytes(
        bytes[20..24]
            .try_into()
            .expect("invariant: four header bytes"),
    )
}

/// Write `values` in `T`, check the header names `code`, and read them back in
/// `T` bit for bit, through MGH and gzip-wrapped MGZ.
fn writes_the_type_code<T: Sample>(code: i32, values: [T; 8]) -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let image = image_of(values)?;
    for name in ["typed.mgh", "typed.mgz"] {
        let path = dir.path().join(name);
        write_mgh(&image, &path, &backend)?;
        if name.ends_with(".mgh") {
            let bytes = std::fs::read(&path)?;
            assert_eq!(type_code(&bytes), code, "{}", T::TYPE);
            assert_eq!(bytes.len(), HEADER_SIZE + 8 * T::TYPE.byte_width());
        }
        let loaded = crate::read_mgh::<T, _, TestBackend, _>(&path, &backend, Exact)?;
        assert_bits_eq(
            loaded.data_slice()?,
            &values,
            &format!("{} {name}", T::TYPE),
        );
    }
    Ok(())
}

#[test]
fn each_mgh_sample_type_writes_its_code() -> Result<()> {
    writes_the_type_code(MRI_UCHAR, [0_u8, 1, 2, 127, 128, 200, 254, 255])?;
    writes_the_type_code(
        MRI_SHORT,
        [i16::MIN, -1024, -1, 0, 1, 3071, 30_000, i16::MAX],
    )?;
    writes_the_type_code(
        MRI_INT,
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
    writes_the_type_code(
        MRI_FLOAT,
        [
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0 / 7.0,
            -123_456.79,
            f32::MAX,
            f32::MIN,
            f32::EPSILON,
        ],
    )
}

#[test]
fn a_series_writes_its_sample_type_code() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("series.mgh");
    let frames = [image_of([1_i16; 8])?, image_of([-2_i16; 8])?];
    crate::write_mgh_series(&path, &frames, &backend)?;
    let bytes = std::fs::read(&path)?;
    assert_eq!(type_code(&bytes), MRI_SHORT);
    let loaded = crate::read_mgh_series::<i16, _, TestBackend, _>(&path, &backend, Exact)?;
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].data_slice()?, [1_i16; 8]);
    assert_eq!(loaded[1].data_slice()?, [-2_i16; 8]);
    Ok(())
}

/// MGH has no code for these types; the writer refuses them before creating
/// a file.
#[test]
fn types_mgh_cannot_store_are_refused() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let unsigned = dir.path().join("u16.mgh");
    let err =
        write_mgh(&image_of([7_u16; 8])?, &unsigned, &backend).expect_err("MGH has no uint16 code");
    assert!(
        format!("{err:#}").contains("MGH cannot store u16 samples"),
        "{err:#}"
    );
    assert!(!unsigned.exists());
    let double = dir.path().join("f64.mgh");
    let err = crate::write_mgh_series(&double, &[image_of([0.5_f64; 8])?], &backend)
        .expect_err("MGH has no float64 code");
    assert!(format!("{err:#}").contains("f64"), "{err:#}");
    Ok(())
}
