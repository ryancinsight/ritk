//! Files the reader must refuse, or map without overflow, end to end.

use crate::hdf5_binary::write_minc2_hdf5;
use crate::scaled_fixture::{write_fixture, ImageRangeFixture};
use crate::{read_minc, read_minc_stored};
use anyhow::Result;
use coeus_core::SequentialBackend;
use consus_core::ByteOrder;
use ritk_codecs::sample::Exact;
use ritk_spatial::Direction;
use tempfile::tempdir;

#[test]
fn a_header_claiming_an_unbacked_slice_count_fails_instead_of_allocating() -> Result<()> {
    // The writer gives an integer image scalar `image-min` / `image-max`, so
    // expanding one map per claimed slice would reserve i32::MAX 32-byte maps.
    let dir = tempdir()?;
    let path = dir.path().join("forged-slice-count.mnc");
    let slices = usize::try_from(i32::MAX)?;
    write_minc2_hdf5(
        &path,
        &[0_i16; 8],
        [slices, 1, 1],
        [0.0; 3],
        [1.0; 3],
        &Direction::identity(),
    )?;

    for error in [
        read_minc::<f32, _, _, _>(&path, &SequentialBackend, Exact).expect_err("unbacked data"),
        read_minc_stored::<i16, _, _, _>(&path, &SequentialBackend, Exact)
            .expect_err("unbacked data"),
    ] {
        assert!(
            format!("{error:#}").contains("voxel data"),
            "expected a voxel-data read error, got {error:#}"
        );
    }
    Ok(())
}

/// A `u8` file whose `valid_range` is `[0, 1]`: the MINC form of a binary
/// mask. A search of consus-hdf5 finds no construction of `Datatype::Boolean`
/// on its read path; its writer encodes one as a 1-byte unsigned integer.
fn binary_mask_file(voxels: [u8; 8]) -> Result<(tempfile::TempDir, std::path::PathBuf)> {
    let dir = tempdir()?;
    let path = dir.path().join("mask.mnc");
    write_fixture(
        &path,
        &voxels,
        [2, 2, 2],
        [0, 1],
        ByteOrder::LittleEndian,
        ImageRangeFixture::Omitted,
    )?;
    Ok((dir, path))
}

#[test]
fn a_binary_mask_reads_its_codes_as_the_default_real_range() -> Result<()> {
    let (_dir, path) = binary_mask_file([0, 1, 1, 0, 1, 0, 0, 1])?;

    let image = read_minc::<f32, _, _, _>(&path, &SequentialBackend, Exact)?;

    assert_eq!(
        image.data_slice()?,
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    );
    Ok(())
}

#[test]
fn a_binary_mask_byte_above_one_is_rejected() -> Result<()> {
    let (_dir, path) = binary_mask_file([0, 1, 1, 0, 1, 2, 0, 1])?;

    let error = read_minc::<f32, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("a code above 1 is outside valid_range [0, 1]");

    assert!(
        format!("{error:#}").contains("stored voxel 5 value 2 is outside valid_range [0, 1]"),
        "unexpected error: {error:#}"
    );
    Ok(())
}

#[test]
fn an_f32_read_refuses_a_map_whose_intermediate_overflows_where_f64_holds_the_value() -> Result<()>
{
    // Slope 6e36 and intercept -3e38 are finite in f32, and every mapped value
    // (at most 3e38) fits f32, but the map runs in f32 and stored 100 forms the
    // intermediate (100 - 0) * 6e36 = 6e38, past f32::MAX.
    let dir = tempdir()?;
    let path = dir.path().join("overflowing-map.mnc");
    write_fixture(
        &path,
        &[0_i16, 25, 50, 100],
        [1, 2, 2],
        [0, 100],
        ByteOrder::LittleEndian,
        ImageRangeFixture::Complete {
            minima: &[-3.0e38],
            maxima: &[3.0e38],
        },
    )?;

    let error = read_minc::<f32, _, _, _>(&path, &SequentialBackend, Exact)
        .expect_err("the 6e38 intermediate exceeds the finite f32 range");
    assert!(
        format!("{error:#}").contains("scaled voxel 3 leaves the finite range of f32")
            && format!("{error:#}").contains("intermediate (x - valid_min) * slope")
            && format!("{error:#}").contains("read into a wider floating-point type"),
        "unexpected error: {error:#}"
    );

    let wide = read_minc::<f64, _, _, _>(&path, &SequentialBackend, Exact)?;
    let expected = [-3.0e38, -1.5e38, 0.0, 3.0e38];
    for (got, want) in wide.data_slice()?.iter().zip(expected) {
        // At most 10 u (P + R) with u = 2^-53, P = |(v - v_min) a| and R = |r_min|
        // (the real-value map derivation in `tests_real_map`). The largest sum is
        // at stored 100: P = 100 * 6e36 = 6e38, R = 3e38, so P + R = 9e38.
        assert!(
            (got - want).abs() <= 10.0 * f64::EPSILON / 2.0 * 9.0e38,
            "{got} vs {want}"
        );
    }
    Ok(())
}
