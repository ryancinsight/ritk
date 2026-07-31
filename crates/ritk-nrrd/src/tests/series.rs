//! Acquisition-series coverage for the NRRD codec.
//!
//! NRRD, unlike NIfTI, does not fix where the non-spatial axis sits. The cases
//! below pin both layouts a diffusion file arrives in — a leading (fastest)
//! gradient axis in the NA-MIC convention Slicer and DTIPrep emit, and a
//! trailing (slowest) one — against a hand-written header whose bytes are laid
//! out by the test, so a stride error cannot pass by agreeing with itself.

use crate::{read_nrrd, read_nrrd_series, write_nrrd, write_nrrd_series};
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use std::io::Write;
use tempfile::tempdir;

type TestBackend = SequentialBackend;

const SPACE_DIRECTIONS: &str = "(0.75,0,0) (0,1.5,0) (0,0,2)";

fn make_image(values: Vec<f32>, dims: [usize; 3]) -> Image<f32, TestBackend, 3> {
    Image::from_flat_on(
        values,
        dims,
        Point::new([-11.0, 7.5, 3.25]),
        Spacing::new([2.0, 1.5, 0.75]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("valid image")
}

/// Build `volumes` images on one grid, volume `v` filled with `v * 100 + i`.
fn series_fixture(volumes: usize, dims: [usize; 3]) -> Vec<Image<f32, TestBackend, 3>> {
    let voxels = dims[0] * dims[1] * dims[2];
    (0..volumes)
        .map(|volume| {
            make_image(
                (0..voxels).map(|i| (volume * 100 + i) as f32).collect(),
                dims,
            )
        })
        .collect()
}

fn voxels_of(image: &Image<f32, TestBackend, 3>) -> Vec<f32> {
    image.data_slice().expect("contiguous host voxels").to_vec()
}

/// Write a 4-D NRRD by hand with the acquisition axis in the requested slot.
///
/// `interleaved` selects the NA-MIC layout (acquisition axis first, varying
/// fastest); otherwise the axis trails and volumes are contiguous.
fn write_manual_series(
    path: &std::path::Path,
    volumes: &[Vec<f32>],
    spatial_sizes: [usize; 3],
    interleaved: bool,
) -> Result<()> {
    let [nx, ny, nz] = spatial_sizes;
    let count = volumes.len();
    let mut file = std::fs::File::create(path)?;

    writeln!(file, "NRRD0004")?;
    writeln!(file, "type: float")?;
    writeln!(file, "dimension: 4")?;
    writeln!(file, "space: left-posterior-superior")?;
    if interleaved {
        writeln!(file, "sizes: {count} {nx} {ny} {nz}")?;
        writeln!(file, "space directions: none {SPACE_DIRECTIONS}")?;
        writeln!(file, "kinds: list domain domain domain")?;
    } else {
        writeln!(file, "sizes: {nx} {ny} {nz} {count}")?;
        writeln!(file, "space directions: {SPACE_DIRECTIONS} none")?;
        writeln!(file, "kinds: domain domain domain list")?;
    }
    writeln!(file, "endian: little")?;
    writeln!(file, "encoding: raw")?;
    writeln!(file, "space origin: (-11,7.5,3.25)")?;
    writeln!(file)?;

    let voxels = nx * ny * nz;
    let mut payload = Vec::with_capacity(count * voxels);
    if interleaved {
        for voxel in 0..voxels {
            for volume in volumes {
                payload.push(volume[voxel]);
            }
        }
    } else {
        for volume in volumes {
            payload.extend_from_slice(volume);
        }
    }
    for value in payload {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[test]
fn series_round_trips_through_the_writer() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("series.nrrd");
    let backend = TestBackend::default();
    let expected = series_fixture(5, [2, 3, 4]);

    write_nrrd_series(&path, &expected, &backend)?;
    let actual = read_nrrd_series::<TestBackend, _>(&path, &backend)?;

    assert_eq!(actual.len(), expected.len(), "volume count must round-trip");
    for (position, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(got.shape(), want.shape(), "volume {position} shape");
        assert_eq!(voxels_of(got), voxels_of(want), "volume {position} voxels");
        assert_eq!(got.origin(), want.origin(), "volume {position} origin");
        assert_eq!(got.spacing(), want.spacing(), "volume {position} spacing");
    }
    Ok(())
}

#[test]
fn leading_acquisition_axis_deinterleaves() -> Result<()> {
    // The NA-MIC DWI layout: volume v, voxel i lives at v + count * i.
    let dir = tempdir()?;
    let path = dir.path().join("interleaved.nrrd");
    let backend = TestBackend::default();
    let volumes = vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        vec![200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0],
    ];

    write_manual_series(&path, &volumes, [2, 2, 2], true)?;
    let series = read_nrrd_series::<TestBackend, _>(&path, &backend)?;

    assert_eq!(series.len(), 3);
    for (position, expected) in volumes.iter().enumerate() {
        assert_eq!(
            voxels_of(&series[position]),
            *expected,
            "volume {position} must be de-interleaved from the fastest axis"
        );
    }
    Ok(())
}

#[test]
fn trailing_acquisition_axis_reads_contiguous_volumes() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("blocked.nrrd");
    let backend = TestBackend::default();
    let volumes = vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
    ];

    write_manual_series(&path, &volumes, [2, 2, 2], false)?;
    let series = read_nrrd_series::<TestBackend, _>(&path, &backend)?;

    assert_eq!(series.len(), 2);
    for (position, expected) in volumes.iter().enumerate() {
        assert_eq!(voxels_of(&series[position]), *expected, "volume {position}");
    }
    Ok(())
}

#[test]
fn both_layouts_decode_to_the_same_series() -> Result<()> {
    // The layouts differ only in stride, so the decoded result must not depend
    // on which one the file used.
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let volumes = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        vec![17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
    ];

    let fast = dir.path().join("fast.nrrd");
    let slow = dir.path().join("slow.nrrd");
    write_manual_series(&fast, &volumes, [2, 2, 2], true)?;
    write_manual_series(&slow, &volumes, [2, 2, 2], false)?;

    let from_fast = read_nrrd_series::<TestBackend, _>(&fast, &backend)?;
    let from_slow = read_nrrd_series::<TestBackend, _>(&slow, &backend)?;

    assert_eq!(from_fast.len(), from_slow.len());
    for position in 0..from_fast.len() {
        assert_eq!(
            voxels_of(&from_fast[position]),
            voxels_of(&from_slow[position]),
            "layout must not change volume {position}"
        );
    }
    Ok(())
}

#[test]
fn series_preserves_the_shared_spatial_grid() -> Result<()> {
    // The acquisition axis contributes a `none` direction slot, so dropping it
    // incorrectly would shift spacing onto the wrong axes.
    let dir = tempdir()?;
    let path = dir.path().join("grid.nrrd");
    let backend = TestBackend::default();
    let volumes = vec![vec![0.0; 8], vec![1.0; 8]];

    write_manual_series(&path, &volumes, [2, 2, 2], true)?;
    let series = read_nrrd_series::<TestBackend, _>(&path, &backend)?;

    let reference = read_nrrd_series::<TestBackend, _>(&path, &backend)?;
    assert_eq!(series[0].spacing(), reference[0].spacing());
    // space directions (0.75,0,0) (0,1.5,0) (0,0,2) in file [x,y,z] order maps
    // to RITK [depth,row,col] = [z,y,x] spacing.
    assert_eq!(series[0].spacing()[0], 2.0, "z spacing");
    assert_eq!(series[0].spacing()[1], 1.5, "y spacing");
    assert_eq!(series[0].spacing()[2], 0.75, "x spacing");
    assert_eq!(
        series[0].spacing(),
        series[1].spacing(),
        "every volume shares one grid"
    );
    Ok(())
}

#[test]
fn single_volume_series_writes_a_rank_three_file() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("one.nrrd");
    let backend = TestBackend::default();
    let expected = series_fixture(1, [2, 2, 2]);

    write_nrrd_series(&path, &expected, &backend)?;

    let text = std::fs::read(&path)?;
    let header = String::from_utf8_lossy(&text[..text.len().min(400)]).to_string();
    assert!(
        header.contains("dimension: 3"),
        "a one-volume series is a rank-3 file, got header: {header}"
    );

    let single = read_nrrd::<TestBackend, _>(&path, &backend)?;
    assert_eq!(voxels_of(&single), voxels_of(&expected[0]));
    Ok(())
}

#[test]
fn rank_three_file_reads_as_a_one_volume_series() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("volume.nrrd");
    let backend = TestBackend::default();
    let image = series_fixture(1, [2, 2, 2]).remove(0);

    write_nrrd(&path, &image, &backend)?;
    let series = read_nrrd_series::<TestBackend, _>(&path, &backend)?;

    assert_eq!(series.len(), 1);
    assert_eq!(voxels_of(&series[0]), voxels_of(&image));
    Ok(())
}

#[test]
fn single_volume_reader_rejects_a_series() -> Result<()> {
    // Volume 0 is decodable alone, so the reader could return it and report
    // success. That is the failure this rejection exists to prevent.
    let dir = tempdir()?;
    let path = dir.path().join("reject.nrrd");
    let backend = TestBackend::default();
    write_nrrd_series(&path, &series_fixture(4, [2, 2, 2]), &backend)?;

    let err = read_nrrd::<TestBackend, _>(&path, &backend)
        .expect_err("a 4-volume series has no single-volume representation");
    let message = format!("{err:#}");
    assert!(
        message.contains("4 volumes"),
        "error must name the volume count, got: {message}"
    );
    Ok(())
}

#[test]
fn four_dimensional_file_without_an_acquisition_axis_is_rejected() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("all_spatial.nrrd");
    let backend = TestBackend::default();
    let mut file = std::fs::File::create(&path)?;
    writeln!(file, "NRRD0004")?;
    writeln!(file, "type: float")?;
    writeln!(file, "dimension: 4")?;
    writeln!(file, "sizes: 2 2 2 2")?;
    writeln!(file, "kinds: domain domain domain domain")?;
    writeln!(file, "encoding: raw")?;
    writeln!(file)?;
    file.write_all(&[0u8; 64])?;
    drop(file);

    let err = read_nrrd_series::<TestBackend, _>(&path, &backend)
        .expect_err("four spatial axes cannot reduce to a 3-D grid");
    assert!(format!("{err:#}").contains("non-spatial acquisition axis"));
    Ok(())
}

#[test]
fn writer_rejects_an_empty_series() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("empty.nrrd");
    let backend = TestBackend::default();
    let empty: Vec<Image<f32, TestBackend, 3>> = Vec::new();

    let err = write_nrrd_series(&path, &empty, &backend)
        .expect_err("a series with no volumes has no header to write");
    assert!(
        format!("{err:#}").contains("at least one volume"),
        "error must name the empty-series contract"
    );
}

#[test]
fn writer_rejects_volumes_on_different_grids() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mismatch.nrrd");
    let backend = TestBackend::default();

    let mut volumes = series_fixture(1, [2, 2, 2]);
    volumes.push(make_image(vec![0.0; 2 * 2 * 3], [2, 2, 3]));

    let err = write_nrrd_series(&path, &volumes, &backend)
        .expect_err("volumes on different grids cannot share one space directions field");
    let message = format!("{err:#}");
    assert!(
        message.contains("volume 1") && message.contains("shape"),
        "error must name the offending volume and field, got: {message}"
    );
}

#[test]
fn truncated_series_payload_is_rejected() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("truncated.nrrd");
    let backend = TestBackend::default();
    write_nrrd_series(&path, &series_fixture(4, [2, 2, 2]), &backend)?;

    let full = std::fs::read(&path)?;
    std::fs::write(&path, &full[..full.len() - 8 * std::mem::size_of::<f32>()])?;

    let err = read_nrrd_series::<TestBackend, _>(&path, &backend)
        .expect_err("a truncated series payload must fail");
    let message = format!("{err:#}");
    // The declared byte count spans every volume, so the shortfall is reported
    // against the whole series (4 volumes x 8 voxels x 4 bytes = 128), not
    // against one volume's worth.
    assert!(
        message.contains("need 128 bytes"),
        "error must name the full series byte requirement, got: {message}"
    );
    Ok(())
}
