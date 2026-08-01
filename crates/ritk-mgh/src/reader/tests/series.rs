//! Acquisition-series coverage: multi-frame round trips, the frame count the
//! writer emits, and the rejections that keep a series from silently decoding
//! as its first frame.

use super::*;

/// Build `volumes` images on one grid, volume `v` filled with `v * 100 + i`.
fn series_fixture(volumes: usize, dims: [usize; 3]) -> Vec<Image<f32, TestBackend, 3>> {
    let voxels = dims[0] * dims[1] * dims[2];
    (0..volumes)
        .map(|volume| {
            let values: Vec<f32> = (0..voxels)
                .map(|index| (volume * 100 + index) as f32)
                .collect();
            make_image(values, dims[0], dims[1], dims[2])
        })
        .collect()
}

fn assert_series_matches(
    actual: &[Image<f32, TestBackend, 3>],
    expected: &[Image<f32, TestBackend, 3>],
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "series frame count must round-trip"
    );
    for (position, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            got.shape(),
            want.shape(),
            "volume {position} shape must round-trip"
        );
        assert_eq!(
            got.data_slice().expect("contiguous host voxels"),
            want.data_slice().expect("contiguous host voxels"),
            "volume {position} voxels must round-trip"
        );
        assert_eq!(
            got.origin(),
            want.origin(),
            "volume {position} origin must round-trip"
        );
        assert_eq!(
            got.spacing(),
            want.spacing(),
            "volume {position} spacing must round-trip"
        );
    }
}

#[test]
fn series_round_trips_through_mgh() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("series.mgh");
    let backend = TestBackend::default();
    let expected = series_fixture(5, [2, 3, 4]);

    write_mgh_series(&path, &expected, &backend)?;
    let actual = read_mgh_series::<TestBackend, _>(&path, &backend)?;

    assert_series_matches(&actual, &expected);
    Ok(())
}

#[test]
fn series_round_trips_through_mgz() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("series.mgz");
    let backend = TestBackend::default();
    let expected = series_fixture(4, [2, 2, 3]);

    write_mgh_series(&path, &expected, &backend)?;
    let actual = read_mgh_series::<TestBackend, _>(&path, &backend)?;

    assert_series_matches(&actual, &expected);
    Ok(())
}

#[test]
fn single_frame_series_writes_nframes_one() -> Result<()> {
    // The canonical on-disk form for one volume is nframes = 1, so a
    // one-element series must stay readable by the ordinary single-volume
    // reader.
    let dir = tempdir()?;
    let path = dir.path().join("one.mgh");
    let backend = TestBackend::default();
    let expected = series_fixture(1, [2, 2, 2]);

    write_mgh_series(&path, &expected, &backend)?;

    let single = read_mgh::<TestBackend, _>(&path, &backend)?;
    assert_eq!(
        single.data_slice().expect("contiguous host voxels"),
        expected[0].data_slice().expect("contiguous host voxels"),
        "a one-frame series reads back through the single-frame reader"
    );
    Ok(())
}

#[test]
fn single_frame_file_reads_as_a_one_volume_series() -> Result<()> {
    // The series reader is the general entry point: an ordinary volume is a
    // series of one, so no caller needs to branch on nframes.
    let dir = tempdir()?;
    let path = dir.path().join("volume.mgh");
    let backend = TestBackend::default();
    let image = series_fixture(1, [2, 2, 2]).remove(0);

    write_mgh(&image, &path, &backend)?;
    let series = read_mgh_series::<TestBackend, _>(&path, &backend)?;

    assert_eq!(series.len(), 1);
    assert_eq!(
        series[0].data_slice().expect("contiguous host voxels"),
        image.data_slice().expect("contiguous host voxels")
    );
    Ok(())
}

#[test]
fn multi_frame_series_round_trips_voxel_order() -> Result<()> {
    // Each frame's voxels must be in the same ZYX order as the input, not
    // interleaved across frames.
    let dir = tempdir()?;
    let path = dir.path().join("order.mgh");
    let backend = TestBackend::default();
    let expected = series_fixture(3, [2, 2, 2]);

    write_mgh_series(&path, &expected, &backend)?;
    let actual = read_mgh_series::<TestBackend, _>(&path, &backend)?;

    assert_series_matches(&actual, &expected);
    Ok(())
}

#[test]
fn single_volume_reader_rejects_multi_frame_rather_than_returning_frame_zero() -> Result<()> {
    // Frame 0 is decodable on its own, so the reader could return it and report
    // success. That is the failure this rejection exists to prevent.
    let dir = tempdir()?;
    let path = dir.path().join("reject.mgh");
    let backend = TestBackend::default();
    write_mgh_series(&path, &series_fixture(6, [2, 2, 2]), &backend)?;

    let err = read_mgh::<TestBackend, _>(&path, &backend)
        .expect_err("a 6-frame series has no single-volume representation");
    let message = format!("{err:#}");

    assert!(
        message.contains("6 frames"),
        "error must name the declared frame count, got: {message}"
    );
    Ok(())
}

#[test]
fn writer_rejects_an_empty_series() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("empty.mgh");
    let backend = TestBackend::default();
    let empty: Vec<Image<f32, TestBackend, 3>> = Vec::new();

    let err = write_mgh_series(&path, &empty, &backend)
        .expect_err("a series with no volumes has no header to write");
    assert!(
        format!("{err:#}").contains("at least one volume"),
        "error must name the empty-series contract"
    );
}

#[test]
fn writer_rejects_volumes_on_different_grids() {
    // An MGH series carries one nframes with one geometry. Writing mismatched
    // volumes would emit a file whose header is correct for only some of its
    // content.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mismatch.mgh");
    let backend = TestBackend::default();

    let mut volumes = series_fixture(1, [2, 2, 2]);
    volumes.push(make_image(vec![0.0; 2 * 2 * 3], 2, 2, 3));

    let err = write_mgh_series(&path, &volumes, &backend)
        .expect_err("volumes on different grids cannot share one MGH header");
    let message = format!("{err:#}");
    assert!(
        message.contains("volume 1") && message.contains("shape"),
        "error must name the offending volume and field, got: {message}"
    );
}

#[test]
fn truncated_series_payload_is_rejected() -> Result<()> {
    // The declared byte range spans every frame, so a file cut short after the
    // first frame must fail rather than decode the frames that are present.
    let dir = tempdir()?;
    let path = dir.path().join("truncated.mgh");
    let backend = TestBackend::default();
    write_mgh_series(&path, &series_fixture(4, [2, 2, 2]), &backend)?;

    let full = std::fs::read(&path)?;
    let one_frame_end = full.len() - 3 * 8 * std::mem::size_of::<f32>();
    std::fs::write(&path, &full[..one_frame_end])?;

    let err = read_mgh_series::<TestBackend, _>(&path, &backend)
        .expect_err("a truncated series payload must fail");
    assert!(
        format!("{err:#}").contains("truncated"),
        "error must name the truncation"
    );
    Ok(())
}

#[test]
fn multi_frame_supports_all_voxel_types() -> Result<()> {
    // MGH supports four data types; multi-frame must decode each correctly.
    for (mri_type, data_bytes, expected) in [
        (
            MRI_UCHAR,
            [0u8, 1, 2, 3, 4, 5, 6, 7].to_vec(),
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ),
        (
            MRI_SHORT,
            // Two frames of 2×1×1 = 2 voxels each, signed 16-bit BE.
            // Frame 0: 10, -20.  Frame 1: 32767, -32768.
            vec![0x00u8, 0x0A, 0xFF, 0xEC, 0x7F, 0xFF, 0x80, 0x00],
            vec![10.0f32, -20.0, 32767.0, -32768.0],
        ),
        (
            MRI_INT,
            // Two frames of 1×1×1 = 1 voxel each, signed 32-bit BE.
            // Frame 0: 42.  Frame 1: -1.
            vec![0x00u8, 0x00, 0x00, 0x2A, 0xFF, 0xFF, 0xFF, 0xFF],
            vec![42.0f32, -1.0],
        ),
        (
            MRI_FLOAT,
            // Two frames of 1×1×1 = 1 voxel each, f32 BE.
            // Frame 0: 1.0.  Frame 1: -2.5.
            {
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&1.0f32.to_be_bytes());
                bytes.extend_from_slice(&(-2.5f32).to_be_bytes());
                bytes
            },
            vec![1.0f32, -2.5],
        ),
    ] {
        let dir = tempdir()?;
        let path = dir.path().join("dtype.mgh");
        let nframes: i32 = 2;
        let (nx, ny, nz) = match mri_type {
            MRI_UCHAR => (1, 4, 1),
            MRI_SHORT => (1, 2, 1),
            _ => (1, 1, 1),
        };
        let bytes = build_mgh_bytes(
            VERSION,
            [nx, ny, nz],
            nframes,
            mri_type,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &data_bytes,
        );
        std::fs::write(&path, &bytes)?;

        let series = read_mgh_series::<TestBackend, _>(&path, &TestBackend::default())?;
        let n_voxels = (nx * ny * nz) as usize;
        assert_eq!(series.len(), nframes as usize);
        for (frame, image) in series.iter().enumerate() {
            let start = frame * n_voxels;
            let slice = &expected[start..start + n_voxels];
            assert_eq!(
                image.data_slice().expect("contiguous host voxels"),
                slice,
                "frame {frame} of type {} must decode correctly",
                mri_type
            );
        }
    }
    Ok(())
}
