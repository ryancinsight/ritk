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
    let actual = read_mgh_series::<f32, _, TestBackend, _>(&path, &backend, Exact)?;

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
    let actual = read_mgh_series::<f32, _, TestBackend, _>(&path, &backend, Exact)?;

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

    let single = read_mgh::<f32, _, TestBackend, _>(&path, &backend, Exact)?;
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
    let series = read_mgh_series::<f32, _, TestBackend, _>(&path, &backend, Exact)?;

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
    let actual = read_mgh_series::<f32, _, TestBackend, _>(&path, &backend, Exact)?;

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

    let err = read_mgh::<f32, _, TestBackend, _>(&path, &backend, Exact)
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

    let err = read_mgh_series::<f32, _, TestBackend, _>(&path, &backend, Exact)
        .expect_err("a truncated series payload must fail");
    assert!(
        format!("{err:#}").contains("truncated"),
        "error must name the truncation"
    );
    Ok(())
}

/// Two frames of `frames[f]` stored as `code` on a `[nx, ny, nz]` grid read
/// back as two images of `T`, each frame bit for bit.
fn frames_read_in_the_stored_type<T: ritk_codecs::sample::Sample + std::fmt::Debug>(
    code: i32,
    [nx, ny, nz]: [i32; 3],
    frames: [&[T]; 2],
) -> Result<()> {
    let mut payload = Vec::new();
    for frame in frames {
        ritk_codecs::sample::write_samples(frame, consus_core::ByteOrder::BigEndian, &mut payload)?;
    }
    let dir = tempdir()?;
    let path = dir.path().join("dtype.mgh");
    let bytes = build_mgh_bytes(
        VERSION,
        [nx, ny, nz],
        2,
        code,
        [1.0, 1.0, 1.0],
        IDENTITY_DIR,
        [0.0, 0.0, 0.0],
        &payload,
    );
    std::fs::write(&path, &bytes)?;
    let series = read_mgh_series::<T, _, TestBackend, _>(&path, &TestBackend::default(), Exact)?;
    assert_eq!(series.len(), 2);
    for (position, (image, frame)) in series.iter().zip(frames).enumerate() {
        assert_eq!(
            image.data_slice()?,
            frame,
            "frame {position} of {}",
            <T as ritk_codecs::sample::Sample>::TYPE
        );
    }
    Ok(())
}

#[test]
fn multi_frame_supports_all_voxel_types() -> Result<()> {
    frames_read_in_the_stored_type(MRI_UCHAR, [1, 4, 1], [&[0_u8, 1, 2, 3], &[4, 5, 6, 255]])?;
    frames_read_in_the_stored_type(
        MRI_SHORT,
        [1, 2, 1],
        [&[10_i16, -20], &[i16::MAX, i16::MIN]],
    )?;
    // 2^24 + 1 has no exact f32: the frame must keep its stored i32.
    frames_read_in_the_stored_type(MRI_INT, [1, 1, 1], [&[16_777_217_i32], &[-1]])?;
    frames_read_in_the_stored_type(MRI_FLOAT, [1, 1, 1], [&[1.0_f32], &[-2.5]])
}
