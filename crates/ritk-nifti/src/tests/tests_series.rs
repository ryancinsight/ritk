//! Acquisition-series coverage: rank-4 round trips, the rank selection the
//! writer makes, and the rejections that keep a series from silently decoding
//! as its first volume.

use super::*;

/// Build `volumes` images on one grid, volume `v` filled with `v * 100 + i`.
///
/// Distinct per-volume values make an ordering or offset error detectable by
/// value, not only by length.
fn series_fixture(volumes: usize, dims: [usize; 3]) -> Vec<Image<f32, TestBackend, 3>> {
    let voxels = dims[0] * dims[1] * dims[2];
    (0..volumes)
        .map(|volume| {
            let values = (0..voxels)
                .map(|index| (volume * 100 + index) as f32)
                .collect();
            make_image(
                values,
                dims,
                Point::new([-11.0, 7.5, 3.25]),
                Spacing::new([2.0, 1.5, 0.75]),
                Direction::identity(),
            )
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
        "series volume count must round-trip"
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
fn series_round_trips_through_nifti1() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("series.nii");
    let backend = TestBackend::default();
    let expected = series_fixture(5, [2, 3, 4]);

    write_nifti_series(&path, &expected, &backend)?;
    let actual = read_nifti_series::<TestBackend, _>(&path, &backend)?;

    assert_series_matches(&actual, &expected);
    Ok(())
}

#[test]
fn series_round_trips_through_nifti2() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("series2.nii");
    let backend = TestBackend::default();
    let expected = series_fixture(3, [2, 2, 2]);

    write_nifti2_series(&path, &expected, &backend)?;
    let actual = read_nifti_series::<TestBackend, _>(&path, &backend)?;

    assert_series_matches(&actual, &expected);
    Ok(())
}

#[test]
fn series_round_trips_through_gzip() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("series.nii.gz");
    let backend = TestBackend::default();
    let expected = series_fixture(4, [2, 2, 3]);

    write_nifti_series(&path, &expected, &backend)?;
    let actual = read_nifti_series::<TestBackend, _>(&path, &backend)?;

    // The gzip read limit derives from the header's declared byte range, so a
    // series that did not extend that range would decompress short.
    assert_series_matches(&actual, &expected);
    Ok(())
}

#[test]
fn single_volume_series_writes_a_rank_three_header() -> Result<()> {
    // The canonical on-disk form for one volume is rank 3, so a one-element
    // series must not raise the rank and must stay readable by the ordinary
    // single-volume reader.
    let dir = tempdir()?;
    let path = dir.path().join("one.nii");
    let backend = TestBackend::default();
    let expected = series_fixture(1, [2, 2, 2]);

    write_nifti_series(&path, &expected, &backend)?;

    let header = NiftiHeader::parse(&std::fs::read(&path)?)?;
    assert_eq!(header.dim[0], 3, "a one-volume series is a rank-3 file");
    assert_eq!(header.volume_count(), 1);

    let single = read_nifti::<TestBackend, _>(&path, &backend)?;
    assert_eq!(
        single.data_slice().expect("contiguous host voxels"),
        expected[0].data_slice().expect("contiguous host voxels"),
        "a one-volume series reads back through the single-volume reader"
    );
    Ok(())
}

#[test]
fn multi_volume_series_writes_a_rank_four_header() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("many.nii");
    let backend = TestBackend::default();

    write_nifti_series(&path, &series_fixture(7, [2, 2, 2]), &backend)?;

    let header = NiftiHeader::parse(&std::fs::read(&path)?)?;
    assert_eq!(header.dim[0], 4, "a multi-volume series is a rank-4 file");
    assert_eq!(
        header.dim[4], 7,
        "the acquisition axis carries the volume count"
    );
    Ok(())
}

#[test]
fn rank_three_file_reads_as_a_one_volume_series() -> Result<()> {
    // The series reader is the general entry point: an ordinary volume is a
    // series of one, so no caller needs to branch on rank.
    let dir = tempdir()?;
    let path = dir.path().join("volume.nii");
    let backend = TestBackend::default();
    let image = series_fixture(1, [2, 2, 2]).remove(0);

    write_nifti(&path, &image, &backend)?;
    let series = read_nifti_series::<TestBackend, _>(&path, &backend)?;

    assert_eq!(series.len(), 1);
    assert_eq!(
        series[0].data_slice().expect("contiguous host voxels"),
        image.data_slice().expect("contiguous host voxels")
    );
    Ok(())
}

#[test]
fn single_volume_reader_rejects_a_series_rather_than_returning_volume_zero() -> Result<()> {
    // Volume 0 is decodable on its own, so the reader could return it and report
    // success. That is the failure this rejection exists to prevent.
    let dir = tempdir()?;
    let path = dir.path().join("reject.nii");
    let backend = TestBackend::default();
    write_nifti_series(&path, &series_fixture(6, [2, 2, 2]), &backend)?;

    let err = read_nifti::<TestBackend, _>(&path, &backend)
        .expect_err("a 6-volume series has no single-volume representation");
    let message = format!("{err:#}");

    assert!(
        message.contains("6 volumes"),
        "error must name the declared volume count, got: {message}"
    );
    Ok(())
}

#[test]
fn label_reader_rejects_a_series() -> Result<()> {
    // Same contract as the image reader: a label map is one volume, so a series
    // must fail rather than decode its first volume as the segmentation.
    let dir = tempdir()?;
    let path = dir.path().join("labels_series.nii");
    let backend = TestBackend::default();
    write_nifti_series(&path, &series_fixture(3, [2, 2, 2]), &backend)?;

    let err = read_nifti_labels(&path).expect_err("a series is not a label map");
    assert!(
        format!("{err:#}").contains("3 volumes"),
        "error must name the declared volume count"
    );
    Ok(())
}

#[test]
fn writer_rejects_an_empty_series() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("empty.nii");
    let backend = TestBackend::default();
    let empty: Vec<Image<f32, TestBackend, 3>> = Vec::new();

    let err = write_nifti_series(&path, &empty, &backend)
        .expect_err("a series with no volumes has no header to write");
    assert!(
        format!("{err:#}").contains("at least one volume"),
        "error must name the empty-series contract"
    );
}

#[test]
fn writer_rejects_volumes_on_different_grids() {
    // A NIfTI series carries one sform. Writing mismatched volumes would emit a
    // file whose geometry is correct for only some of its content.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mismatch.nii");
    let backend = TestBackend::default();

    let mut volumes = series_fixture(1, [2, 2, 2]);
    volumes.push(make_image(
        vec![0.0; 2 * 2 * 3],
        [2, 2, 3],
        Point::new([-11.0, 7.5, 3.25]),
        Spacing::new([2.0, 1.5, 0.75]),
        Direction::identity(),
    ));

    let err = write_nifti_series(&path, &volumes, &backend)
        .expect_err("volumes on different grids cannot share one sform");
    let message = format!("{err:#}");
    assert!(
        message.contains("volume 1") && message.contains("shape"),
        "error must name the offending volume and field, got: {message}"
    );
}

#[test]
fn writer_rejects_volumes_with_different_spacing() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("spacing.nii");
    let backend = TestBackend::default();

    let mut volumes = series_fixture(1, [2, 2, 2]);
    volumes.push(make_image(
        vec![0.0; 8],
        [2, 2, 2],
        Point::new([-11.0, 7.5, 3.25]),
        Spacing::new([2.0, 1.5, 0.5]),
        Direction::identity(),
    ));

    let err = write_nifti_series(&path, &volumes, &backend)
        .expect_err("differing spacing cannot share one sform");
    assert!(
        format!("{err:#}").contains("volume 1"),
        "error must name the offending volume"
    );
}

#[test]
fn truncated_series_payload_is_rejected() -> Result<()> {
    // The declared byte range spans every volume, so a file cut short after the
    // first volume must fail rather than decode the volumes that are present.
    let dir = tempdir()?;
    let path = dir.path().join("truncated.nii");
    let backend = TestBackend::default();
    write_nifti_series(&path, &series_fixture(4, [2, 2, 2]), &backend)?;

    let full = std::fs::read(&path)?;
    let one_volume_end = full.len() - 3 * 8 * std::mem::size_of::<f32>();
    std::fs::write(&path, &full[..one_volume_end])?;

    let err = read_nifti_series::<TestBackend, _>(&path, &backend)
        .expect_err("a truncated series payload must fail");
    assert!(
        format!("{err:#}").contains("truncated"),
        "error must name the truncation"
    );
    Ok(())
}
