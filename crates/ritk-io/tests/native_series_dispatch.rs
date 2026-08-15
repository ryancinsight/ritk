use ritk_io::{
    read_image_native, read_image_series_native, read_native_dicom_series,
    write_dicom_series_native, write_image_native, NativeBackend, NativeImage, NativeSeries,
};
use ritk_spatial::{Direction, Point, Spacing};

fn native_volume() -> NativeImage {
    let dims = [2usize, 2, 3];
    let values: Vec<f32> = (0..12).map(|index| index as f32 * 0.5 - 1.0).collect();
    NativeImage::from_flat(
        values,
        dims,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 0.75, 1.25]),
        Direction::identity(),
    )
    .expect("test image")
}

fn native_series_fixture(volumes: usize, dims: [usize; 3]) -> NativeSeries {
    let voxel_count = dims[0] * dims[1] * dims[2];
    let backend = NativeBackend::default();
    (0..volumes)
        .map(|volume| {
            let values = (0..voxel_count)
                .map(|index| (volume * 100 + index) as f32 * 0.5 - 1.0)
                .collect();
            NativeImage::from_flat_on(
                values,
                dims,
                Point::new([1.0, 2.0, 3.0]),
                Spacing::new([0.5, 0.75, 1.25]),
                Direction::identity(),
                &backend,
            )
            .expect("series fixture image")
        })
        .collect()
}

fn assert_series_matches(actual: &NativeSeries, expected: &NativeSeries, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: volume count");
    for (position, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            got.shape(),
            want.shape(),
            "{context}: volume {position} shape"
        );
        assert_eq!(
            got.data_slice().expect("contiguous host voxels"),
            want.data_slice().expect("contiguous host voxels"),
            "{context}: volume {position} voxels"
        );
        assert_eq!(
            got.origin(),
            want.origin(),
            "{context}: volume {position} origin"
        );
        assert_eq!(
            got.spacing(),
            want.spacing(),
            "{context}: volume {position} spacing"
        );
        assert_eq!(
            got.direction(),
            want.direction(),
            "{context}: volume {position} direction"
        );
    }
}

#[test]
fn native_dispatch_reads_nifti_series() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("series.nii");
    let backend = NativeBackend::default();
    let expected = native_series_fixture(4, [2, 3, 4]);
    ritk_nifti::write_nifti_series(&path, &expected, &backend).expect("write NIfTI series");
    let actual = read_image_series_native(&path).expect("read via dispatch");
    assert_series_matches(&actual, &expected, "NIfTI series dispatch");
}

#[test]
fn native_dispatch_reads_nifti_gzip_series() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("series.nii.gz");
    let backend = NativeBackend::default();
    let expected = native_series_fixture(3, [2, 2, 3]);
    ritk_nifti::write_nifti_series(&path, &expected, &backend).expect("write gzipped NIfTI series");
    let actual = read_image_series_native(&path).expect("read via dispatch");
    assert_series_matches(&actual, &expected, "gzipped NIfTI series dispatch");
}

#[test]
fn native_dispatch_reads_nrrd_series() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("series.nrrd");
    let backend = NativeBackend::default();
    let expected = native_series_fixture(5, [2, 3, 4]);
    ritk_nrrd::write_nrrd_series(&path, &expected, &backend).expect("write NRRD series");
    let actual = read_image_series_native(&path).expect("read via dispatch");
    assert_series_matches(&actual, &expected, "NRRD series dispatch");
}

#[test]
fn native_dispatch_reads_mgh_and_mgz_series() {
    let dir = tempfile::tempdir().expect("tempdir");
    let backend = NativeBackend::default();
    let expected = native_series_fixture(6, [2, 3, 2]);
    for extension in ["mgh", "mgz", "mgh.gz"] {
        let path = dir.path().join(format!("series.{extension}"));
        ritk_mgh::write_mgh_series(&path, &expected, &backend).expect("write MGH series");
        let actual = read_image_series_native(&path).expect("read via dispatch");
        assert_series_matches(&actual, &expected, extension);
    }
}

#[test]
fn native_dispatch_reads_compound_mgh_gzip_suffix() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("volume.mgh.gz");
    let expected = native_volume();
    write_image_native(&path, &expected).expect("write compressed MGH");
    let actual = read_image_native(&path).expect("read compressed MGH through dispatch");
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(
        actual.data_slice().expect("contiguous actual voxels"),
        expected.data_slice().expect("contiguous expected voxels")
    );
}

#[test]
fn native_dispatch_reads_dicom_directory_as_one_volume() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("dicom-series");
    let dims = [3, 2, 4];
    let values = (0..dims.iter().product())
        .map(|index| index as f32 * 1.25 - 7.0)
        .collect();
    let writer_image = ritk_image::Image::<f32, coeus_core::MoiraiBackend, 3>::from_flat(
        values,
        dims,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 0.75, 1.25]),
        Direction::identity(),
    )
    .expect("DICOM writer image");
    write_dicom_series_native(&path, &writer_image).expect("write DICOM directory");

    let backend = NativeBackend::default();
    let expected =
        vec![read_native_dicom_series(&path, &backend).expect("read direct DICOM directory")];
    let actual = read_image_series_native(&path).expect("read DICOM directory through dispatch");
    assert_series_matches(&actual, &expected, "DICOM directory dispatch");
}

#[test]
fn native_dispatch_reads_single_volume_series() {
    let dir = tempfile::tempdir().expect("tempdir");
    let backend = NativeBackend::default();
    let expected = native_series_fixture(1, [2, 3, 4]);

    let nifti_path = dir.path().join("one.nii");
    ritk_nifti::write_nifti_series(&nifti_path, &expected, &backend).expect("write NIfTI");
    assert_series_matches(
        &read_image_series_native(&nifti_path).expect("read NIfTI"),
        &expected,
        "NIfTI single-volume series",
    );

    let nrrd_path = dir.path().join("one.nrrd");
    ritk_nrrd::write_nrrd_series(&nrrd_path, &expected, &backend).expect("write NRRD");
    assert_series_matches(
        &read_image_series_native(&nrrd_path).expect("read NRRD"),
        &expected,
        "NRRD single-volume series",
    );

    let mgh_path = dir.path().join("one.mgh");
    ritk_mgh::write_mgh_series(&mgh_path, &expected, &backend).expect("write MGH");
    assert_series_matches(
        &read_image_series_native(&mgh_path).expect("read MGH"),
        &expected,
        "MGH single-volume series",
    );
}

#[test]
fn cross_codec_series_differential_nifti_nrrd_mgh() {
    let dir = tempfile::tempdir().expect("tempdir");
    let backend = NativeBackend::default();
    let expected = native_series_fixture(4, [3, 4, 5]);
    let nii_path = dir.path().join("differential.nii");
    let nrrd_path = dir.path().join("differential.nrrd");
    let mgh_path = dir.path().join("differential.mgh");
    ritk_nifti::write_nifti_series(&nii_path, &expected, &backend).expect("write NIfTI");
    ritk_nrrd::write_nrrd_series(&nrrd_path, &expected, &backend).expect("write NRRD");
    ritk_mgh::write_mgh_series(&mgh_path, &expected, &backend).expect("write MGH");
    let nii = read_image_series_native(&nii_path).expect("read NIfTI");
    let nrrd = read_image_series_native(&nrrd_path).expect("read NRRD");
    let mgh = read_image_series_native(&mgh_path).expect("read MGH");
    assert_series_matches(&nii, &expected, "NIfTI vs fixture");
    assert_series_matches(&nrrd, &expected, "NRRD vs fixture");
    assert_series_matches(&mgh, &expected, "MGH vs fixture");
    assert_series_matches(&nii, &nrrd, "NIfTI vs NRRD");
    assert_series_matches(&nii, &mgh, "NIfTI vs MGH");
    assert_series_matches(&nrrd, &mgh, "NRRD vs MGH");
}

#[test]
fn native_dispatch_rejects_unsupported_series_format() {
    let dir = tempfile::tempdir().expect("tempdir");
    let vtk_path = dir.path().join("image.vtk");
    write_image_native(&vtk_path, &native_volume()).expect("write VTK image");
    let error = read_image_series_native(&vtk_path).expect_err("VTK has no series reader");
    assert!(format!("{error:#}").contains("not yet supported"));
}

// ── Series writing through the dispatch ───────────────────────────────────────

/// A series survives a write/read round trip through the dispatch, per format.
///
/// The oracle is the reader that already existed: whatever
/// `write_image_series_native` produces must come back through
/// `read_image_series_native` unchanged. Checking voxel values and not just the
/// volume count is what distinguishes a real round trip from a file that merely
/// parses.
#[test]
fn a_written_series_reads_back_unchanged() {
    let dims = [2usize, 3, 4];
    let expected = native_series_fixture(3, dims);
    let dir = tempfile::tempdir().expect("tempdir");

    for extension in ["nii", "nrrd", "mgh"] {
        let path = dir.path().join(format!("series.{extension}"));
        ritk_io::write_image_series_native(&path, &expected)
            .unwrap_or_else(|error| panic!(".{extension} series writes: {error:#}"));

        let actual = read_image_series_native(&path)
            .unwrap_or_else(|error| panic!(".{extension} series reads back: {error:#}"));

        assert_eq!(
            actual.len(),
            expected.len(),
            ".{extension} must return every volume written"
        );
        for (index, (written, read)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(read.shape(), written.shape(), ".{extension} volume {index}");
            assert_eq!(
                read.data_slice().expect("contiguous"),
                written.data_slice().expect("contiguous"),
                ".{extension} volume {index} voxels must survive the round trip"
            );
        }
    }
}

#[test]
fn an_unwritable_format_is_rejected_by_name() {
    // The message has to name the format, because the caller's next step is to
    // reach for that format's own series writer.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("series.png");

    let error = ritk_io::write_image_series_native(&path, &native_series_fixture(2, [2, 2, 2]))
        .expect_err("PNG has no native series writer");
    let message = format!("{error:#}");
    assert!(
        message.contains("series I/O is not yet supported"),
        "unexpected error: {message}"
    );
}

#[test]
fn an_unknown_extension_is_rejected_before_writing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("series.unknown-extension");

    let error = ritk_io::write_image_series_native(&path, &native_series_fixture(2, [2, 2, 2]))
        .expect_err("no format can be inferred");
    assert!(
        format!("{error:#}").contains("cannot infer native series output format"),
        "unexpected error: {error:#}"
    );
}
