//! Confinement of the detached `file:` name to the header's directory.
//!
//! The name comes from the header, and `Path::join` discards its base for an
//! absolute argument. Each rejection case places a decodable payload at the
//! location the name points to, so an unguarded reader would return voxels
//! from outside the header's directory instead of failing.

use coeus_core::SequentialBackend;
use ritk_image::Image;

use super::header_with;
use crate::{read_mif, read_mif_series, MifError};

/// Voxel values of the two-voxel payload every detached fixture holds.
const DETACHED_VOXELS: [f32; 2] = [1.5, -2.25];

/// Write the two-voxel `Float32LE` payload at `path`, creating its directory.
fn write_detached_payload(path: &std::path::Path) {
    let parent = path.parent().expect("payload path has a parent");
    std::fs::create_dir_all(parent).expect("create payload directory");
    let bytes: Vec<u8> = DETACHED_VOXELS
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path, bytes).expect("write detached payload");
}

/// Write a header at `header_path` whose `file:` key is `file_key`.
fn write_detached_header(header_path: &std::path::Path, file_key: &str) {
    let parent = header_path.parent().expect("header path has a parent");
    std::fs::create_dir_all(parent).expect("create header directory");
    std::fs::write(header_path, header_with("2 1 1", file_key)).expect("write detached header");
}

/// An absolute or rooted `file:` name is rejected before any file opens.
///
/// The payload sits at the absolute location the header names, so reverting
/// the guard turns the expected error into a successful read of that file.
#[test]
fn an_absolute_detached_path_is_rejected_before_any_file_opens() {
    let backend = SequentialBackend;
    let dir = tempfile::tempdir().expect("temp dir");
    let payload = dir.path().join("outside.dat");
    write_detached_payload(&payload);
    assert!(
        payload.is_absolute(),
        "fixture must exercise an absolute path"
    );
    let absolute = payload.to_string_lossy().into_owned();

    let mut names = vec![absolute, "/outside.dat".to_owned()];
    if cfg!(windows) {
        names.push("C:outside.dat".to_owned());
        names.push(r"\server\share\outside.dat".to_owned());
    }
    for name in names {
        let header = dir.path().join("header").join("absolute.mif");
        write_detached_header(&header, &format!("{name} 0"));

        let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&header, &backend);
        let err = result.expect_err("an absolute detached path must not be opened");
        assert!(
            matches!(
                &err,
                MifError::DetachedPathNotConfined { path } if path == std::path::Path::new(&name)
            ),
            "name {name:?}: got {err}"
        );
    }
}

/// A `..` component is rejected wherever it appears in the `file:` name.
///
/// The payload sits in the header's parent directory, where `..` points.
#[test]
fn a_parent_directory_component_in_a_detached_path_is_rejected() {
    let backend = SequentialBackend;
    let dir = tempfile::tempdir().expect("temp dir");
    write_detached_payload(&dir.path().join("outside.dat"));
    let header = dir.path().join("header").join("parent.mif");

    for name in [
        "../outside.dat",
        "./../outside.dat",
        "inner/../../outside.dat",
    ] {
        write_detached_header(&header, &format!("{name} 0"));

        let result: Result<Image<f32, SequentialBackend, 3>, _> = read_mif(&header, &backend);
        let err = result.expect_err("a '..' detached path must not be opened");
        assert!(
            matches!(
                &err,
                MifError::DetachedPathNotConfined { path } if path == std::path::Path::new(name)
            ),
            "name {name:?}: got {err}"
        );
    }
}

/// The series reader shares the confinement check.
#[test]
fn the_series_reader_rejects_an_escaping_detached_path() {
    let backend = SequentialBackend;
    let dir = tempfile::tempdir().expect("temp dir");
    write_detached_payload(&dir.path().join("outside.dat"));
    let header = dir.path().join("header").join("series.mif");
    write_detached_header(&header, "../outside.dat 0");

    let result: Result<Vec<Image<f32, SequentialBackend, 3>>, _> =
        read_mif_series(&header, &backend);
    let err = result.expect_err("a '..' detached path must not be opened");
    assert!(
        matches!(&err, MifError::DetachedPathNotConfined { .. }),
        "got {err}"
    );
}

/// A sibling name, a subdirectory path, and a leading `./` stay confined and
/// read the payload's exact voxel values.
#[test]
fn a_confined_detached_path_reads_exact_voxels() {
    let backend = SequentialBackend;
    for name in ["payload.dat", "data/payload.dat", "./payload.dat"] {
        let dir = tempfile::tempdir().expect("temp dir");
        let header = dir.path().join("confined.mif");
        write_detached_header(&header, &format!("{name} 0"));
        write_detached_payload(&dir.path().join(name));

        let image: Image<f32, SequentialBackend, 3> =
            read_mif(&header, &backend).expect("a confined detached path reads");
        assert_eq!(image.shape(), [1, 1, 2], "name {name:?}");
        let voxels = image.data_cow_on(&backend).expect("host voxels");
        assert_eq!(&*voxels, DETACHED_VOXELS.as_slice(), "name {name:?}");
    }
}
