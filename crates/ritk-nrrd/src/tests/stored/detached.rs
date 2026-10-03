//! Stored NRRD detached tests.

use super::*;

#[test]
fn detached_data_path_rejects_parent_traversal() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("volume.nhdr");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "data file: ../outside.raw",
    ];
    write_header(&path, &fields, &[])?;
    let error = read_nrrd_stored(&path).expect_err("parent traversal must be rejected");
    assert!(matches!(
        error,
        NrrdStoredReadError::InvalidDetachedPath { .. }
    ));
    Ok(())
}
