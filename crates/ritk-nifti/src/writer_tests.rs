use super::*;
use crate::header::{HeaderDims, HeaderSpatial, NiftiDatatype};
use anyhow::Result;
use std::fs;
use tempfile::tempdir;

#[test]
fn invalid_header_does_not_truncate_an_existing_destination() -> Result<()> {
    let directory = tempdir()?;
    let destination = directory.path().join("existing.nii");
    let original = b"preserve existing image";
    fs::write(&destination, original)?;

    let header = NiftiHeader::new_volume(
        HeaderDims {
            nx: 1,
            ny: 1,
            nz: 1,
        },
        NiftiDatatype::Float32,
        HeaderSpatial {
            pixdim: [1.0; 8],
            srow_x: [1.0, 0.0, 0.0, f64::MAX],
            srow_y: [0.0, 1.0, 0.0, 0.0],
            srow_z: [0.0, 0.0, 1.0, 0.0],
        },
    )?;

    let error = write_single_file_with(&destination, &header, |writer| {
        writer.write_all(b"replacement")?;
        Ok(())
    })
    .expect_err("unrepresentable header must fail before opening the destination");

    assert!(error.to_string().contains("f32-representable"));
    assert_eq!(fs::read(destination)?, original);
    Ok(())
}
