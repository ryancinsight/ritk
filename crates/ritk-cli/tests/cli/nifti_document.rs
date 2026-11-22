//! Single-file NIfTI conversions preserve the complete decompressed document.

use std::{fs, process::Command};

use ritk_nifti::NiftiDocument;

const INTEGER_SERIES: &[u8] = include_bytes!("../fixtures/integer-series.nii");

#[test]
fn nifti_repack_preserves_every_document_byte() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("integer-series.nii");
    let gzip = directory.path().join("integer-series.NII.GZ");
    let restored = directory.path().join("restored.nii");
    fs::write(&source, INTEGER_SERIES)?;
    for (input, output) in [(&source, &gzip), (&gzip, &restored)] {
        let result = Command::new(env!("CARGO_BIN_EXE_ritk"))
            .args(["convert", "--input"])
            .arg(input)
            .arg("--output")
            .arg(output)
            .output()?;
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            NiftiDocument::read(output)?.uncompressed_bytes(),
            INTEGER_SERIES
        );
    }
    assert!(fs::read(gzip)?.starts_with(&[0x1f, 0x8b]));
    assert_eq!(fs::read(restored)?, INTEGER_SERIES);
    Ok(())
}

#[test]
fn nifti_repack_rejects_truncated_samples_before_opening_destination() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let source = directory.path().join("truncated.nii");
    let output = directory.path().join("existing.nii");
    fs::write(&source, &INTEGER_SERIES[..INTEGER_SERIES.len() - 1])?;
    fs::write(&output, INTEGER_SERIES)?;
    let result = Command::new(env!("CARGO_BIN_EXE_ritk"))
        .args(["convert", "--input"])
        .arg(source)
        .arg("--output")
        .arg(&output)
        .output()?;
    assert!(!result.status.success());
    assert_eq!(fs::read(output)?, INTEGER_SERIES);
    Ok(())
}
