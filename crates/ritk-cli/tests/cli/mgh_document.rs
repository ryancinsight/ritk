//! A container change must preserve all frames, the stored integer lanes,
//! header metadata and the bytes following the samples.
use std::{fs, path::Path, process::Command};

fn repack(source: &Path, output: &Path) -> std::io::Result<std::process::Output> {
    Command::new(env!("CARGO_BIN_EXE_ritk"))
        .arg("convert")
        .arg("--input")
        .arg(source)
        .arg("--output")
        .arg(output)
        .output()
}

#[test]
fn mgh_repack_preserves_frames_footer_and_existing_output_on_rejection() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let raw = directory.path().join("source.mgh");
    let compressed = directory.path().join("pack.MGH.GZ");
    let restored = directory.path().join("result.mgh");
    // Version 1, a 2×1×1 grid, two frames, MRI_INT, 37 degrees of freedom.
    // The RAS validity flag is unset. The opaque padding and footer must survive.
    let mut bytes: Vec<_> = [1_i32, 2, 1, 1, 2, 1, 37]
        .into_iter()
        .flat_map(i32::to_be_bytes)
        .collect();
    bytes.resize(284, 0);
    bytes[100] = 0x4e;
    for value in [16_777_217_i32, -16_777_217, i32::MIN, i32::MAX] {
        bytes.extend(value.to_be_bytes());
    }
    let sample_end = bytes.len();
    for value in [1.25_f32, 0.5, 4.0, 3.5, 256.0] {
        bytes.extend(value.to_be_bytes());
    }
    fs::write(&raw, &bytes)?;
    for (source, output) in [(&raw, &compressed), (&compressed, &restored)] {
        let result = repack(source, output)?;
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert!(fs::read(&compressed)?.starts_with(&[0x1f, 0x8b]));
    assert_eq!(fs::read(&restored)?, bytes);

    fs::write(&raw, &bytes[..sample_end - 1])?;
    assert!(!repack(&raw, &restored)?.status.success());
    assert_eq!(fs::read(&restored)?, bytes);
    Ok(())
}
