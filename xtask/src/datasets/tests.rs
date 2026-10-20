#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::catalog::validation::{
    extract_zip, is_html_payload, looks_like_nifti_header, validate_nifti_payload,
};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

#[test]
fn nifti_header_detection_accepts_nifti1_little_endian() {
    let h = 348_i32.to_le_bytes();
    assert!(looks_like_nifti_header(&h));
}

#[test]
fn html_payload_detection_flags_doctype() {
    let html = b"<!doctype html><html><head></head><body>not nii</body></html>";
    assert!(is_html_payload(html));
}

#[test]
fn validate_nifti_payload_rejects_html_masquerade() {
    let html = b"<!doctype html><html><head></head><body>404</body></html>";
    let err = validate_nifti_payload("bad.nii.gz", html)
        .unwrap_err()
        .to_string();
    assert!(err.contains("appears to be HTML"));
}

#[test]
fn validate_nifti_payload_accepts_valid_gzip_nifti_header() {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&348_i32.to_le_bytes()).unwrap();
    let gz = encoder.finish().unwrap();
    validate_nifti_payload("ok.nii.gz", &gz).unwrap();
}

fn stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, payload) in entries {
        writer
            .start_file(*name, SimpleFileOptions::default())
            .unwrap();
        writer.write_all(payload).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

#[test]
fn extract_zip_restores_nested_entries_byte_for_byte() {
    let archive = stored_zip(&[("a/b/c.nii", &[1, 2, 3, 4]), ("top.txt", b"top")]);
    let dest = tempfile::tempdir().unwrap();
    extract_zip(&archive, dest.path()).unwrap();
    assert_eq!(
        std::fs::read(dest.path().join("a/b/c.nii")).unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(std::fs::read(dest.path().join("top.txt")).unwrap(), b"top");
}

#[test]
fn extract_zip_never_writes_outside_the_destination() {
    let archive = stored_zip(&[("../escaped.txt", b"x"), ("kept.txt", b"ok")]);
    let parent = tempfile::tempdir().unwrap();
    let dest = parent.path().join("dest");
    std::fs::create_dir(&dest).unwrap();
    let err = extract_zip(&archive, &dest).unwrap_err();
    assert!(
        matches!(
            err.downcast_ref::<zip::result::ZipError>(),
            Some(zip::result::ZipError::InvalidArchive(_))
        ),
        "unexpected error: {err:?}"
    );
    assert!(!parent.path().join("escaped.txt").exists());
    assert!(!dest.join("escaped.txt").exists());
}
