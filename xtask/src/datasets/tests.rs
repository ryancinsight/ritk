use super::catalog::{is_html_payload, looks_like_nifti_header, validate_nifti_payload};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;

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
