#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::catalog::validation::{
    extract_zip, is_html_payload, looks_like_nifti_header, member_path, validate_nifti_payload,
    UnsafeMemberPath,
};
use consus_zip::{Method, ZipWriter};
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

fn stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = ZipWriter::new(Vec::new());
    for (name, payload) in entries {
        writer.add_entry(name, payload, Method::Stored).unwrap();
    }
    writer.finish().unwrap()
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
fn extract_zip_creates_directory_members() {
    let archive = stored_zip(&[("empty/", b""), ("empty/inner/", b"")]);
    let dest = tempfile::tempdir().unwrap();
    extract_zip(&archive, dest.path()).unwrap();
    assert!(dest.path().join("empty/inner").is_dir());
}

/// Extracts an archive holding a safe member and `name`, returning the
/// rejection after checking that nothing was written.
fn rejected(name: &str) -> UnsafeMemberPath {
    let archive = stored_zip(&[("kept.txt", b"ok"), (name, b"x")]);
    let parent = tempfile::tempdir().unwrap();
    let dest = parent.path().join("dest");
    std::fs::create_dir(&dest).unwrap();
    let err = extract_zip(&archive, &dest).unwrap_err();
    assert_eq!(std::fs::read_dir(&dest).unwrap().count(), 0);
    assert_eq!(std::fs::read_dir(parent.path()).unwrap().count(), 1);
    err.downcast::<UnsafeMemberPath>().unwrap()
}

#[test]
fn extract_zip_never_writes_outside_the_destination() {
    for (name, reason) in [
        ("../escaped.txt", "contains a `.` or `..` component"),
        ("a/../../escaped.txt", "contains a `.` or `..` component"),
        ("./a", "contains a `.` or `..` component"),
        ("/abs.txt", "is absolute"),
        ("C:/abs.txt", "contains a backslash or colon"),
        ("a\\..\\..\\x", "contains a backslash or colon"),
        ("a/./b", "contains a `.` or `..` component"),
    ] {
        let err = rejected(name);
        assert_eq!((err.name.as_str(), err.reason), (name, reason));
    }
}

#[test]
fn member_path_rejects_an_empty_name_and_drops_empty_components() {
    assert_eq!(member_path("").unwrap_err().reason, "is empty");
    assert_eq!(
        member_path("a//b/").unwrap(),
        std::path::PathBuf::from("a").join("b")
    );
}

#[test]
fn extract_zip_refuses_to_overwrite_an_existing_file() {
    let archive = stored_zip(&[("a.txt", b"new")]);
    let dest = tempfile::tempdir().unwrap();
    std::fs::write(dest.path().join("a.txt"), b"old").unwrap();
    let err = extract_zip(&archive, dest.path()).unwrap_err();
    assert_eq!(
        err.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::AlreadyExists
    );
    assert_eq!(std::fs::read(dest.path().join("a.txt")).unwrap(), b"old");
}

#[cfg(unix)]
#[test]
fn extract_zip_refuses_to_write_through_a_symbolic_link() {
    let archive = stored_zip(&[("link/escaped.txt", b"x")]);
    let outside = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), dest.path().join("link")).unwrap();
    let err = extract_zip(&archive, dest.path()).unwrap_err();
    assert!(err.to_string().ends_with("is a symbolic link"), "{err}");
    assert_eq!(std::fs::read_dir(outside.path()).unwrap().count(), 0);
}
