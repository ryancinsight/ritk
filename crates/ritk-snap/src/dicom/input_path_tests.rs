use super::*;

#[test]
fn classify_directory_uses_directory_as_dicom_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let classified = classify_dicom_input_path(dir.path());

    assert_eq!(
        classified,
        DicomInputPath::Directory(dir.path().to_path_buf())
    );
    assert_eq!(
        classified.dicom_root(),
        Some(dir.path()),
        "directory input must be the DICOM root"
    );
}

#[test]
fn classify_dicomdir_file_uses_parent_as_dicom_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let dicomdir = dir.path().join("DICOMDIR");
    std::fs::write(&dicomdir, []).expect("create DICOMDIR marker file");

    let classified = classify_dicom_input_path(&dicomdir);

    assert_eq!(
        classified,
        DicomInputPath::DicomDirFile {
            file: dicomdir.clone(),
            root: dir.path().to_path_buf() }
    );
    assert_eq!(
        classified.dicom_root(),
        Some(dir.path()),
        "DICOMDIR file input must load from its parent directory"
    );
}

#[test]
fn classify_dicomdir_is_case_insensitive() {
    let dir = tempfile::tempdir().expect("tempdir");
    let dicomdir = dir.path().join("dicomdir");
    std::fs::write(&dicomdir, []).expect("create dicomdir marker file");

    let classified = classify_dicom_input_path(&dicomdir);

    assert_eq!(
        classified.dicom_root(),
        Some(dir.path()),
        "DICOMDIR file name matching must be case-insensitive"
    );
}

#[test]
fn classify_other_file_has_no_dicom_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("image.nii.gz");
    std::fs::write(&file, []).expect("create non-DICOMDIR file");

    let classified = classify_dicom_input_path(&file);

    assert_eq!(classified, DicomInputPath::OtherFile(file));
    assert_eq!(
        classified.dicom_root(),
        None,
        "non-DICOMDIR files must not be treated as DICOM roots"
    );
}

#[test]
fn classify_single_dicom_file_by_extension_uses_parent_as_dicom_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("slice_0001.dcm");
    std::fs::write(&file, b"not-a-real-dicom").expect("create .dcm file");

    let classified = classify_dicom_input_path(&file);

    assert_eq!(
        classified,
        DicomInputPath::SingleDicomFile {
            file: file.clone(),
            root: dir.path().to_path_buf() }
    );
    assert_eq!(
        classified.dicom_root(),
        Some(dir.path()),
        "single .dcm file input must load from parent directory"
    );
}

#[test]
fn classify_single_dicom_file_by_preamble_uses_parent_as_dicom_root() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("IM000001");

    let mut header = vec![0u8; 132];
    header[128..132].copy_from_slice(b"DICM");
    std::fs::write(&file, header).expect("create DICOM preamble file");

    let classified = classify_dicom_input_path(&file);

    assert_eq!(
        classified,
        DicomInputPath::SingleDicomFile {
            file: file.clone(),
            root: dir.path().to_path_buf() }
    );
    assert_eq!(
        classified.dicom_root(),
        Some(dir.path()),
        "DICM-preamble file input must load from parent directory"
    );
}
