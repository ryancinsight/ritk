//! Tests for the loader module.

use super::*;
use crate::dicom::series_tree::SeriesEntry;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::borrow::Cow;
use tempfile::tempdir;

#[test]
fn sort_series_entries_is_deterministic() {
    let mut entries = vec![
        SeriesEntry {
            series_uid: Cow::Borrowed("UID-B"),
            folder: Cow::Borrowed(std::path::Path::new("z/path")),
            patient_name: Cow::Borrowed("B"),
            patient_id: Cow::Borrowed("P2"),
            modality: Cow::Borrowed("MR"),
            series_description: Cow::Borrowed("S2"),
            num_slices: 1,
            study_date: Some(Cow::Borrowed("20260102")),
            study_uid: Some(Cow::Borrowed("ST2")),
        },
        SeriesEntry {
            series_uid: Cow::Borrowed("UID-A2"),
            folder: Cow::Borrowed(std::path::Path::new("b/path")),
            patient_name: Cow::Borrowed("A"),
            patient_id: Cow::Borrowed("P1"),
            modality: Cow::Borrowed("CT"),
            series_description: Cow::Borrowed("S1"),
            num_slices: 1,
            study_date: Some(Cow::Borrowed("20260101")),
            study_uid: Some(Cow::Borrowed("ST1")),
        },
        SeriesEntry {
            series_uid: Cow::Borrowed("UID-A1"),
            folder: Cow::Borrowed(std::path::Path::new("a/path")),
            patient_name: Cow::Borrowed("A"),
            patient_id: Cow::Borrowed("P1"),
            modality: Cow::Borrowed("CT"),
            series_description: Cow::Borrowed("S1"),
            num_slices: 1,
            study_date: Some(Cow::Borrowed("20260101")),
            study_uid: Some(Cow::Borrowed("ST1")),
        },
    ];
    scan::sort_series_entries_deterministically(&mut entries);
    let ordered_uids: Vec<&str> = entries.iter().map(|e| e.series_uid.as_ref()).collect();
    assert_eq!(ordered_uids, vec!["UID-A1", "UID-A2", "UID-B"]);
}

/// Load the OpenNeuro T1w NIfTI file when it is present on disk and verify
/// that shape, spacing, and pixel data satisfy basic sanity invariants.
///
/// Skip the test when the file does not exist (CI environments without
/// large test data).
///
/// # Mathematical specification
/// - `shape[i] > 0` for all i (no zero-extent dimension).
/// - `spacing[i] > 0.0` for all i (positive voxel pitch).
/// - `pixels.len() == shape[0] * shape[1] * shape[2]`.
#[test]
fn test_load_nifti_volume_shape() {
    let path_buf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("openneuro")
        .join("sub-01_T1w.nii.gz");
    let path = path_buf.as_path();
    if !path.exists() {
        eprintln!(
            "SKIP test_load_nifti_volume_shape: test data absent at {:?}",
            path
        );
        return;
    }
    let vol =
        load_nifti_volume(path).expect("load_nifti_volume must succeed on a valid NIfTI file");
    // Shape invariant: every dimension must be > 0.
    let [d, r, c] = vol.shape;
    assert!(d > 0, "depth dimension must be > 0, got {d}");
    assert!(r > 0, "rows dimension must be > 0, got {r}");
    assert!(c > 0, "cols dimension must be > 0, got {c}");
    // Spacing invariant: all spacing values must be positive (mm/voxel).
    for (i, &sp) in vol.spacing.iter().enumerate() {
        assert!(sp > 0.0, "spacing[{i}] must be > 0.0 mm/voxel, got {sp}");
    }
    // Pixel count must exactly match the declared shape.
    let expected_len = d * r * c;
    assert_eq!(
        vol.data.len(),
        expected_len,
        "pixel data length {actual} must equal depth×rows×cols = {expected_len}",
        actual = vol.data.len(),
    );
    // Source path must be recorded.
    assert_eq!(
        vol.source.as_deref(),
        Some(path_buf.as_path()),
        "source path must be recorded in LoadedVolume"
    );
}

#[test]
fn test_load_volume_from_bytes_nifti_roundtrip_shape() {
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("drop_test.nii");
    let device = <B as ritk_image::tensor::Backend>::Device::default();
    let shape = Shape::new([3, 2, 4]);
    let data = TensorData::new((0..24).map(|v| v as f32).collect::<Vec<_>>(), shape);
    let tensor = Tensor::<B, 3>::from_data(data, &device);
    let image = Image::new(
        tensor,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.8, 0.9, 1.7]),
        Direction::identity(),
    );
    ritk_io::write_nifti(&path, &image).expect("write synthetic nifti");
    let bytes = std::fs::read(&path).expect("read written nifti bytes");
    let vol = load_volume_from_bytes("dropped.nii", &bytes)
        .expect("load_volume_from_bytes should load valid nifti bytes");
    assert_eq!(vol.shape, [3, 2, 4]);
    assert_eq!(vol.data.len(), 24);
    assert!(vol.spacing[0] > 0.0 && vol.spacing[1] > 0.0 && vol.spacing[2] > 0.0);
}

#[test]
fn test_load_dicom_series_from_named_bytes_batch() {
    let dir_buf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("2_head_mri_t2")
        .join("DICOM");
    let dir = dir_buf.as_path();
    if !dir.exists() {
        eprintln!(
            "SKIP test_load_dicom_series_from_named_bytes_batch: fixture absent at {:?}",
            dir
        );
        return;
    }
    let mut owned_files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .expect("read DICOM fixture directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .map(|path| {
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("slice.dcm")
                .to_owned();
            let bytes = std::fs::read(&path).expect("read DICOM fixture file bytes");
            (name, bytes)
        })
        .collect();
    owned_files.sort_by(|a, b| a.0.cmp(&b.0));
    let borrowed: Vec<(String, &[u8])> = owned_files
        .iter()
        .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
        .collect();
    let vol = load_dicom_series_from_named_bytes(&borrowed)
        .expect("load_dicom_series_from_named_bytes should load valid DICOM batch");
    let [d, r, c] = vol.shape;
    assert!(
        d > 0 && r > 0 && c > 0,
        "loaded DICOM shape must be non-zero"
    );
    assert_eq!(vol.data.len(), d * r * c);
}

/// Load the skull CT DICOM series when it is present and verify basic
/// structural invariants on the returned volume.
///
/// # Mathematical specification
/// - `shape[0] > 0` (depth, i.e. number of slices, must be positive).
/// - `shape[1] > 0` and `shape[2] > 0` (rows and cols must be positive).
/// - `spacing[i] > 0.0` for all i.
/// - `data.len() == shape[0] * shape[1] * shape[2]`.
#[test]
fn test_load_dicom_volume_shape() {
    let path_buf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("2_skull_ct")
        .join("DICOM");
    let path = path_buf.as_path();
    if !path.exists() {
        eprintln!(
            "SKIP test_load_dicom_volume_shape: test data absent at {:?}",
            path
        );
        return;
    }
    let vol =
        load_dicom_volume(path).expect("load_dicom_volume must succeed on a valid DICOM directory");
    let [depth, rows, cols] = vol.shape;
    assert!(depth > 0, "depth (num slices) must be > 0, got {depth}");
    assert!(rows > 0, "rows must be > 0, got {rows}");
    assert!(cols > 0, "cols must be > 0, got {cols}");
    for (i, &sp) in vol.spacing.iter().enumerate() {
        assert!(sp > 0.0, "spacing[{i}] must be > 0.0, got {sp}");
    }
    assert_eq!(
        vol.data.len(),
        depth * rows * cols,
        "pixel buffer length must equal depth×rows×cols"
    );
}

/// Load the head T2 MRI DICOM series (MRI-DIR porcine phantom, CC BY 4.0)
/// when present and verify basic structural invariants.
///
/// # Mathematical specification
/// - `shape[0] > 0` (depth / number of slices must be positive).
/// - `spacing[i] > 0.0` for all i.
/// - `modality == Some("MR")`.
/// - `data.len() == shape[0] * shape[1] * shape[2]`.
#[test]
fn test_load_head_mri_t2_volume_shape() {
    let path_buf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("test_data")
        .join("2_head_mri_t2")
        .join("DICOM");
    let path = path_buf.as_path();
    if !path.exists() {
        eprintln!(
            "SKIP test_load_head_mri_t2_volume_shape: test data absent at {:?}",
            path
        );
        return;
    }
    let vol = load_dicom_volume(path)
        .expect("load_dicom_volume must succeed on the head T2 MRI DICOM directory");
    let [depth, rows, cols] = vol.shape;
    assert!(depth > 0, "depth must be > 0, got {depth}");
    assert!(rows > 0, "rows must be > 0, got {rows}");
    assert!(cols > 0, "cols must be > 0, got {cols}");
    for (i, &sp) in vol.spacing.iter().enumerate() {
        assert!(sp > 0.0, "spacing[{i}] must be > 0.0 mm/voxel, got {sp}");
    }
    assert_eq!(
        vol.data.len(),
        depth * rows * cols,
        "pixel buffer length must equal depth×rows×cols"
    );
    // Modality must be MR.
    assert_eq!(
        vol.modality.as_deref(),
        Some("MR"),
        "modality must be MR for the T2 head series"
    );
}

/// `scan_folder_for_series` must return an empty [`SeriesTree`] — not an
/// error — when the target directory contains no DICOM files.
#[test]
fn test_scan_folder_for_series_empty_dir() {
    let dir = tempfile::tempdir().expect("tempdir must be created");
    let tree = scan_folder_for_series(dir.path())
        .expect("scan_folder_for_series must not error on an empty directory");
    assert_eq!(
        tree.total_series(),
        0,
        "empty directory must produce an empty SeriesTree"
    );
}

/// `load_dicom_series_from_stored_instances` must reject an empty input slice.
///
/// Analytical basis: the function's contract requires at least one instance
/// to construct a DICOM series. An empty slice produces a descriptive error
/// rather than silently succeeding or panicking.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_dicom_series_from_stored_instances_empty_input_errors() {
    let result = load_dicom_series_from_stored_instances(&[]);
    assert!(result.is_err());
    let msg = format!("{:#}", result.unwrap_err());
    assert!(
        msg.contains("no SCP-received DICOM instances"),
        "error must describe empty input, got: {msg}"
    );
}
