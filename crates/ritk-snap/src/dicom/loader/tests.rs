//! Tests for the loader module.

mod fixtures;

use super::*;
use crate::dicom::series_tree::SeriesEntry;
use ritk_spatial::{Direction, Point, Spacing};
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

/// Both file and byte entry points preserve binary-exact NIfTI values and geometry.
#[test]
fn nifti_file_and_bytes_preserve_values_and_geometry() {
    let dir = tempdir().expect("create NIfTI fixture directory");
    let backend = coeus_core::SequentialBackend;
    let pixels: Vec<f32> = (0_u8..24)
        .map(|value| f32::from(value) * 0.5 - 4.0)
        .collect();
    let origin = [1.25, -2.5, 3.75];
    let spacing = [0.5, 1.5, 2.0];
    let image = ritk_image::Image::from_flat_on(
        pixels.clone(),
        fixtures::SHAPE,
        Point::new(origin),
        Spacing::new(spacing),
        Direction::identity(),
        &backend,
    )
    .expect("construct NIfTI fixture image");
    for filename in ["study.nii", "study.nii.gz"] {
        let path = dir.path().join(filename);
        ritk_io::write_image_native(&path, &image).expect("write NIfTI fixture");
        let bytes = std::fs::read(&path).expect("read NIfTI fixture bytes");
        let from_file = load_nifti_volume(&path).expect("load NIfTI file");
        let from_bytes = load_volume_from_bytes(filename, &bytes).expect("load NIfTI bytes");
        for volume in [&from_file, &from_bytes] {
            assert_eq!(volume.shape, fixtures::SHAPE);
            assert_eq!(volume.data.as_slice(), pixels);
            assert_eq!(volume.spacing, spacing);
            assert_eq!(volume.origin, origin);
            assert_eq!(
                volume.direction,
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            );
            assert_eq!(volume.channels, 1);
            assert!(volume.metadata.is_none(), "NIfTI carries no DICOM metadata");
        }
        assert_eq!(from_file.source.as_deref(), Some(path.as_path()));
    }
}

fn assert_study(volume: &crate::LoadedVolume, modality: &str) {
    // Integer samples and powers-of-two rescale coefficients are exact in f32.
    let expected: Vec<f32> = fixtures::SAMPLES
        .iter()
        .map(|&raw| 2.0 * f32::from(raw) - 20.0)
        .collect();
    assert_eq!(volume.data.as_slice(), expected);
    assert_eq!(volume.shape, fixtures::SHAPE);
    assert_eq!(volume.channels, 1);
    assert_eq!(volume.spacing, fixtures::SPACING);
    assert_eq!(volume.origin, fixtures::ORIGIN);
    assert_eq!(volume.direction, fixtures::DIRECTION);
    assert_eq!(volume.modality.as_deref(), Some(modality));
    let metadata = volume.metadata.as_ref().expect("DICOM metadata retained");
    assert_eq!(
        metadata.series_instance_uid.as_deref(),
        Some(fixtures::SERIES_UID)
    );
    assert_eq!(
        metadata.study_instance_uid.as_deref(),
        Some("2.25.20260905")
    );
    // P(d,r,c) = (10+2d, 20+0.5c, 30+1.5r), independently from the encoded IOP/IPP.
    assert_eq!(
        crate::ui::voxel_to_lps([2, 1, 3], volume.origin, volume.direction, volume.spacing),
        [14.0, 21.5, 31.5]
    );
}

/// Spatial sorting must override reversed names, instance numbers, and byte order.
#[test]
fn dicom_file_bytes_and_scanned_studies_preserve_values_and_geometry() {
    for modality in ["CT", "MR"] {
        let dir = tempdir().expect("create DICOM fixture directory");
        let files = fixtures::write_study(dir.path(), modality).expect("write synthetic study");
        let from_file = load_dicom_volume(dir.path()).expect("load DICOM directory");
        let named: Vec<(String, &[u8])> = files
            .iter()
            .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
            .collect();
        let from_bytes = load_dicom_series_from_named_bytes(&named).expect("load DICOM byte batch");
        let scan_input: Vec<(&str, &[u8])> = files
            .iter()
            .rev()
            .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
            .collect();
        let scanned = ritk_io::scan_dicom_part10_bytes(&scan_input).expect("scan DICOM byte batch");
        let from_scanned =
            dicom_load::load_volume_from_scanned_series(scanned).expect("load scanned DICOM study");
        for volume in [&from_file, &from_bytes, &from_scanned] {
            assert_study(volume, modality);
        }
        assert_eq!(from_file.source.as_deref(), Some(dir.path()));
        assert_eq!(from_bytes.source, None);
        assert_eq!(from_scanned.source, None);
    }
}

/// Image pixels follow the loaded data on all three storage axes.
/// This pins the current linear-exact display formula, not default DICOM LINEAR.
#[test]
fn dicom_study_renders_independent_grayscale_oracles_on_all_axes() {
    use crate::render::{NamedColorMap, RenderBufferPool, SliceRenderer, WindowLevel};
    let dir = tempdir().expect("create render fixture directory");
    fixtures::write_study(dir.path(), "CT").expect("write render study");
    let volume = load_dicom_volume(dir.path()).expect("load render study");
    let mut pool = RenderBufferPool::default();
    // L=-20, U=490 maps decoded (2*raw-20) to byte raw exactly.
    let window = WindowLevel::new(235.0, 510.0);
    let cases: [(usize, usize, [usize; 2], &[u8]); 3] = [
        (0, 1, [4, 2], &[80, 90, 100, 110, 120, 130, 140, 150]),
        (
            1,
            1,
            [4, 3],
            &[40, 50, 60, 70, 120, 130, 140, 150, 200, 210, 220, 230],
        ),
        (2, 2, [2, 3], &[20, 60, 100, 140, 180, 220]),
    ];
    for (axis, index, size, expected) in cases {
        let allocating =
            SliceRenderer::render(&volume, axis, index, window, NamedColorMap::Grayscale);
        let scratch = SliceRenderer::render_with_scratch(
            &mut pool,
            &volume,
            axis,
            index,
            window,
            NamedColorMap::Grayscale,
        );
        for image in [&allocating, &scratch] {
            assert_eq!(image.size, size);
            let actual: Vec<[u8; 4]> = image.pixels.iter().map(egui::Color32::to_array).collect();
            let rgba: Vec<[u8; 4]> = expected
                .iter()
                .map(|&value| [value, value, value, 255])
                .collect();
            assert_eq!(actual, rgba, "axis {axis}");
        }
    }
}

#[test]
fn dicom_byte_loader_rejects_malformed_and_truncated_instances() {
    let dir = tempdir().expect("create malformed fixture directory");
    let files = fixtures::write_study(dir.path(), "CT").expect("write truncation source");
    let (_, bytes) = files.first().expect("fixture includes a slice");
    // Header-only and mid-pixel truncation both reject; neither may produce a partial volume.
    let short_pixels = bytes
        .len()
        .checked_sub(3)
        .expect("fixture has pixel payload");
    for malformed in [
        &b"not a DICOM object"[..],
        &bytes[..132],
        &bytes[..short_pixels],
    ] {
        let error = load_dicom_series_from_named_bytes(&[("broken.dcm".to_owned(), malformed)])
            .expect_err("malformed DICOM instance must reject");
        let diagnostic = format!("{error:#}");
        assert!(
            diagnostic.contains("DICOM"),
            "failure retains format context: {diagnostic}"
        );
    }
    let error = load_dicom_series_from_named_bytes(&[]).expect_err("empty byte batch must reject");
    assert_eq!(error.to_string(), "empty DICOM byte batch");
}

#[test]
fn test_scan_folder_for_series_empty_dir() {
    let dir = tempdir().expect("create empty study directory");
    let tree = scan_folder_for_series(dir.path()).expect("scan empty directory");
    assert_eq!(tree.total_series(), 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_load_dicom_series_from_stored_instances_empty_input_errors() {
    let error =
        load_dicom_series_from_stored_instances(&[]).expect_err("empty SCP batch must reject");
    assert_eq!(error.to_string(), "no SCP-received DICOM instances to load");
}
