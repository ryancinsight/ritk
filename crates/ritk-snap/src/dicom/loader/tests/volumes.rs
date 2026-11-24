//! Single-frame volume round-trips and byte-loader refusals.

use super::super::*;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use super::{assert_study, fixtures};

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

/// Spatial sorting must override reversed names, instance numbers, and byte order.
#[test]
fn dicom_file_bytes_and_scanned_studies_preserve_values_and_geometry() {
    for modality in ["CT", "MR"] {
        let dir = tempdir().expect("create DICOM fixture directory");
        let files = fixtures::write_study(
            dir.path(),
            modality,
            fixtures::SERIES_UID,
            &fixtures::SAMPLES,
        )
        .expect("write synthetic study");
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

#[test]
fn dicom_byte_loader_rejects_malformed_and_truncated_instances() {
    let dir = tempdir().expect("create malformed fixture directory");
    let files = fixtures::write_study(dir.path(), "CT", fixtures::SERIES_UID, &fixtures::SAMPLES)
        .expect("write truncation source");
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

/// NiBabel's MGH writer encoded this rotated single-slice short image.
/// RAS(x,y,z) = (11.125 - 0.75y, 19 + 0.5x, 28.75 + 1.25z).
#[test]
fn freesurfer_compressed_paths_preserve_samples_and_patient_coordinates() -> anyhow::Result<()> {
    use crate::ui::{decide_dropped_input_action, DroppedInput, DroppedInputAction};
    let dir = tempdir()?;
    let bytes = include_bytes!("../../../../tests/fixtures/freesurfer-affine.mgz");
    for filename in ["series.mgz", "series.mgh.gz", "series.MGH.GZ"] {
        let path = dir.path().join(filename);
        std::fs::write(&path, bytes)?;
        let drop = DroppedInput::new(Some(path.clone()), filename.into(), String::new(), None);
        assert_eq!(
            decide_dropped_input_action(&[drop]),
            DroppedInputAction::LoadVolume(path.clone())
        );
        let volume = load_volume_from_path(&path)?;
        assert_eq!(volume.source.as_deref(), Some(path.as_path()));
        assert_eq!(volume.channels, 1);
        assert_eq!(volume.shape, [1, 3, 4]);
        assert_eq!(volume.spacing, [1.25, 0.75, 0.5]);
        assert_eq!(
            volume.data.as_slice(),
            [-7., -4., -1., 2., 5., 8., 11., 14., 17., 20., 23., 26.]
        );
        // Probe each voxel and the plane normal even though depth is one.
        for depth in 0..=1 {
            for row in 0..3 {
                for col in 0..4 {
                    assert_eq!(
                        crate::ui::voxel_to_lps(
                            [depth, row, col],
                            volume.origin,
                            volume.direction,
                            volume.spacing
                        ),
                        [
                            -11.125 + 0.75 * row as f64,
                            -19.0 - 0.5 * col as f64,
                            28.75 + 1.25 * depth as f64
                        ]
                    );
                }
            }
        }
    }
    Ok(())
}
