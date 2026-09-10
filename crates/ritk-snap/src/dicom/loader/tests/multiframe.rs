//! Multi-frame organization, channels, and its refusals.

use super::super::*;
use tempfile::tempdir;

use super::fixtures;

/// A multi-frame object must reach the RITK multi-frame reader as one spatial
/// volume. The ordinary series path requests frame zero and would therefore
/// lose the second frame; these assertions make that regression observable in
/// both filesystem and dropped-byte workflows.
#[test]
fn dicom_multiframe_file_and_bytes_preserve_all_frames_and_geometry() {
    let dir = tempdir().expect("create multiframe fixture directory");
    let (filename, bytes) =
        fixtures::write_multiframe(dir.path(), fixtures::MULTIFRAME_SHAPE[0], None)
            .expect("write multiframe fixture");
    let path = dir.path().join(&filename);
    let from_file = load_dicom_volume(&path).expect("load multiframe file");
    let from_bytes = load_volume_from_bytes(&filename, &bytes).expect("load multiframe bytes");
    let expected = [-8.0_f32, -6.0, -4.0, -2.0, 12.0, 14.0, 16.0, 18.0];

    for volume in [&from_file, &from_bytes] {
        assert_eq!(volume.data.as_slice(), expected);
        assert_eq!(volume.shape, fixtures::MULTIFRAME_SHAPE);
        assert_eq!(volume.channels, 1);
        assert_eq!(volume.spacing, [2.0, 1.5, 0.5]);
        assert_eq!(volume.origin, [10.0, 20.0, 30.0]);
        assert_eq!(
            volume.direction,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(volume.pixel_at(0, 0, 0), -8.0);
        assert_eq!(volume.pixel_at(1, 0, 0), 12.0);
        assert_eq!(
            volume
                .metadata
                .as_ref()
                .expect("multiframe metadata retained")
                .dimensions,
            fixtures::MULTIFRAME_SHAPE
        );
    }
    assert_eq!(from_file.source.as_deref(), Some(path.as_path()));
    assert_eq!(from_bytes.source, None);
}

/// The rendered axial frame must contain the second frame's values as well as
/// the decoded data, proving the viewer-facing image path is connected to the
/// multiframe reader rather than only a metadata probe.
#[test]
fn dicom_multiframe_render_uses_each_frame() {
    use crate::render::{NamedColorMap, SliceRenderer, WindowLevel};

    let dir = tempdir().expect("create multiframe render directory");
    let (filename, bytes) =
        fixtures::write_multiframe(dir.path(), 2, None).expect("write multiframe render fixture");
    let volume = load_volume_from_bytes(&filename, &bytes).expect("load multiframe render input");
    let window = WindowLevel::new(5.0, 26.0);

    let first = SliceRenderer::render(&volume, 0, 0, window, NamedColorMap::Grayscale);
    let second = SliceRenderer::render(&volume, 0, 1, window, NamedColorMap::Grayscale);
    let first_values: Vec<[u8; 4]> = first.pixels.iter().map(egui::Color32::to_array).collect();
    let second_values: Vec<[u8; 4]> = second.pixels.iter().map(egui::Color32::to_array).collect();
    let expected = |values: &[u8]| {
        values
            .iter()
            .map(|&value| [value, value, value, 255])
            .collect::<Vec<_>>()
    };
    assert_eq!(first_values, expected(&[0, 20, 41, 61]));
    assert_eq!(second_values, expected(&[204, 224, 245, 255]));
}

/// The real RITK RGB multi-frame decoder and both viewer-facing ingress paths
/// preserve every color channel through all three orthogonal slice views.
#[test]
fn dicom_color_multiframe_preserves_channels_and_display() {
    use crate::render::{NamedColorMap, SliceRenderer, WindowLevel};

    let dir = tempdir().expect("create RGB multiframe fixture directory");
    let (filename, bytes) =
        fixtures::write_color_multiframe(dir.path()).expect("write RGB multiframe fixture");
    let path = dir.path().join(&filename);
    let from_file = load_dicom_volume(&path).expect("load RGB multiframe file");
    let from_bytes = load_volume_from_bytes(&filename, &bytes).expect("load RGB multiframe bytes");
    let expected_data: Vec<f32> = fixtures::COLOR_MULTIFRAME_RAW
        .iter()
        .map(|&sample| f32::from(sample))
        .collect();

    for volume in [&from_file, &from_bytes] {
        assert_eq!(volume.shape, fixtures::COLOR_MULTIFRAME_SHAPE);
        assert_eq!(volume.channels, 3);
        assert_eq!(volume.data.as_slice(), expected_data.as_slice());
        assert_eq!(volume.pixel_channels(0, 0, 0), &[255.0, 0.0, 0.0]);
        assert_eq!(volume.pixel_channels(1, 1, 1), &[64.0, 64.0, 64.0]);
        let window = WindowLevel::new(-10_000.0, 1.0);
        let expected = [
            (
                0,
                0,
                vec![
                    [255, 0, 0, 255],
                    [0, 255, 0, 255],
                    [0, 0, 255, 255],
                    [255, 255, 255, 255],
                ],
            ),
            (
                1,
                0,
                vec![
                    [255, 0, 0, 255],
                    [0, 255, 0, 255],
                    [0, 255, 255, 255],
                    [255, 0, 255, 255],
                ],
            ),
            (
                2,
                0,
                vec![
                    [255, 0, 0, 255],
                    [0, 0, 255, 255],
                    [0, 255, 255, 255],
                    [255, 255, 0, 255],
                ],
            ),
        ];
        for (axis, index, expected_pixels) in expected {
            let rendered = SliceRenderer::render(volume, axis, index, window, NamedColorMap::Hot);
            let actual: Vec<[u8; 4]> = rendered
                .pixels
                .iter()
                .map(egui::Color32::to_array)
                .collect();
            assert_eq!(
                actual, expected_pixels,
                "RGB display channels for axis {axis}"
            );
        }
    }
    assert_eq!(from_file.source.as_deref(), Some(path.as_path()));
    assert_eq!(from_bytes.source, None);
}

#[test]
fn dicom_multiframe_rejects_temporal_organization() {
    let dir = tempdir().expect("create temporal fixture directory");
    let (filename, bytes) = fixtures::write_multiframe(dir.path(), 2, Some(2))
        .expect("write temporal multiframe fixture");
    let path = dir.path().join(&filename);
    let file_error = load_dicom_volume(&path).expect_err("temporal multiframe must reject");
    let byte_error = load_volume_from_bytes(&filename, &bytes)
        .expect_err("temporal multiframe bytes must reject");
    for error in [file_error, byte_error] {
        let diagnostic = format!("{error:#}");
        assert!(
            diagnostic.contains("temporal organization")
                || diagnostic.contains("NumberOfTemporalPositions"),
            "temporal rejection must remain explicit: {diagnostic}"
        );
    }
}

#[test]
fn dicom_multiframe_rejects_declared_frame_count_mismatch() {
    let dir = tempdir().expect("create invalid-count fixture directory");
    let (filename, bytes) = fixtures::write_multiframe(dir.path(), 3, None)
        .expect("write invalid-count multiframe fixture");
    let error = load_volume_from_bytes(&filename, &bytes)
        .expect_err("declared frame count exceeding pixel data must reject");
    let diagnostic = format!("{error:#}");
    assert!(
        diagnostic.contains("frame 2")
            || diagnostic.contains("expected")
            || diagnostic.contains("pixel"),
        "frame count mismatch must identify the decode failure: {diagnostic}"
    );
}
