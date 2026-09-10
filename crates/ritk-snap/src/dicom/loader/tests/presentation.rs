//! Grayscale presentation: rescale, inversion, and VOI.

use super::super::*;
use tempfile::tempdir;

use super::fixtures;

/// Signed stored samples, modality rescale, preserved VOI metadata, and
/// MONOCHROME1 inversion survive both filesystem and byte-batch ingress.
#[test]
fn dicom_grayscale_presentation_preserves_signed_rescale_and_inversion() {
    use crate::render::{GrayscalePresentation, NamedColorMap, SliceRenderer, WindowLevel};
    use ritk_io::DicomTag;

    let dir = tempdir().expect("create grayscale presentation directory");
    let (filename, bytes) =
        fixtures::write_grayscale_presentation(dir.path(), "MONOCHROME1", Some("LINEAR_EXACT"))
            .expect("write grayscale presentation fixture");
    let path = dir.path().join(&filename);
    let from_file = load_dicom_volume(&path).expect("load grayscale presentation file");
    let from_bytes = load_volume_from_bytes(&filename, &bytes).expect("load grayscale bytes");

    for volume in [&from_file, &from_bytes] {
        assert_eq!(volume.data.as_slice(), &[-30.0, -10.0, 10.0, 30.0]);
        assert_eq!(volume.shape, [1, 1, 4]);
        let metadata = volume.metadata.as_ref().expect("DICOM metadata retained");
        assert_eq!(
            metadata.photometric_interpretation.as_deref(),
            Some("MONOCHROME1")
        );
        let function = metadata.slices[0]
            .preservation
            .object
            .get(DicomTag::new(0x0028, 0x1056))
            .and_then(|node| node.value.as_text());
        assert_eq!(function, Some("LINEAR_EXACT"));

        let presentation = GrayscalePresentation::for_volume(volume)
            .expect("grayscale presentation metadata is admitted");
        assert_eq!(
            presentation.voi_function,
            crate::render::VoiLutFunction::LinearExact
        );
        assert!(presentation.invert);
        let rendered = SliceRenderer::render(
            volume,
            0,
            0,
            WindowLevel::new(0.0, 40.0),
            NamedColorMap::Grayscale,
        );
        let actual: Vec<[u8; 4]> = rendered
            .pixels
            .iter()
            .map(egui::Color32::to_array)
            .collect();
        assert_eq!(
            actual,
            [
                [255, 255, 255, 255],
                [191, 191, 191, 255],
                [64, 64, 64, 255],
                [0, 0, 0, 255]
            ]
        );
    }
    assert_eq!(from_file.source.as_deref(), Some(path.as_path()));
    assert_eq!(from_bytes.source, None);
}

#[test]
fn dicom_grayscale_presentation_rejects_unknown_voi_function() {
    let dir = tempdir().expect("create invalid grayscale presentation directory");
    let (filename, bytes) =
        fixtures::write_grayscale_presentation(dir.path(), "MONOCHROME2", Some("POLYNOMIAL"))
            .expect("write invalid grayscale presentation fixture");
    let path = dir.path().join(&filename);
    let file_error = load_dicom_volume(&path).expect_err("unsupported VOI function must reject");
    let byte_error = load_volume_from_bytes(&filename, &bytes)
        .expect_err("unsupported VOI function bytes must reject");
    for error in [file_error, byte_error] {
        let diagnostic = format!("{error:#}");
        assert!(
            diagnostic.contains("POLYNOMIAL"),
            "unsupported function remains visible: {diagnostic}"
        );
    }
}

/// Image pixels follow the loaded data on all three storage axes.
/// This uses an explicit window that preserves the fixture's stored samples.
#[test]
fn dicom_study_renders_independent_grayscale_oracles_on_all_axes() {
    use crate::render::{NamedColorMap, RenderBufferPool, SliceRenderer, WindowLevel};
    let dir = tempdir().expect("create render fixture directory");
    fixtures::write_study(dir.path(), "CT", fixtures::SERIES_UID).expect("write render study");
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
