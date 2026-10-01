#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;
use ritk_core::rejection::assert_rejects;
use ritk_image::Image;
use ritk_io::ImageReader;
use ritk_spatial::{Direction, Point, Spacing};
use tempfile::tempdir;

use crate::commands::Backend;

/// Build a small deterministic 3-D image for testing.
///
/// Shape is [3, 4, 5] (nz=3, ny=4, nx=5).  Voxel value at flat index i is
/// `i as f32`. Origin is `[4, 3, 2]`, direction is identity, and spacing is
/// `[1, 1.5, 2]`.
fn make_test_image() -> Image<f32, Backend, 3> {
    make_test_image_with_offset(0.0)
}

fn make_test_image_with_offset(offset: f32) -> Image<f32, Backend, 3> {
    let n = 3 * 4 * 5;
    let values: Vec<f32> = (0..n).map(|i| i as f32 + offset).collect();
    Image::from_flat_on(
        values,
        [3, 4, 5],
        Point::new([4.0, 3.0, 2.0]),
        Spacing::new([1.0, 1.5, 2.0]),
        Direction::identity(),
        &Backend::default(),
    )
    .expect("invariant: image data matches shape")
}

/// Every lossless volume writer and reader converts through the same RITK image
/// values. TIFF is the exception for spatial metadata: its native reader uses
/// unit spacing because TIFF has no physical-space fields.
#[test]
fn test_lossless_format_read_write_matrix() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("source.nii");
    let source = make_test_image();
    write_image(&input, &source, ImageFormat::NIfTI).unwrap();

    let cases = [
        ("nifti", "nii", OutputFormat::Nifti),
        ("metaimage", "mha", OutputFormat::MetaImage),
        ("nrrd", "nrrd", OutputFormat::Nrrd),
        ("minc", "mnc", OutputFormat::Minc),
        ("mgh", "mgh", OutputFormat::Mgh),
        ("tiff", "tiff", OutputFormat::Tiff),
        ("vtk", "vtk", OutputFormat::Vtk),
        ("analyze", "hdr", OutputFormat::Analyze),
    ];

    for (name, extension, format) in cases {
        let encoded = dir.path().join(format!("volume-{name}.{extension}"));
        run(ConvertArgs {
            input: input.clone(),
            output: encoded.clone(),
            format: Some(format),
            series_uid: None,
        })
        .unwrap_or_else(|error| panic!("{name} writer failed: {error:#}"));

        let decoded = dir.path().join(format!("decoded-{name}.nii"));
        run(ConvertArgs {
            input: encoded,
            output: decoded.clone(),
            format: None,
            series_uid: None,
        })
        .unwrap_or_else(|error| panic!("{name} reader failed: {error:#}"));

        let recovered = read_image(&decoded).unwrap();
        assert_eq!(recovered.shape(), source.shape(), "{name} shape");
        assert_eq!(
            recovered.data_slice().unwrap(),
            source.data_slice().unwrap(),
            "{name} voxel values"
        );
        if name == "tiff" {
            assert_eq!(recovered.origin().to_array(), [0.0; 3]);
            assert_eq!(recovered.spacing().to_array(), [1.0; 3]);
            assert_eq!(recovered.direction(), &Direction::identity());
        } else {
            assert_eq!(recovered.origin().to_array(), source.origin().to_array());
            assert_eq!(recovered.spacing().to_array(), source.spacing().to_array());
            assert_eq!(recovered.direction(), source.direction());
        }
    }
}

/// PNG conversion preserves integer grayscale samples and drops geometry,
/// which PNG does not represent.
#[test]
fn test_png_input_output_round_trip() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("slice.nii");
    let output = dir.path().join("slice.png");
    let source = Image::from_flat_on(
        vec![0.0, 255.0, 256.0, 65_535.0],
        [1, 2, 2],
        Point::new([4.0, 3.0, 2.0]),
        Spacing::new([1.0, 1.5, 2.0]),
        Direction::identity(),
        &Backend::default(),
    )
    .unwrap();
    write_image(&input, &source, ImageFormat::NIfTI).unwrap();

    run(ConvertArgs {
        input,
        output: output.clone(),
        format: Some(OutputFormat::Png),
        series_uid: None,
    })
    .unwrap();

    let recovered = read_image(&output).unwrap();
    assert_eq!(recovered.shape(), source.shape());
    assert_eq!(
        recovered.data_slice().unwrap(),
        source.data_slice().unwrap()
    );
    assert_eq!(recovered.origin().to_array(), [0.0; 3]);
    assert_eq!(recovered.spacing().to_array(), [1.0; 3]);
    assert_eq!(recovered.direction(), &Direction::identity());
}

/// JPEG conversion accepts one grayscale slice and rejects a volume rather
/// than silently dropping slices. The constant 8×8 block survives quality-75
/// JPEG quantization exactly under the codec's DC quantizer.
#[test]
fn test_jpeg_input_output_and_volume_rejection() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("slice.nii");
    let jpeg = dir.path().join("slice.jpg");
    let decoded_nifti = dir.path().join("decoded.nii");
    let slice = Image::from_flat_on(
        vec![128.0; 8 * 8],
        [1, 8, 8],
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
        &Backend::default(),
    )
    .expect("invariant: the raster data matches its 2-D shape");
    write_image(&input, &slice, ImageFormat::NIfTI).unwrap();

    run(ConvertArgs {
        input: input.clone(),
        output: jpeg.clone(),
        format: None,
        series_uid: None,
    })
    .unwrap();
    let raster = read_image(&jpeg).unwrap();
    assert_eq!(raster.shape(), [1, 8, 8]);
    assert_eq!(raster.data_slice().unwrap(), &[128.0; 64]);

    run(ConvertArgs {
        input: jpeg,
        output: decoded_nifti.clone(),
        format: None,
        series_uid: None,
    })
    .unwrap();
    let recovered = read_image(&decoded_nifti).unwrap();
    assert_eq!(recovered.shape(), raster.shape());
    assert_eq!(
        recovered.data_slice().unwrap(),
        raster.data_slice().unwrap()
    );

    let volume_input = dir.path().join("volume.nii");
    let volume_output = dir.path().join("volume.jpg");
    write_image(&volume_input, &make_test_image(), ImageFormat::NIfTI).unwrap();
    let error = run(ConvertArgs {
        input: volume_input,
        output: volume_output.clone(),
        format: Some(OutputFormat::Jpeg),
        series_uid: None,
    })
    .expect_err("JPEG must reject a multi-slice volume");
    let error = format!("{error:#}");
    assert!(error.contains("JPEG only supports 2-D images"), "{error}");
    assert!(
        !volume_output.exists(),
        "rejected volume leaves no JPEG file"
    );
}

// ── Positive: explicit --format overrides extension ───────────────────────

/// When `--format nifti` is passed the output extension is ignored and the
/// resulting file decodes as a NIfTI image with the original voxel values.
#[test]
fn test_convert_explicit_format_flag_overrides_extension() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    // Deliberately give the output a non-NIfTI extension.
    let output = dir.path().join("output.data");

    let image = make_test_image();
    write_image(&input, &image, ImageFormat::NIfTI).unwrap();

    run(ConvertArgs {
        input: input.clone(),
        output: output.clone(),
        format: Some(OutputFormat::Nifti),
        series_uid: None,
    })
    .unwrap();

    let reader = ritk_io::format::nifti::native::NiftiReader::new(Backend::default());
    let recovered = ImageReader::read(&reader, &output).unwrap();
    assert_eq!(recovered.shape(), image.shape());
    assert_eq!(recovered.data_slice().unwrap(), image.data_slice().unwrap());
}

// ── Negative: unknown output extension without --format ───────────────────

/// When the output path has an unrecognised extension and no `--format`
/// flag is provided, the command must return an error (not panic).
#[test]
fn test_convert_unknown_output_extension_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("output.xyz");

    let image = make_test_image();
    write_image(&input, &image, ImageFormat::NIfTI).unwrap();

    let result = run(ConvertArgs {
        input,
        output,
        format: None,
        series_uid: None,
    });
    let msg = result
        .expect_err("unknown output extension must yield an error")
        .to_string();
    assert!(
        msg.contains("Cannot infer output format"),
        "error must explain the problem, got: {msg}"
    );
}

// ── Negative: non-existent input file ─────────────────────────────────────

/// Attempting to convert a path that does not exist must return an error.
#[test]
fn test_convert_missing_input_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("does_not_exist.nii");
    let output = dir.path().join("output.nii");

    let result = run(ConvertArgs {
        input,
        output,
        format: None,
        series_uid: None,
    });
    assert_rejects(result, "Failed to read NIfTI file");
}

// ── ADR 0003 Phase A: native-dispatch coverage ────────────────────────────

/// `is_read_capable`/`is_write_capable` match every format route owned by
/// `ritk-io`; a drift would reject a real codec or misroute conversion.
#[test]
fn test_native_capability_predicates_match_dispatch() {
    use super::super::{is_read_capable, is_write_capable};

    let read_and_write = [
        ImageFormat::NIfTI,
        ImageFormat::Nrrd,
        ImageFormat::Analyze,
        ImageFormat::Mgh,
        ImageFormat::MetaImage,
        ImageFormat::Minc,
        ImageFormat::Tiff,
        ImageFormat::Jpeg,
        ImageFormat::Vtk,
        ImageFormat::Png,
    ];
    for fmt in read_and_write {
        assert!(is_read_capable(fmt), "{fmt:?} must read natively");
        assert!(is_write_capable(fmt), "{fmt:?} must write natively");
    }

    assert!(is_read_capable(ImageFormat::Png), "PNG reads natively");
    assert!(is_write_capable(ImageFormat::Png), "PNG writes natively");

    assert!(is_read_capable(ImageFormat::Dicom), "DICOM reads natively");
    assert!(
        is_write_capable(ImageFormat::Dicom),
        "DICOM writes natively"
    );

    assert!(is_read_capable(ImageFormat::Vtk), "VTK reads natively");
    assert!(is_write_capable(ImageFormat::Vtk), "VTK writes natively");
}

/// CLI conversion uses the same RITK writer as direct image output.
#[test]
fn test_native_convert_output_matches_direct_ritk_io_write() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let native_output = dir.path().join("via_convert.nii");
    let direct_output = dir.path().join("via_native_direct.nii");

    let image = make_test_image();
    write_image(&input, &image, ImageFormat::NIfTI).unwrap();

    run(ConvertArgs {
        input: input.clone(),
        output: native_output.clone(),
        format: None,
        series_uid: None,
    })
    .unwrap();

    ritk_io::write_image_native(&direct_output, &image).unwrap();

    assert_eq!(
        std::fs::read(&native_output).unwrap(),
        std::fs::read(&direct_output).unwrap(),
        "convert must preserve the native NIfTI serialization contract"
    );
}

#[path = "tests/dicom.rs"]
mod dicom;
