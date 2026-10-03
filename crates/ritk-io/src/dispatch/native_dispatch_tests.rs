#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;
use ritk_spatial::{Direction, Point, Spacing};

fn native_volume() -> NativeImage {
    let dims = [2usize, 2, 3];
    let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 1.0).collect();
    NativeImage::from_flat(
        values,
        dims,
        Point::new([1.0, 2.0, 3.0]),
        Spacing::new([0.5, 0.75, 1.25]),
        Direction::identity(),
    )
    .expect("test image")
}

#[test]
fn native_capability_matrix_matches_dispatch() {
    for fmt in [
        ImageFormat::NIfTI,
        ImageFormat::MetaImage,
        ImageFormat::Nrrd,
        ImageFormat::Minc,
        ImageFormat::Mgh,
        ImageFormat::Tiff,
        ImageFormat::Vtk,
        ImageFormat::Jpeg,
        ImageFormat::Analyze,
    ] {
        assert!(is_native_read_capable(fmt), "{fmt:?} must read natively");
        assert!(is_native_write_capable(fmt), "{fmt:?} must write natively");
    }
    assert!(is_native_read_capable(ImageFormat::Png));
    assert!(is_native_read_capable(ImageFormat::Dicom));
    assert!(is_native_write_capable(ImageFormat::Png));
    assert!(is_native_write_capable(ImageFormat::Dicom));
    assert_eq!(
        ImageFormat::from_path(std::path::Path::new("scan.mnc")),
        Some(ImageFormat::Minc)
    );
    assert_eq!(ImageFormat::from_str_name("minc"), Some(ImageFormat::Minc));
}

#[test]
fn native_dispatch_round_trips_nrrd_values() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("native.nrrd");
    let image = native_volume();

    write_image_native(&path, &image).expect("native write");
    let loaded = read_image_native(&path).expect("native read");

    assert_eq!(loaded.shape(), image.shape());
    assert_eq!(loaded.data_slice().unwrap(), image.data_slice().unwrap());
    assert_eq!(loaded.origin(), image.origin());
    assert_eq!(loaded.spacing(), image.spacing());
}

#[test]
fn native_dispatch_round_trips_vtk_values() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("native.vtk");
    let image = native_volume();

    write_image_native(&path, &image).expect("native VTK write");
    let loaded = read_image_native(&path).expect("native VTK read");
    assert_eq!(loaded.shape(), image.shape());
    assert_eq!(loaded.data_slice().unwrap(), image.data_slice().unwrap());
    assert_eq!(loaded.origin(), image.origin());
    assert_eq!(loaded.spacing(), image.spacing());
}

#[test]
fn native_dispatch_writes_png_with_explicit_format() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("slice.png");
    let image = NativeImage::from_flat(
        vec![0.0, 255.0, 256.0, 65_535.0],
        [1, 2, 2],
        ritk_spatial::Point::new([4.0, 3.0, 2.0]),
        ritk_spatial::Spacing::new([1.0, 1.5, 2.0]),
        ritk_spatial::Direction::identity(),
    )
    .expect("test image");

    write_image_native_with_format(&path, &image, ImageFormat::Png).expect("write PNG");
    let loaded = read_image_native(&path).expect("read PNG");

    assert_eq!(loaded.shape(), [1, 2, 2]);
    assert_eq!(
        loaded.data_slice().expect("contiguous"),
        &[0.0, 255.0, 256.0, 65_535.0]
    );
    assert_eq!(loaded.origin().to_array(), [0.0; 3]);
    assert_eq!(loaded.spacing().to_array(), [1.0; 3]);
}
