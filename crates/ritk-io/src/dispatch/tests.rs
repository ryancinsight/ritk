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

/// A single `[1, rows, cols]` slice.
///
/// Slice-shaped on purpose: PNG's writer rejects a volume, so a volume
/// fixture would report a shape-policy failure as a missing capability.
/// Every other writer accepts this shape, which isolates the capability
/// query from per-codec shape rules.
fn slice_volume() -> NativeImage {
    let dims = [1usize, 3, 4];
    let values: Vec<f32> = (0..12).map(|i| i as f32 * 16.0).collect();
    NativeImage::from_flat(
        values,
        dims,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .expect("test slice")
}

/// Every [`ImageFormat`] variant, in one place.
///
/// The two tests below iterate this list rather than a hand-written
/// subset, so every format that gains a route is exercised by the
/// capability matrix.
///
/// Membership is maintained by hand and is **not** checked against the
/// enum: Rust cannot enumerate variants without a macro, and the
/// uniqueness assertion in `every_format_round_trips_through_path_and_name`
/// only proves this list has no duplicates. `canonical_extension` below
/// matches every variant exhaustively, so adding a variant is a compile
/// error *there*; nothing makes the same change add it here. A variant
/// missing from this list is silently untested — keep the two in step.
const ALL_FORMATS: [ImageFormat; 12] = [
    ImageFormat::NIfTI,
    ImageFormat::MetaImage,
    ImageFormat::Nrrd,
    ImageFormat::Png,
    ImageFormat::Dicom,
    ImageFormat::Mgh,
    ImageFormat::Tiff,
    ImageFormat::Vtk,
    ImageFormat::Jpeg,
    ImageFormat::Analyze,
    ImageFormat::Minc,
    ImageFormat::Mif,
];

/// The on-disk extension [`ImageFormat::from_path`] must map back to `fmt`.
///
/// Deliberately *not* `as_str()`: `as_str` is the CLI/Python name, while
/// `from_path` recognises real file extensions (`.nii`, `.mha`, `.jpg`),
/// and the two are not the same string. Keeping this table separate makes
/// a drifted `from_path` arm fail here rather than in a consumer.
fn canonical_extension(fmt: ImageFormat) -> &'static str {
    match fmt {
        ImageFormat::NIfTI => "nii",
        ImageFormat::MetaImage => "mha",
        ImageFormat::Nrrd => "nrrd",
        ImageFormat::Png => "png",
        ImageFormat::Dicom => "dcm",
        ImageFormat::Mgh => "mgh",
        ImageFormat::Tiff => "tiff",
        ImageFormat::Vtk => "vtk",
        ImageFormat::Jpeg => "jpg",
        ImageFormat::Analyze => "hdr",
        ImageFormat::Minc => "mnc",
        ImageFormat::Mif => "mif",
    }
}

/// Both format enumerations are mutually consistent for every listed variant.
///
/// `from_path` must invert `canonical_extension` and `from_str_name` must
/// invert `as_str`. A variant listed here but missing from `from_path`
/// shows up as a `None`. This says nothing about a variant the list omits:
/// see [`ALL_FORMATS`] for why membership cannot be verified here.
#[test]
fn every_format_round_trips_through_path_and_name() {
    let mut seen = std::collections::HashSet::new();
    for fmt in ALL_FORMATS {
        assert!(seen.insert(fmt), "{fmt:?} is listed in ALL_FORMATS twice");
        let path = std::path::PathBuf::from(format!("image.{}", canonical_extension(fmt)));
        assert_eq!(
            ImageFormat::from_path(&path),
            Some(fmt),
            "from_path must recognise {}",
            path.display()
        );
        assert_eq!(
            ImageFormat::from_str_name(fmt.as_str()),
            Some(fmt),
            "from_str_name must invert as_str for {fmt:?}"
        );
    }
    assert_eq!(
        seen.len(),
        ALL_FORMATS.len(),
        "ALL_FORMATS must not list any variant twice"
    );
}

/// The capability queries are not assertions about intent — they are
/// predictions about the dispatch, checked here against it.
///
/// For every format: `write_image_native` must succeed exactly when
/// [`is_native_write_capable`] says it can, and a file this module just
/// wrote must read back through [`read_image_native`] whenever
/// [`is_native_read_capable`] claims a reader. A format whose capability
/// flag drifts from its route fails here.
#[test]
fn native_capability_matrix_matches_dispatch() {
    for fmt in ALL_FORMATS {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join(format!("capability.{}", canonical_extension(fmt)));
        let image = slice_volume();

        let written = write_image_native(&path, &image);
        assert_eq!(
            written.is_ok(),
            is_native_write_capable(fmt),
            "{fmt:?}: write_image_native disagreed with is_native_write_capable \
             (capable={}, result={written:?})",
            is_native_write_capable(fmt)
        );

        if written.is_err() {
            continue;
        }
        assert!(
            is_native_read_capable(fmt),
            "{fmt:?}: writing a file this dispatch produced implies a reader"
        );
        let loaded = read_image_native(&path)
            .unwrap_or_else(|error| panic!("{fmt:?}: written file must read back: {error}"));
        assert_eq!(loaded.shape(), image.shape(), "{fmt:?}: shape parity");
    }
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

/// The dispatch rejects a volume written to a single PNG path.
///
/// PNG is a slice format: `write_png` refuses `depth != 1` rather than
/// silently keeping slice 0. The route is write-capable, so this is the
/// codec's shape policy rejecting an unrepresentable direction, not a
/// missing adapter.
#[test]
fn native_dispatch_rejects_a_png_volume() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("volume.png");
    let error = write_image_native(&path, &native_volume())
        .expect_err("a [2, 2, 3] volume is not a single PNG slice");
    assert!(
        error.to_string().contains("single slice"),
        "PNG volume rejection must name the slice constraint, got: {error}"
    );
}
