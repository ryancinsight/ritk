//! Value-semantic coverage for the unified image reader and writer contracts.

use crate::domain::{ImageReader, ImageWriter};
use coeus_core::SequentialBackend;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

/// Fixture tensor shape in RITK `[depth, row, col]` order.
///
/// All three extents differ, so an axis permutation changes the shape.
const DIMS: [usize; 3] = [2, 3, 4];
/// Fixture origin in RITK `[depth, row, col]` axis order.
///
/// Each component is an exact integer multiple of the matching spacing
/// component (`0.9 × 3`, `0.75 × −4`, `1.5 × 3`), because Analyze stores the
/// origin as an integer voxel coordinate and cannot represent an arbitrary
/// one. Choosing a representable origin keeps one fixture valid for every
/// codec instead of special-casing the quantizing format.
const ORIGIN: [f64; 3] = [2.7, -3.0, 4.5];
/// Fixture spacing in RITK `[Δdepth, Δrow, Δcol]` order.
///
/// Non-isotropic and non-monotonic, so an axis permutation cannot cancel out.
const SPACING: [f64; 3] = [1.5, 0.75, 0.9];

/// Which spatial fields a codec's writer→reader contract preserves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SpatialFidelity {
    /// The format stores no physical-space metadata, so geometry must be the
    /// default: zero origin, unit spacing, identity direction.
    None,
    /// Spacing and origin round-trip; direction is unrepresentable and must
    /// remain the identity the fixture writes.
    SpacingAndOrigin,
    /// Spacing, origin, and direction all round-trip.
    Full,
}

/// A direction with determinant −1, no symmetry, and axis-aligned unit columns.
///
/// Determinant −1 matches the canonical orientation every RITK codec produces
/// (`docs/architecture.md`), and the lack of symmetry means a transposed or
/// reordered direction matrix cannot round-trip unnoticed.
fn full_fidelity_direction() -> Direction<3> {
    Direction::from_rows([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
}

/// Assert two coordinate triples agree within a text-encoding tolerance.
///
/// Format headers store geometry as text or `f32`, so exact equality is not a
/// property of the contract; agreement to well below a micrometre is.
fn assert_geometry_close(actual: [f64; 3], expected: [f64; 3], label: &str) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() < 1e-6,
            "{label}[{axis}]: expected {}, got {}",
            expected[axis],
            actual[axis]
        );
    }
}

/// Round-trip a native volume through the unified [`crate::domain::ImageWriter`]
/// then [`crate::domain::ImageReader`] adapters; assert voxel, shape, and the
/// spatial metadata that `fidelity` says the format preserves.
///
/// This catches geometry a codec *drops* or rewrites asymmetrically — a writer
/// that discards `direction` while the reader returns identity, for instance.
/// It cannot catch a *self-consistent* transposition: a writer and reader that
/// apply the same axis permutation round-trip exactly, which is how
/// RITK-MGH-SPATIAL-AXIS-CONVENTION-001 survived a round-trip suite. Axis-order
/// correctness is therefore pinned by file-format oracles, not by round trips —
/// see `ritk-mgh`'s hand-built-header and byte-layout tests, `ritk-mif`'s
/// transform-less-header oracle, and `ritk-vtk`'s hand-built-file and
/// `SPACING`/`DIMENSIONS` agreement oracles.
fn assert_native_writer_reader_round_trips<W, R>(
    path: &Path,
    writer: &W,
    reader: &R,
    fidelity: SpatialFidelity,
) where
    W: crate::domain::ImageWriter<NativeImage<f32, SequentialBackend, 3>>,
    R: ImageReader<NativeImage<f32, SequentialBackend, 3>>,
{
    let n = DIMS[0] * DIMS[1] * DIMS[2];
    let voxels: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 4.0).collect();
    let direction = if fidelity == SpatialFidelity::Full {
        full_fidelity_direction()
    } else {
        Direction::identity()
    };
    let image = NativeImage::from_flat_on(
        voxels.clone(),
        DIMS,
        Point::new(ORIGIN),
        Spacing::new(SPACING),
        direction,
        &SequentialBackend,
    )
    .expect("native image");

    writer.write(path, &image).expect("contract write");
    let loaded: NativeImage<f32, SequentialBackend, 3> = reader.read(path).expect("contract read");

    assert_eq!(loaded.shape(), DIMS, "shape parity");
    assert_eq!(
        loaded.data_slice().expect("contiguous"),
        voxels.as_slice(),
        "native writer→reader contract must preserve voxels exactly"
    );

    let (expected_spacing, expected_origin) = match fidelity {
        SpatialFidelity::None => ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        _ => (SPACING, ORIGIN),
    };
    assert_geometry_close(loaded.spacing().to_array(), expected_spacing, "spacing");
    assert_geometry_close(loaded.origin().to_array(), expected_origin, "origin");

    let expected_direction = if fidelity == SpatialFidelity::Full {
        full_fidelity_direction()
    } else {
        Direction::identity()
    };
    for row in 0..3 {
        for col in 0..3 {
            let got = loaded.direction()[(row, col)];
            let want = expected_direction[(row, col)];
            assert!(
                (got - want).abs() < 1e-6,
                "direction[{row},{col}]: expected {want}, got {got}"
            );
        }
    }
}

#[test]
fn native_nrrd_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.nrrd"),
        &super::nrrd::native::NrrdWriter::new(SequentialBackend),
        &super::nrrd::native::NrrdReader::new(SequentialBackend),
        SpatialFidelity::Full,
    );
}

#[test]
fn native_analyze_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.hdr"),
        &super::analyze::AnalyzeWriter::new(SequentialBackend),
        &super::analyze::AnalyzeReader::new(SequentialBackend),
        SpatialFidelity::SpacingAndOrigin,
    );
}

#[test]
fn native_mgh_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mgh"),
        &super::mgh::native::MghWriter::new(SequentialBackend),
        &super::mgh::native::MghReader::new(SequentialBackend),
        SpatialFidelity::Full,
    );
}

#[test]
fn native_metaimage_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mha"),
        &super::metaimage::native::MetaImageWriter::new(SequentialBackend),
        &super::metaimage::native::MetaImageReader::new(SequentialBackend),
        SpatialFidelity::Full,
    );
}

#[test]
fn native_minc_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mnc"),
        &super::minc::native::MincWriter::new(SequentialBackend),
        &super::minc::native::MincReader::new(SequentialBackend),
        SpatialFidelity::Full,
    );
}

#[test]
fn native_mif_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mif"),
        &super::mif::native::MifWriter::new(SequentialBackend),
        &super::mif::native::MifReader::new(SequentialBackend),
        SpatialFidelity::Full,
    );
}

#[test]
fn native_vtk_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Legacy VTK structured points cannot carry a direction matrix, so the
    // contract stops at spacing and origin — exactly `SpacingAndOrigin`.
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.vtk"),
        &super::vtk::native::VtkWriter::new(SequentialBackend),
        &super::vtk::native::VtkReader::new(SequentialBackend),
        SpatialFidelity::SpacingAndOrigin,
    );
}

#[test]
fn native_tiff_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.tiff"),
        &super::tiff::native::TiffWriter::new(SequentialBackend),
        &super::tiff::native::TiffReader::new(SequentialBackend),
        SpatialFidelity::None,
    );
}

#[test]
fn native_tiff_reader_matches_coeus() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("vol.tiff"),
        &super::tiff::native::TiffWriter::new(SequentialBackend),
        &super::tiff::native::TiffReader::new(SequentialBackend),
        SpatialFidelity::None,
    );
}

#[test]
fn native_jpeg_reader_matches_coeus() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slice.jpg");
    let image = NativeImage::from_flat_on(
        vec![16.0, 128.0, 240.0],
        [1, 1, 3],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("native JPEG fixture");
    ImageWriter::write(
        &super::jpeg::native::JpegWriter::new(SequentialBackend),
        &path,
        &image,
    )
    .expect("jpeg write");
    let loaded = ImageReader::read(
        &super::jpeg::native::JpegReader::new(SequentialBackend),
        &path,
    )
    .expect("jpeg read");
    assert_eq!(loaded.shape(), [1, 1, 3]);
    let values = loaded.data_slice().expect("contiguous JPEG data");
    assert!(values[0] <= 24.0);
    assert!((values[1] - 128.0).abs() <= 12.0);
    assert!(values[2] >= 228.0);
}

/// Write a synthetic 8-bit grayscale PNG (no Coeus PNG writer exists).
fn write_gray_png(path: &Path, width: u32, height: u32, seed: u8) {
    let img = image::GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x * 7 + y * 13) as u8).wrapping_add(seed)])
    });
    img.save(path).expect("png save");
}

#[test]
fn native_png_reader_matches_coeus() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slice.png");
    write_gray_png(&path, 12, 8, 3);
    let loaded = ImageReader::read(
        &super::png::native::PngReader::new(SequentialBackend),
        &path,
    )
    .expect("native PNG read");
    assert_eq!(loaded.shape(), [1, 8, 12]);
    assert_eq!(loaded.data_slice().expect("contiguous PNG data").len(), 96);
}

#[test]
fn native_png_series_reader_matches_coeus() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_gray_png(&dir.path().join("s000.png"), 6, 4, 11);
    write_gray_png(&dir.path().join("s001.png"), 6, 4, 71);
    let loaded = ImageReader::read(
        &super::png::native::PngSeriesReader::new(SequentialBackend),
        dir.path(),
    )
    .expect("native PNG series read");
    assert_eq!(loaded.shape(), [2, 4, 6]);
    assert_eq!(loaded.data_slice().expect("contiguous PNG data").len(), 48);
}

/// PNG now writes through the unified contract, not only reads.
///
/// The fixture's extremes are exactly `0` and `255`, so the codec's min/max
/// window is the identity map and every intermediate value is representable in
/// 8 bits — the round trip is therefore exact rather than tolerance-bounded.
/// Values that fall between representable levels are still lossy, which is why
/// the codec's contract is stated as "ranks and shape", not "values".
#[test]
fn native_png_writer_reader_contract_round_trips_a_slice() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("contract.png");

    let values: Vec<f32> = (0..12).map(|i| (i * 255 / 11) as f32).collect();
    let image = NativeImage::from_flat_on(
        values.clone(),
        [1usize, 3, 4],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("slice fixture");

    ImageWriter::write(&super::png::native::PngWriter, &path, &image).expect("png write");
    let loaded = ImageReader::read(
        &super::png::native::PngReader::new(SequentialBackend),
        &path,
    )
    .expect("png read");

    assert_eq!(loaded.shape(), [1, 3, 4], "PNG round-trip preserves shape");
    assert_eq!(
        loaded.data_slice().expect("contiguous PNG data"),
        values.as_slice(),
        "with the window pinned at 0..255 the 8-bit map is the identity"
    );
}

/// The PNG writer rejects a volume rather than writing only its first slice.
///
/// This is the codec's shape policy, reached through the same `ImageWriter`
/// route the dispatch uses — not a dispatch-level special case.
#[test]
fn native_png_writer_rejects_a_volume() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("volume.png");
    let image = NativeImage::from_flat_on(
        vec![0.0f32; 24],
        [2usize, 3, 4],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("volume fixture");

    let error = ImageWriter::write(&super::png::native::PngWriter, &path, &image)
        .expect_err("a [2, 3, 4] volume is not a single PNG slice");
    assert!(
        error.to_string().contains("single slice"),
        "rejection must name the slice constraint, got: {error}"
    );
}
