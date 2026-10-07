//! Differential parity and round-trip verification for the native series
//! writer [`write_dicom_series_native`].
//!
//! Native series writer parity and round-trip verification.

use crate::format::dicom::read_native_dicom_series;
use crate::format::dicom::writer::write_dicom_series_native;
use crate::format::dicom::{load_dicom_series_with_metadata, write_dicom_series};
use coeus_core::{MoiraiBackend, SequentialBackend};
use eunomia::FloatElement;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

const DIMS: [usize; 3] = [3, 4, 5];
const ORIGIN: [f64; 3] = [1.0, -2.0, 3.0];
const SPACING: [f64; 3] = [0.75, 0.5, 2.5];

/// Canonical RITK axis-aligned orientation for tensor axes `[depth, row, col] =
/// [z, y, x]`: depth→+z, row→+y, col→+x (`docs/architecture.md` §7–§9, and the
/// DICOM readers' `assemble_direction` fallback). `Direction::identity()` is
/// *not* this orientation — it maps depth→+x and col→+z, which is a transposed
/// volume and (as the tests below pin) not faithfully DICOM-representable with
/// preserved index order.
fn canonical_direction() -> Direction<3> {
    Direction::from_row_major([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
}

fn ramp() -> Vec<f32> {
    let n = DIMS[0] * DIMS[1] * DIMS[2];
    (0..n).map(|i| f32::from_count(i) * 1.5 - 37.0).collect()
}

fn native_image(data: Vec<f32>) -> NativeImage<f32, MoiraiBackend, 3> {
    NativeImage::<f32, MoiraiBackend, 3>::from_flat(
        data,
        DIMS,
        Point::new(ORIGIN),
        Spacing::new(SPACING),
        canonical_direction(),
    )
    .expect("native series image construction")
}

/// Per-voxel absolute tolerance for the unsigned-16-bit rescale round-trip.
///
/// For range `R`, the stored slope is `R / 65535`; integer rounding contributes
/// at most half a slope step. Four f32 operations in normalization contribute
/// at most `γ₄·65535` codes, where `γₙ = n·u/(1−n·u)` and `u = ε/2`. DICOM DS
/// carries at least nine significant digits in 16 bytes; decimal and f32
/// rescale parsing contribute at most `(5e-9 + u)(R + max|value|)`. The reader's
/// multiply and add contribute `γ₂(R + max|value|)`. The fixture's linear ramp
/// gives every slice the same `R = (slice_len − 1)·1.5`.
fn voxel_tolerance(data: &[f32]) -> f32 {
    let slice_len = DIMS[1] * DIMS[2];
    let slice_range = (f32::from_count(slice_len) - 1.0) * 1.5;
    let slope = slice_range / 65535.0_f32;
    let unit_roundoff = f32::EPSILON / 2.0_f32;
    let gamma_four = 4.0 * unit_roundoff / (1.0 - 4.0 * unit_roundoff);
    let gamma_two = 2.0 * unit_roundoff / (1.0 - 2.0 * unit_roundoff);
    let max_absolute_value = data.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let rescale_bound = slice_range + max_absolute_value;
    slope * (0.5 + gamma_four * 65535.0)
        + (5.0e-9 + unit_roundoff) * rescale_bound
        + gamma_two * rescale_bound
}

fn assert_voxels_within_tolerance(original: &[f32], recovered: &[f32], label: &str) {
    let tol = voxel_tolerance(original);
    assert_eq!(recovered.len(), original.len(), "{label}: voxel count");
    for (idx, (&orig, &got)) in original.iter().zip(recovered.iter()).enumerate() {
        let err = (got - orig).abs();
        assert!(
            err <= tol,
            "{label}: voxel[{idx}]: |{got} - {orig}| = {err} > tol {tol}"
        );
    }
}

#[test]
fn native_series_writer_round_trips_native_image() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let native_dir = tmp.path().join("series_native");

    let data = ramp();
    write_dicom_series_native(&native_dir, &native_image(data.clone())).expect("native write");
    let native = read_native_dicom_series(&native_dir, &SequentialBackend).expect("native read");

    assert_eq!(native.shape(), DIMS, "shape parity");
    assert_voxels_within_tolerance(
        &data,
        native.data_slice().expect("native contiguous"),
        "native round-trip",
    );
    assert_eq!(native.origin().to_array(), ORIGIN, "origin parity");
    assert_eq!(native.spacing().to_array(), SPACING, "spacing parity");
    assert_eq!(
        native.direction().to_row_major(),
        canonical_direction().to_row_major(),
        "direction parity"
    );
}

/// A native-written series round-trips through the native reader to the same
/// voxels (within the per-slice rescale bound, see [`voxel_tolerance`]) and
/// geometry.
#[test]
fn native_series_writer_round_trips_through_native_reader() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("series_rt");

    let data = ramp();
    write_dicom_series_native(&dir, &native_image(data.clone())).expect("native write");

    let reloaded = read_native_dicom_series(&dir, &SequentialBackend).expect("native read");
    assert_eq!(reloaded.shape(), DIMS, "shape must round-trip");

    assert_voxels_within_tolerance(
        &data,
        reloaded.data_slice().expect("contiguous reloaded data"),
        "native reader round-trip",
    );

    // Every geometry value in this fixture has an exact DS representation.
    for k in 0..3 {
        assert_eq!(reloaded.origin().to_array()[k], ORIGIN[k]);
        assert_eq!(reloaded.spacing().to_array()[k], SPACING[k]);
    }
    let expected_dir = canonical_direction().to_row_major();
    for (k, (&got, &exp)) in reloaded
        .direction()
        .to_row_major()
        .iter()
        .zip(expected_dir.iter())
        .enumerate()
    {
        assert_eq!(got, exp, "direction[{k}] must round-trip");
    }
}

/// Both DICOM series loaders must agree with the canonical RITK axis order.
///
/// RITK tensor data is `[depth, row, col]` (axis 0 slowest-varying), so
/// `Spacing<3>` is `[Δdepth, Δrow, Δcol]` and direction column 0 is the depth
/// axis — the order NIfTI, NRRD, MetaImage and Analyze all produce
/// (`docs/architecture.md` §7–§9), and the order `reader::assemble_direction`
/// already assembles. `read_native_dicom_series` (the `series` module loader)
/// and `load_dicom_series_with_metadata` (the `reader` module loader) are two
/// independent read paths over the same files; before this test they disagreed
/// on the direction axes. `SPACING` and `DIMS` are deliberately non-isotropic
/// and non-cubic so a transposition cannot cancel out.
#[test]
fn dicom_series_loaders_preserve_canonical_axis_order() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("series_axis_order");

    let data = ramp();
    write_dicom_series_native(&dir, &native_image(data.clone())).expect("native write");

    let native = read_native_dicom_series(&dir, &SequentialBackend).expect("native read");
    let (legacy, legacy_meta) =
        load_dicom_series_with_metadata::<SequentialBackend, _>(&dir, &SequentialBackend)
            .expect("legacy read");

    let expected_direction = canonical_direction().to_row_major();
    for (name, spacing, direction, voxels) in [
        (
            "series-module loader",
            native.spacing().to_array(),
            native.direction().to_row_major(),
            native.data_slice().expect("native contiguous"),
        ),
        (
            "reader-module loader",
            legacy.spacing().to_array(),
            legacy.direction().to_row_major(),
            legacy.data_slice().expect("legacy contiguous"),
        ),
    ] {
        assert_eq!(
            spacing, SPACING,
            "{name}: spacing must be [Δdepth, Δrow, Δcol]"
        );
        assert_eq!(
            direction, expected_direction,
            "{name}: direction columns must be [depth, row, col]"
        );
        assert_voxels_within_tolerance(&data, voxels, name);
    }
    assert_eq!(
        legacy_meta.spacing, SPACING,
        "reader metadata must return [Δdepth, Δrow, Δcol]"
    );
}

/// The substrate-free writer must not transpose `Spacing<3>` or the direction.
///
/// Same oracle as above, through the Coeus-carrier entry point.
#[test]
fn coeus_series_writer_preserves_canonical_axis_order() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("series_axis_order_coeus");

    let device = SequentialBackend;
    let tensor =
        ritk_image::tensor::Tensor::<f32, SequentialBackend>::from_slice_on(DIMS, &ramp(), &device);
    let image = ritk_image::Image::<f32, SequentialBackend, 3>::new(
        tensor,
        Point::new(ORIGIN),
        Spacing::new(SPACING),
        canonical_direction(),
    )
    .expect("coeus series image construction");

    write_dicom_series(&dir, &image).expect("coeus write");
    let (loaded, meta) = load_dicom_series_with_metadata::<SequentialBackend, _>(&dir, &device)
        .expect("legacy read");

    assert_eq!(
        meta.spacing, SPACING,
        "metadata must keep [Δdepth, Δrow, Δcol]"
    );
    assert_eq!(
        loaded.spacing().to_array(),
        SPACING,
        "Coeus series writer must not transpose Spacing<3>"
    );
    assert_eq!(
        loaded.direction().to_row_major(),
        canonical_direction().to_row_major(),
        "Coeus series writer must not transpose the direction"
    );
}
