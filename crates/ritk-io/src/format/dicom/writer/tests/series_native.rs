//! Differential parity and round-trip verification for the native series
//! writer [`write_dicom_series_native`].
//!
//! Native series writer parity and round-trip verification.

use crate::format::dicom::read_native_dicom_series;
use crate::format::dicom::writer::write_dicom_series_native;
use coeus_core::{MoiraiBackend, SequentialBackend};
use eunomia::FloatElement;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

const DIMS: [usize; 3] = [3, 4, 5];
const ORIGIN: [f64; 3] = [1.0, -2.0, 3.0];
const SPACING: [f64; 3] = [0.75, 0.5, 2.5];

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
        Direction::identity(),
    )
    .expect("native series image construction")
}

#[test]
fn native_series_writer_round_trips_native_image() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let native_dir = tmp.path().join("series_native");

    let data = ramp();
    write_dicom_series_native(&native_dir, &native_image(data.clone())).expect("native write");
    let native = read_native_dicom_series(&native_dir, &SequentialBackend).expect("native read");

    assert_eq!(native.shape(), DIMS, "shape parity");
    let slice_len = DIMS[1] * DIMS[2];
    let slice_range = (f32::from_count(slice_len) - 1.0) * 1.5;
    let slope = slice_range / 65535.0_f32;
    let ds_half_ulp = 0.5e-6_f32;
    let tol = 65535.0_f32 * ds_half_ulp + ds_half_ulp + slope / 2.0_f32;
    for (idx, (&orig, &got)) in data
        .iter()
        .zip(native.data_slice().expect("native contiguous").iter())
        .enumerate()
    {
        let err = (got - orig).abs();
        assert!(
            err <= tol,
            "voxel[{idx}]: |{got} - {orig}| = {err} > tol {tol}"
        );
    }
    assert_eq!(native.origin().to_array(), ORIGIN, "origin parity");
    assert_eq!(native.spacing().to_array(), SPACING, "spacing parity");
    assert_eq!(
        native.direction().to_row_major(),
        Direction::<3>::identity().to_row_major(),
        "direction parity"
    );
}

/// A native-written series round-trips through the native reader to the same
/// voxels (within the per-slice rescale bound) and geometry.
///
/// For range `R`, the stored slope is `R / 65535`; integer rounding contributes
/// at most half a slope step. Four f32 operations in normalization contribute
/// at most `γ₄·65535` codes, where `γₙ = n·u/(1−n·u)` and `u = ε/2`. DICOM DS
/// carries at least nine significant digits in 16 bytes; decimal and f32
/// rescale parsing contribute at most `(5e-9 + u)(R + max|value|)`. The reader's
/// multiply and add contribute `γ₂(R + max|value|)`. The fixture's linear ramp
/// gives every slice the same `R = (slice_len − 1)·1.5`.
#[test]
fn native_series_writer_round_trips_through_native_reader() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("series_rt");

    let data = ramp();
    write_dicom_series_native(&dir, &native_image(data.clone())).expect("native write");

    let reloaded = read_native_dicom_series(&dir, &SequentialBackend).expect("native read");
    assert_eq!(reloaded.shape(), DIMS, "shape must round-trip");

    let slice_len = DIMS[1] * DIMS[2];
    let slice_range = (f32::from_count(slice_len) - 1.0) * 1.5;
    let slope = slice_range / 65535.0_f32;
    let unit_roundoff = f32::EPSILON / 2.0_f32;
    let gamma_four = 4.0_f32 * unit_roundoff / (1.0_f32 - 4.0_f32 * unit_roundoff);
    let gamma_two = 2.0_f32 * unit_roundoff / (1.0_f32 - 2.0_f32 * unit_roundoff);
    let max_absolute_value = data.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let rescale_bound = slice_range + max_absolute_value;
    let tol = slope * (0.5_f32 + gamma_four * 65535.0_f32)
        + (5.0e-9_f32 + unit_roundoff) * rescale_bound
        + gamma_two * rescale_bound;

    let recovered = reloaded.data_slice().expect("contiguous reloaded data");
    assert_eq!(recovered.len(), data.len(), "voxel count must round-trip");
    for (idx, (&orig, &got)) in data.iter().zip(recovered.iter()).enumerate() {
        let err = (got - orig).abs();
        assert!(
            err <= tol,
            "voxel[{idx}]: |{got} - {orig}| = {err} > tol {tol}"
        );
    }

    // Every geometry value in this fixture has an exact DS representation.
    for k in 0..3 {
        assert_eq!(reloaded.origin().to_array()[k], ORIGIN[k]);
        assert_eq!(reloaded.spacing().to_array()[k], SPACING[k]);
    }
    let expected_dir = Direction::<3>::identity().to_row_major();
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
