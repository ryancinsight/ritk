//! Differential parity and round-trip verification for the native series
//! writer [`write_dicom_series_native`].
//!
//! Native series writer parity and round-trip verification.

use crate::format::dicom::read_native_dicom_series;
use crate::format::dicom::writer::write_dicom_series_native;
use coeus_core::{MoiraiBackend, SequentialBackend};
use dicom::core::Tag;
use dicom::object::open_file;
use ritk_image::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};

const DIMS: [usize; 3] = [3, 4, 5];
const ORIGIN: [f64; 3] = [1.0, -2.0, 3.0];
const SPACING: [f64; 3] = [0.75, 0.5, 2.5];

fn ramp() -> Vec<f32> {
    let n = DIMS[0] * DIMS[1] * DIMS[2];
    (0..n).map(|i| i as f32 * 1.5 - 37.0).collect()
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
    let slice_range = (slice_len as f32 - 1.0) * 1.5;
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

/// A small, representable image range survives DICOM Decimal String encoding.
///
/// The two input endpoints quantize exactly to 0 and 65535, so only the
/// `RescaleSlope` path contributes reconstruction error. At this magnitude,
/// the 16-byte Decimal String retains at least seven significant digits.
/// Its relative rounding bound is therefore 0.5×10⁻⁶. Three f32 roundings
/// cover slope division, Decimal String parsing, and sample multiplication;
/// the composed factor bounds the restored endpoint error.
#[test]
fn native_series_writer_preserves_small_range_calibration() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("small_range");
    let span = 1.0e-8_f32;
    let slice_len = DIMS[1] * DIMS[2];
    let data: Vec<f32> = (0..DIMS[0] * slice_len)
        .map(|index| {
            if (index % slice_len).is_multiple_of(2) {
                0.0
            } else {
                span
            }
        })
        .collect();

    write_dicom_series_native(&dir, &native_image(data.clone())).expect("native write");

    let first_slice = open_file(dir.join("slice_0000.dcm")).expect("open first slice");
    let encoded_slope = first_slice
        .element(Tag(0x0028, 0x1053))
        .expect("RescaleSlope (0028,1053) is present")
        .to_str()
        .expect("RescaleSlope is readable")
        .trim()
        .parse::<f32>()
        .expect("RescaleSlope is numeric");
    let expected_slope = span / f32::from(u16::MAX);
    assert!(encoded_slope.is_finite() && encoded_slope > 0.0);
    assert_eq!(encoded_slope, expected_slope);

    let reloaded = read_native_dicom_series(&dir, &SequentialBackend).expect("native read");
    assert_eq!(reloaded.shape(), DIMS);
    let recovered = reloaded.data_slice().expect("contiguous reloaded data");
    assert_eq!(recovered.len(), data.len());

    let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
    let decimal_relative_bound = 0.5_f64 / 1_000_000.0;
    let relative_error_bound = (1.0 + unit_roundoff).powi(3) * (1.0 + decimal_relative_bound) - 1.0;
    let error_bound = f64::from(span) * relative_error_bound;
    for (index, (&expected, &actual)) in data.iter().zip(recovered).enumerate() {
        let error = (f64::from(actual) - f64::from(expected)).abs();
        assert!(
            error <= error_bound,
            "voxel[{index}]: |{actual} - {expected}| = {error} > derived bound {error_bound}"
        );
    }
}

/// A native-written series round-trips through the native reader to the same
/// voxels (within the per-slice rescale bound) and geometry.
///
/// Per-slice `normalize_samples` reconstruction bound: for a slice of range
/// `R`, `slope = R / 65535`, and integer quantization adds at most `slope/2`.
/// The Decimal String formatter preserves values that fit the 16-byte limit
/// and uses bounded scientific notation otherwise. The linear ramp gives every
/// slice the same range `R = (slice_len − 1) · 1.5`.
#[test]
fn native_series_writer_round_trips_through_native_reader() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().join("series_rt");

    let data = ramp();
    write_dicom_series_native(&dir, &native_image(data.clone())).expect("native write");

    let reloaded = read_native_dicom_series(&dir, &SequentialBackend).expect("native read");
    assert_eq!(reloaded.shape(), DIMS, "shape must round-trip");

    let slice_len = DIMS[1] * DIMS[2];
    let slice_range = (slice_len as f32 - 1.0) * 1.5;
    let slope = slice_range / 65535.0_f32;
    let ds_half_ulp = 0.5e-6_f32;
    let tol = 65535.0_f32 * ds_half_ulp + ds_half_ulp + slope / 2.0_f32;

    let recovered = reloaded.data_slice().expect("contiguous reloaded data");
    assert_eq!(recovered.len(), data.len(), "voxel count must round-trip");
    for (idx, (&orig, &got)) in data.iter().zip(recovered.iter()).enumerate() {
        let err = (got - orig).abs();
        assert!(
            err <= tol,
            "voxel[{idx}]: |{got} - {orig}| = {err} > tol {tol}"
        );
    }

    // Geometry round-trips exactly to DS {:.6} precision.
    let geom_tol = 1e-4;
    for k in 0..3 {
        assert!(
            (reloaded.origin().to_array()[k] - ORIGIN[k]).abs() <= geom_tol,
            "origin[{k}] must round-trip"
        );
        assert!(
            (reloaded.spacing().to_array()[k] - SPACING[k]).abs() <= geom_tol,
            "spacing[{k}] must round-trip"
        );
    }
    let expected_dir = Direction::<3>::identity().to_row_major();
    for (k, (&got, &exp)) in reloaded
        .direction()
        .to_row_major()
        .iter()
        .zip(expected_dir.iter())
        .enumerate()
    {
        assert!(
            (got - exp).abs() <= geom_tol,
            "direction[{k}] must round-trip"
        );
    }
}
