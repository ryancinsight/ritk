//! Differential parity and round-trip verification for the native series
//! writer [`write_dicom_series_native`].
//!
//! Burn is retained as a dev-only oracle: these tests assert the native series
//! writer emits pixel-and-geometry-identical output to the Burn
//! [`write_dicom_series`] for identical voxels (both route through the shared
//! substrate-free `write_series_flat` core), and that a native-written series
//! round-trips through the native reader to the same voxels and geometry.

use crate::format::dicom::read_native_dicom_series;
use crate::format::dicom::writer::{write_dicom_series, write_dicom_series_native};
use coeus_core::SequentialBackend;
use coeus_core::{MoiraiBackend, SequentialBackend};
use ritk_core::image::Image as BurnImage;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::backend::Backend as BurnBackend;
use coeus_tensor::Tensor;
use ritk_image::tensor::{Shape, TensorData};
use ritk_spatial::{Direction, Point, Spacing};

type OracleBackend = SequentialBackend;

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

fn burn_image(data: Vec<f32>) -> BurnImage<OracleBackend, 3> {
    let device = <OracleBackend as BurnBackend>::Device::default();
    let tensor =
        Tensor::<OracleBackend, 3>::from_data((data, (DIMS)), &device);
    BurnImage::new(
        tensor,
        Point::new(ORIGIN),
        Spacing::new(SPACING),
        Direction::identity(),
    )
}

/// The native and Burn series writers must encode identical pixel data and
/// geometry: both re-read through the native reader agree bit-for-bit on
/// voxels and to floating-point tolerance on origin/spacing/direction.
///
/// Per-call random UIDs preclude full-file byte equality, so parity is asserted
/// on the substrate-invariant decoded content — the encode contract.
#[test]
fn native_series_writer_matches_burn_series_writer() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let native_dir = tmp.path().join("series_native");
    let burn_dir = tmp.path().join("series_burn");

    let data = ramp();
    write_dicom_series_native(&native_dir, &native_image(data.clone())).expect("native write");
    write_dicom_series(&burn_dir, &burn_image(data)).expect("burn write");

    let native = read_native_dicom_series(&native_dir, &SequentialBackend).expect("native read");
    let burn_reload =
        read_native_dicom_series(&burn_dir, &SequentialBackend).expect("burn-file read");

    assert_eq!(native.shape(), burn_reload.shape(), "shape parity");
    assert_eq!(
        native.data_slice().expect("native contiguous"),
        burn_reload.data_slice().expect("burn-file contiguous"),
        "native and burn series writers must encode identical pixel data"
    );
    assert_eq!(
        native.origin().to_array(),
        burn_reload.origin().to_array(),
        "origin parity"
    );
    assert_eq!(
        native.spacing().to_array(),
        burn_reload.spacing().to_array(),
        "spacing parity"
    );
    assert_eq!(
        native.direction().to_row_major(),
        burn_reload.direction().to_row_major(),
        "direction parity"
    );
}

/// A native-written series round-trips through the native reader to the same
/// voxels (within the per-slice rescale bound) and geometry.
///
/// Per-slice `normalize_to_u16` reconstruction bound: for a slice of range `R`,
/// `slope = R / 65535`; DS `{:.6}` formatting adds ≤ 0.5e-6 per coefficient, and
/// quantization adds ≤ slope/2. The linear ramp gives every slice the same
/// range `R = (slice_len − 1) · 1.5`.
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
