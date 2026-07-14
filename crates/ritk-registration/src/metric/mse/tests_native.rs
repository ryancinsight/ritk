//! Parity: the Coeus-native MSE engine reproduces the Burn engine.
//!
//! Scope: these tests verify the MSE *reduction* over the resampled volume. The
//! resample path itself (fixed grid → native batch transforms → native affine →
//! native trilinear) is the shared [`super::super::native_resample`] substrate,
//! already differentially verified against Burn under identity/integer/fractional
//! translation by the native NGF parity suite — so it is not re-verified here.
//! MSE is intentionally full-volume (no mask, matching [`super::MeanSquaredError`]),
//! so a translated test would compare the two kernels' *out-of-bounds* policies
//! (which differ — hence the NGF suite masks the interior); the differential
//! oracle below therefore uses only the identity transform, an exact voxel
//! resample with no out-of-bounds sampling.
//!
//! Oracles:
//! - Analytical: MSE of identical images under identity is 0 (every squared
//!   difference vanishes).
//! - Differential: `mse_value_native` reproduces Burn `MeanSquaredError::forward`
//!   on reversed-image data under identity (both read bit-identical on-grid host
//!   values; the two mean-squared reductions agree to the index↔world `f32`
//!   round-trip, tol 1e-4 over an O(1) MSE magnitude).

use super::super::super::trait_::Metric;
use super::super::MeanSquaredError;
use super::mse_value_native;

use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image as BurnImage;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;
use ritk_transform::TranslationTransform;

type BB = NdArray<f32>;
type NB = SequentialBackend;

/// Smooth, anisotropic-gradient volume of `[d, h, w]` so the fixed and moving
/// intensities are non-degenerate (distinct per axis) — a constant or separable
/// field would hide column/axis mix-ups the parity test must catch.
fn ramp(d: usize, h: usize, w: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; d * h * w];
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let (zf, yf, xf) = (z as f32, y as f32, x as f32);
                v[(z * h + y) * w + x] =
                    (0.30 * xf + 0.17 * yf * yf + 0.09 * zf + 0.05 * xf * yf).sin() + 0.2 * zf;
            }
        }
    }
    v
}

fn burn_image(data: Vec<f32>, shape: [usize; 3]) -> BurnImage<BB, 3> {
    let device = Default::default();
    BurnImage::new(
        Tensor::from_data(TensorData::new(data, Shape::new(shape)), &device),
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn native_image(data: Vec<f32>, shape: [usize; 3]) -> NativeImage<f32, NB, 3> {
    NativeImage::from_flat(
        data,
        shape,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .expect("native image from flat")
}

fn burn_translation(t: [f32; 3]) -> TranslationTransform<BB, 3> {
    let device = Default::default();
    TranslationTransform::<BB, 3>::new(Tensor::from_data(TensorData::new(t.to_vec(), [3]), &device))
}

/// Native affine = identity matrix + `t` translation (center 0), so
/// `T(x) = x + t` — the exact semantics of [`TranslationTransform`].
fn native_translation(t: [f32; 3]) -> AtlasAffineTransform<NB, 3> {
    let matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    AtlasAffineTransform::<NB, 3>::construct(&matrix, &t, &[0.0, 0.0, 0.0])
}

/// Analytical oracle: MSE of identical images under the identity transform is 0.
#[test]
fn native_mse_identical_is_zero() {
    let shape = [4usize, 5, 6];
    let data = ramp(shape[0], shape[1], shape[2]);
    let img = native_image(data.clone(), shape);
    let mse = mse_value_native(&img, &img, &native_translation([0.0, 0.0, 0.0]));
    assert!(mse.abs() < 1e-6, "identical-image MSE must be 0, got {mse}");
}

/// Differential parity under the identity transform (exact voxel resample, no
/// out-of-bounds sampling): the native reduction reproduces the Burn engine.
#[test]
fn native_matches_burn_identity() {
    let shape = [4usize, 5, 6];
    let fixed = ramp(shape[0], shape[1], shape[2]);
    // Moving = reversed fixed so the MSE is non-trivial (positive), not 0.
    let moving: Vec<f32> = fixed.iter().rev().copied().collect();

    let burn = MeanSquaredError::new()
        .forward(
            &burn_image(fixed.clone(), shape),
            &burn_image(moving.clone(), shape),
            &burn_translation([0.0, 0.0, 0.0]),
        )
        .into_scalar();
    let native = mse_value_native(
        &native_image(fixed, shape),
        &native_image(moving, shape),
        &native_translation([0.0, 0.0, 0.0]),
    );

    assert!(
        burn > 0.0,
        "reversed-image MSE should be positive, got {burn}"
    );
    assert!(
        (burn - native).abs() < 1e-4,
        "identity MSE divergence: burn {burn} vs native {native}"
    );
}
