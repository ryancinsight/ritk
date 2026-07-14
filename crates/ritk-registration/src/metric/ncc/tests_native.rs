//! Parity: the Coeus-native NCC engine reproduces the Burn engine.
//!
//! Scope: these tests verify the five-moment ZNCC *reduction*. The resample path
//! (fixed grid → native batch transforms → native affine → native trilinear) is
//! the shared [`super::super::native_resample`] substrate, already differentially
//! verified against Burn by the native NGF suite. NCC is global (whole-volume),
//! so the differential oracle uses only the identity transform (exact voxel
//! resample, no out-of-bounds sampling whose policy differs between kernels).
//!
//! Oracles:
//! - Analytical: `NCC(F, F) = 1`, so the loss `−NCC` of an image with itself is
//!   `−1` (perfect linear correlation; the variance clamp keeps the denominator
//!   finite).
//! - Differential: `ncc_loss_native` reproduces Burn
//!   `NormalizedCrossCorrelation::forward` on reversed-image data under identity
//!   (both read bit-identical on-grid host values; the two moment reductions
//!   agree to the index↔world `f32` round-trip and summation-order rounding of an
//!   O(1) correlation, tol 1e-4).

use super::super::super::trait_::Metric;
use super::super::NormalizedCrossCorrelation;
use super::ncc_loss_native;

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

/// Smooth, anisotropic-gradient volume so the intensities are non-degenerate
/// (distinct per axis) — a constant field has zero variance and undefined NCC.
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

fn native_translation(t: [f32; 3]) -> AtlasAffineTransform<NB, 3> {
    let matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    AtlasAffineTransform::<NB, 3>::construct(&matrix, &t, &[0.0, 0.0, 0.0])
}

/// Analytical oracle: NCC of an image with itself is 1, so the loss is −1.
#[test]
fn native_ncc_self_is_minus_one() {
    let shape = [4usize, 5, 6];
    let data = ramp(shape[0], shape[1], shape[2]);
    let img = native_image(data, shape);
    let loss = ncc_loss_native(&img, &img, &native_translation([0.0, 0.0, 0.0]));
    assert!(
        (loss + 1.0).abs() < 1e-5,
        "self-NCC loss must be −1, got {loss}"
    );
}

/// Differential parity under the identity transform (exact voxel resample).
#[test]
fn native_matches_burn_identity() {
    let shape = [4usize, 5, 6];
    let fixed = ramp(shape[0], shape[1], shape[2]);
    // Moving = reversed fixed → a non-trivial (imperfect) correlation.
    let moving: Vec<f32> = fixed.iter().rev().copied().collect();

    let burn = NormalizedCrossCorrelation::new()
        .forward(
            &burn_image(fixed.clone(), shape),
            &burn_image(moving.clone(), shape),
            &burn_translation([0.0, 0.0, 0.0]),
        )
        .into_scalar();
    let native = ncc_loss_native(
        &native_image(fixed, shape),
        &native_image(moving, shape),
        &native_translation([0.0, 0.0, 0.0]),
    );

    assert!(
        (burn - native).abs() < 1e-4,
        "identity NCC-loss divergence: burn {burn} vs native {native}"
    );
}
