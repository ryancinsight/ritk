//! Parity: the Coeus-native LNCC engine reproduces the Burn engine.
//!
//! Scope: the moving resample is the shared [`super::super::native_resample`]
//! substrate (differentially verified by the native NGF suite) and the local
//! statistics are the shared native separable Gaussian
//! (`GaussianFilter::apply_native`, differentially verified against the Burn
//! `conv1d` path in `ritk-filter`). This suite verifies that the LNCC *reduction*
//! (local covariance over the geometric mean of local variances, negated and
//! averaged) composes those substrates into the same value as the Burn engine.
//! The differential oracle uses the identity transform (exact voxel resample).
//!
//! Oracles:
//! - Self-correlation: identical images are locally perfectly correlated, so the
//!   loss `−mean(LNCC)` is strongly negative (each voxel's `v_F/(v_F+ε) ≈ 1`
//!   wherever local variance ≫ ε).
//! - Differential: `lncc_loss_native` reproduces Burn
//!   `LocalNormalizedCrossCorrelation::forward` on reversed-image data under
//!   identity. Both convolve identical Gaussian kernels at identical spacing;
//!   they differ only by the accumulation order of `conv1d` vs the native
//!   correlation (`O(width·ε·‖I‖∞)` per axis), propagated through the local
//!   ratio and averaged — tol 1e-3 (conservative over ε_f32 ≈ 6e-8).

use super::super::super::trait_::Metric;
use super::super::LocalNormalizedCrossCorrelation;
use super::lncc_loss_native;

use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_filter::GaussianSigma;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image as BurnImage;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;
use ritk_transform::TranslationTransform;

type BB = NdArray<f32>;
type NB = SequentialBackend;

const SIGMA: f64 = 1.5;
const EPS: f32 = 1e-5;

/// Smooth, anisotropic-gradient volume so local variances are non-degenerate
/// per axis (a constant or separable field would hide axis mix-ups).
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

/// Self-correlation oracle: identical images are locally perfectly correlated,
/// so the loss is strongly negative.
#[test]
fn native_lncc_self_is_strongly_negative() {
    let shape = [6usize, 7, 8];
    let data = ramp(shape[0], shape[1], shape[2]);
    let img = native_image(data, shape);
    let loss = lncc_loss_native(
        &img,
        &img,
        &native_translation([0.0, 0.0, 0.0]),
        GaussianSigma::new_unchecked(SIGMA),
        EPS,
    );
    assert!(
        loss < -0.9,
        "identical-image LNCC loss should be ≈ −1, got {loss}"
    );
}

/// Differential parity under the identity transform (exact voxel resample):
/// the native reduction reproduces the Burn engine.
#[test]
fn native_matches_burn_identity() {
    let shape = [6usize, 7, 8];
    let fixed = ramp(shape[0], shape[1], shape[2]);
    // Moving = reversed fixed → a non-trivial local-correlation field.
    let moving: Vec<f32> = fixed.iter().rev().copied().collect();

    let burn = LocalNormalizedCrossCorrelation::<BB>::new(GaussianSigma::new_unchecked(SIGMA))
        .forward(
            &burn_image(fixed.clone(), shape),
            &burn_image(moving.clone(), shape),
            &burn_translation([0.0, 0.0, 0.0]),
        )
        .into_scalar();
    let native = lncc_loss_native(
        &native_image(fixed, shape),
        &native_image(moving, shape),
        &native_translation([0.0, 0.0, 0.0]),
        GaussianSigma::new_unchecked(SIGMA),
        EPS,
    );

    assert!(
        (burn - native).abs() < 1e-3,
        "identity LNCC-loss divergence: burn {burn} vs native {native}"
    );
}
