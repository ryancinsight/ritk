//! Parity: the Coeus-native NGF engine reproduces the Burn engine.
//!
//! Oracles:
//! - Analytical: NGF of a constant volume is 0 (every gradient vanishes, so the
//!   squared normalized dot product is 0 at every voxel — the `η` clamp keeps
//!   the denominator finite).
//! - Differential: `ngf_value_native` reproduces the established Burn
//!   `NgfFixedPrep` on the same data and transform. Under identity/integer
//!   translation the moving resample lands on exact voxel coordinates, so both
//!   substrates return bit-identical host values (tol 1e-5, absorbing only the
//!   index↔world round-trip's `f32` narrowing). Under a fractional translation
//!   the two trilinear kernels differ only by `f32` rounding of the 8-neighbour
//!   lerp cascade: per-sample error ≈ 8·ε_f32 ≈ 5e-7, amplified ~2× by the
//!   finite-difference moving gradient and carried into an O(1) NGF ratio, so
//!   the averaged metric agrees to < 1e-4 (conservative over ε_f32 ≈ 6e-8).

use super::super::fixed_prep::NgfFixedPrep;
use super::{ngf_value_native, NgfFixedPrepNative};

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
/// gradient fields are non-degenerate (distinct per axis) — a constant or
/// separable field would hide column/axis mix-ups the parity test must catch.
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

/// Burn identity `t = 0` translation.
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

/// Analytical oracle: NGF of a constant volume is exactly 0.
#[test]
fn native_ngf_constant_is_zero() {
    let shape = [4usize, 5, 6];
    let n: usize = shape.iter().product();
    let img = native_image(vec![0.7f32; n], shape);
    let ngf = ngf_value_native(&img, &img, &native_translation([0.0, 0.0, 0.0]), None, None);
    assert!(ngf.abs() < 1e-6, "constant-volume NGF must be 0, got {ngf}");
}

/// Differential parity under the identity transform: the native engine
/// reproduces the Burn engine bit-faithfully (exact voxel resample).
#[test]
fn native_matches_burn_identity() {
    let shape = [4usize, 5, 6];
    let data = ramp(shape[0], shape[1], shape[2]);

    let burn = NgfFixedPrep::<BB, 3>::new(&burn_image(data.clone(), shape), None, None).eval(
        &burn_image(data.clone(), shape),
        &burn_translation([0.0, 0.0, 0.0]),
    );
    let native = NgfFixedPrepNative::<NB>::new(&native_image(data.clone(), shape), None, None)
        .eval(
            &native_image(data, shape),
            &native_translation([0.0, 0.0, 0.0]),
        );

    assert!(burn > 0.0, "self-NGF should be positive, got {burn}");
    assert!(
        (burn - native).abs() < 1e-5,
        "identity NGF divergence: burn {burn} vs native {native}"
    );
}

/// Differential parity under an integer translation (still an exact voxel
/// resample on both substrates).
#[test]
fn native_matches_burn_integer_shift() {
    let shape = [5usize, 6, 7];
    let fixed = ramp(shape[0], shape[1], shape[2]);
    // Moving = fixed shifted, so the transformed sample lands on real structure.
    let mut moving = vec![0.0f32; fixed.len()];
    for (i, m) in moving.iter_mut().enumerate() {
        *m = fixed[fixed.len() - 1 - i];
    }
    let t = [0.0, 1.0, 2.0]; // world axis-major translation, integer components

    let burn = NgfFixedPrep::<BB, 3>::new(&burn_image(fixed.clone(), shape), None, None)
        .eval(&burn_image(moving.clone(), shape), &burn_translation(t));
    let native = NgfFixedPrepNative::<NB>::new(&native_image(fixed.clone(), shape), None, None)
        .eval(&native_image(moving, shape), &native_translation(t));

    assert!(
        (burn - native).abs() < 1e-5,
        "integer-shift NGF divergence: burn {burn} vs native {native}"
    );
}

/// Differential parity under a fractional translation (trilinear interpolation
/// exercised), over an interior mask so every sampled neighbour is in-bounds and
/// the two kernels' out-of-bounds policies are not compared.
#[test]
fn native_matches_burn_fractional_shift() {
    let shape = [6usize, 6, 6];
    let fixed = ramp(shape[0], shape[1], shape[2]);
    let mut moving = vec![0.0f32; fixed.len()];
    for (i, m) in moving.iter_mut().enumerate() {
        *m = 0.5 * fixed[i] + 0.5 * fixed[fixed.len() - 1 - i];
    }
    // Interior mask: exclude the 1-voxel border so a ±0.5 shift stays in-bounds.
    let [d, h, w] = shape;
    let mut mask = vec![false; d * h * w];
    for z in 1..d - 1 {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                mask[(z * h + y) * w + x] = true;
            }
        }
    }
    let t = [0.5, 0.5, 0.5];

    let burn = NgfFixedPrep::<BB, 3>::new(&burn_image(fixed.clone(), shape), Some(&mask), None)
        .eval(&burn_image(moving.clone(), shape), &burn_translation(t));
    let native =
        NgfFixedPrepNative::<NB>::new(&native_image(fixed.clone(), shape), Some(&mask), None)
            .eval(&native_image(moving, shape), &native_translation(t));

    assert!(
        burn > 0.0,
        "masked self-ish NGF should be positive, got {burn}"
    );
    assert!(
        (burn - native).abs() < 1e-4,
        "fractional-shift NGF divergence: burn {burn} vs native {native}"
    );
}
