//! Parity: the `AtlasAffineTransform` specialization constructors reproduce the
//! Burn `TranslationTransform` / `ScaleTransform` / `RigidTransform` /
//! `VersorRigid3DTransform` `transform_points` results.
//!
//! Oracle: differential vs the established Burn specializations on sample points.
//! All evaluate `T(x) = A(x − c) + c + t`; the native constructors build the
//! rotation matrix in the same host math (f32 Euler trig, f64→f32 quaternion) as
//! the Burn builders, so results agree to f32 rounding of the matvec (tol 1e-5).

use super::super::rigid::RigidTransform;
use super::super::scale::ScaleTransform;
use super::super::translation::TranslationTransform;
use super::super::versor::VersorRigid3DTransform;
use super::AtlasAffineTransform;

use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_core::transform::Transform;
use ritk_image::native::Image;
use ritk_image::tensor::Tensor;
use ritk_spatial::{Direction, Point, Spacing};

type BB = NdArray<f32>;
type NB = SequentialBackend;

const TOL: f32 = 1e-5;

/// `[N, 3]` flat, a spread of non-symmetric points so an axis/column mix-up in
/// either path is caught.
fn sample_points() -> Vec<f32> {
    vec![
        0.0, 0.0, 0.0, //
        1.0, 2.0, 3.0, //
        -2.0, 0.5, 4.0, //
        3.0, -1.0, 2.0,
    ]
}

fn burn_points(flat: &[f32], n: usize) -> Tensor<BB, 2> {
    let device = Default::default();
    Tensor::<BB, 1>::from_floats(flat, &device).reshape([n, 3])
}

fn native_points(flat: &[f32], n: usize) -> Image<f32, NB, 2> {
    Image::from_flat(
        flat.to_vec(),
        [n, 3],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .expect("native points image")
}

fn burn_vec1<const N: usize>(v: [f32; N]) -> Tensor<BB, 1> {
    let device = Default::default();
    Tensor::<BB, 1>::from_floats(v, &device)
}

fn assert_parity(burn: Tensor<BB, 2>, native: Vec<f32>, label: &str) {
    let bd = burn.into_data();
    let bs = bd.as_slice::<f32>().unwrap();
    assert_eq!(bs.len(), native.len(), "{label}: length mismatch");
    for (i, (&b, &n)) in bs.iter().zip(native.iter()).enumerate() {
        assert!(
            (b - n).abs() < TOL,
            "{label}: idx {i} burn {b} vs native {n}"
        );
    }
}

#[test]
fn translation_parity() {
    let flat = sample_points();
    let n = flat.len() / 3;
    let t = [1.5, -2.0, 0.5];
    let burn =
        TranslationTransform::<BB, 3>::new(burn_vec1(t)).transform_points(burn_points(&flat, n));
    let native = AtlasAffineTransform::<NB, 3>::from_translation(&t)
        .transform_points(&native_points(&flat, n))
        .unwrap()
        .data_vec();
    assert_parity(burn, native, "translation");
}

#[test]
fn scale_parity() {
    let flat = sample_points();
    let n = flat.len() / 3;
    let s = [2.0, 0.5, 1.5];
    let c = [1.0, -1.0, 0.5];
    let burn = ScaleTransform::<BB, 3>::new(burn_vec1(s), burn_vec1(c))
        .transform_points(burn_points(&flat, n));
    let native = AtlasAffineTransform::<NB, 3>::from_scale(&s, &c)
        .transform_points(&native_points(&flat, n))
        .unwrap()
        .data_vec();
    assert_parity(burn, native, "scale");
}

#[test]
fn rigid_parity() {
    let flat = sample_points();
    let n = flat.len() / 3;
    let t = [0.3, -0.7, 1.1];
    let rot = [0.4, -0.2, 0.8]; // Euler α, β, γ (radians)
    let c = [0.5, 0.5, 0.5];
    let burn = RigidTransform::<BB, 3>::new(burn_vec1(t), burn_vec1(rot), burn_vec1(c))
        .transform_points(burn_points(&flat, n));
    let native = AtlasAffineTransform::<NB, 3>::from_euler_rigid(&t, &rot, &c)
        .transform_points(&native_points(&flat, n))
        .unwrap()
        .data_vec();
    assert_parity(burn, native, "rigid");
}

#[test]
fn versor_parity() {
    let flat = sample_points();
    let n = flat.len() / 3;
    let t = [0.2, 1.0, -0.5];
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let quat = [s, 0.0, 0.0, s]; // 90° about X
    let c = [0.5, 0.5, 0.5];
    let burn = VersorRigid3DTransform::<BB>::new(burn_vec1(t), burn_vec1(quat), burn_vec1(c))
        .transform_points(burn_points(&flat, n));
    let native = AtlasAffineTransform::<NB, 3>::from_versor(&t, &quat, &c)
        .transform_points(&native_points(&flat, n))
        .unwrap()
        .data_vec();
    assert_parity(burn, native, "versor");
}
