//! Pixelwise three-image arithmetic filters.
//!
//! Each filter combines three co-registered images of identical shape via a
//! pointwise ternary operation applied independently to every voxel.
//!
//! - `TernaryAddImageFilter`: `out(x) = a(x) + b(x) + c(x)`
//! - `TernaryMagnitudeImageFilter`: `out(x) = √(a² + b² + c²)`
//! - `TernaryMagnitudeSquaredImageFilter`: `out(x) = a² + b² + c²`
//!
//! Spatial metadata is taken from the first input. A shape mismatch returns
//! `Err`. One generic [`TernaryOpFilter<Op>`] over a [`TernaryOp`] ZST covers
//! every variant (cf. [`super::binary_ops::BinaryOpFilter`]).
//!
//! # ITK / SimpleITK Parity
//! `TernaryAddImageFilter`, `TernaryMagnitudeImageFilter`,
//! `TernaryMagnitudeSquaredImageFilter` (`sitk.TernaryAdd`,
//! `sitk.TernaryMagnitude`, `sitk.TernaryMagnitudeSquared`).

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec as extract, rebuild};

/// Pointwise ternary operation on a voxel triple (zero-sized strategy type).
pub trait TernaryOp: Default {
    /// Apply the operation to one voxel triple.
    fn apply(a: f32, b: f32, c: f32) -> f32;
}

/// Sum: `a + b + c`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TernaryAddOp;

/// Magnitude: `√(a² + b² + c²)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TernaryMagnitudeOp;

/// Squared magnitude: `a² + b² + c²`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TernaryMagnitudeSquaredOp;

impl TernaryOp for TernaryAddOp {
    #[inline]
    fn apply(a: f32, b: f32, c: f32) -> f32 {
        a + b + c
    }
}

impl TernaryOp for TernaryMagnitudeOp {
    #[inline]
    fn apply(a: f32, b: f32, c: f32) -> f32 {
        (a * a + b * b + c * c).sqrt()
    }
}

impl TernaryOp for TernaryMagnitudeSquaredOp {
    #[inline]
    fn apply(a: f32, b: f32, c: f32) -> f32 {
        a * a + b * b + c * c
    }
}

/// Generic pixelwise three-image filter parameterised by a [`TernaryOp`] ZST.
#[derive(Debug, Clone, Default)]
pub struct TernaryOpFilter<Op: TernaryOp> {
    _op: core::marker::PhantomData<Op>,
}

impl<Op: TernaryOp> TernaryOpFilter<Op> {
    /// Create a new filter.
    pub fn new() -> Self {
        Self {
            _op: core::marker::PhantomData,
        }
    }

    /// Apply the ternary operation to three co-registered images.
    pub fn apply<B: Backend>(
        &self,
        a: &Image<B, 3>,
        b: &Image<B, 3>,
        c: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let (sa, sb, sc) = (a.shape(), b.shape(), c.shape());
        anyhow::ensure!(
            sa == sb && sb == sc,
            "ternary image filter: shape mismatch {:?} / {:?} / {:?}",
            sa,
            sb,
            sc
        );
        let (av, dims) = extract(a)?;
        let (bv, _) = extract(b)?;
        let (cv, _) = extract(c)?;
        let out: Vec<f32> = av
            .iter()
            .zip(bv.iter())
            .zip(cv.iter())
            .map(|((&x, &y), &z)| Op::apply(x, y, z))
            .collect();
        Ok(rebuild(out, dims, a))
    }
}

/// Pixelwise sum of three images. ITK Parity: `TernaryAddImageFilter`.
pub type TernaryAddImageFilter = TernaryOpFilter<TernaryAddOp>;
/// Pixelwise magnitude of three images. ITK Parity: `TernaryMagnitudeImageFilter`.
pub type TernaryMagnitudeImageFilter = TernaryOpFilter<TernaryMagnitudeOp>;
/// Pixelwise squared magnitude of three images. ITK Parity: `TernaryMagnitudeSquaredImageFilter`.
pub type TernaryMagnitudeSquaredImageFilter = TernaryOpFilter<TernaryMagnitudeSquaredOp>;

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_image::test_support as ts;

    type B = burn_ndarray::NdArray<f32>;

    fn img(v: Vec<f32>) -> Image<B, 3> {
        ts::make_image::<B, 3>(v, [1, 1, 3])
    }

    #[test]
    fn ternary_add_sums_three() {
        let out = TernaryAddImageFilter::new()
            .apply(
                &img(vec![1.0, 2.0, 3.0]),
                &img(vec![10.0, 20.0, 30.0]),
                &img(vec![100.0, 200.0, 300.0]),
            )
            .unwrap();
        assert_eq!(out.data_slice().into_owned(), vec![111.0, 222.0, 333.0]);
    }

    #[test]
    fn ternary_magnitude_and_squared() {
        let a = img(vec![2.0, 1.0, 0.0]);
        let b = img(vec![3.0, 2.0, 0.0]);
        let c = img(vec![6.0, 2.0, 5.0]);
        let m = TernaryMagnitudeImageFilter::new()
            .apply(&a, &b, &c)
            .unwrap();
        let s = TernaryMagnitudeSquaredImageFilter::new()
            .apply(&a, &b, &c)
            .unwrap();
        // 2-3-6 → 7; 1-2-2 → 3; 0-0-5 → 5
        for (got, exp) in m.data_slice().into_owned().iter().zip([7.0f32, 3.0, 5.0]) {
            assert!(
                (got - exp).abs() < 1e-5,
                "magnitude: got {got}, expected {exp}"
            );
        }
        for (got, exp) in s.data_slice().into_owned().iter().zip([49.0f32, 9.0, 25.0]) {
            assert!(
                (got - exp).abs() < 1e-5,
                "magnitude²: got {got}, expected {exp}"
            );
        }
    }
}
