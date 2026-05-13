//! Pixelwise two-image arithmetic filters.
//!
//! Each filter combines two co-registered images with matching shapes via a
//! pointwise binary operation applied independently to every voxel.
//!
//! # Mathematical Specification
//!
//! Let `A, B : ℤ³ → ℝ` be two images with identical shape `[nz, ny, nx]`:
//!
//! - `AddImageFilter`:      `out(x) = A(x) + B(x)`
//! - `SubtractImageFilter`: `out(x) = A(x) − B(x)`
//! - `MultiplyImageFilter`: `out(x) = A(x) × B(x)`
//! - `DivideImageFilter`:   `out(x) = A(x) / B(x)`  (division by zero yields 0)
//! - `ImageMinFilter`:      `out(x) = min(A(x), B(x))`
//! - `ImageMaxFilter`:      `out(x) = max(A(x), B(x))`
//!
//! Spatial metadata (origin, spacing, direction) is taken from the **first** input image.
//! Both images must have identical shapes; a shape mismatch returns `Err`.
//!
//! # ITK / SimpleITK / ImageJ Parity
//!
//! | Filter                 | ITK class              | ImageJ (Process > Image Calculator) |
//! |------------------------|------------------------|--------------------------------------|
//! | `AddImageFilter`       | `AddImageFilter`       | Add                                  |
//! | `SubtractImageFilter`  | `SubtractImageFilter`  | Subtract                             |
//! | `MultiplyImageFilter`  | `MultiplyImageFilter`  | Multiply                             |
//! | `DivideImageFilter`    | `DivideImageFilter`    | Divide                               |
//! | `ImageMinFilter`       | `MinimumImageFilter`   | Min                                  |
//! | `ImageMaxFilter`       | `MaximumImageFilter`   | Max                                  |

use crate::filter::ops::{extract_vec as extract, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

fn check_shapes(a: [usize; 3], b: [usize; 3]) -> anyhow::Result<()> {
    anyhow::ensure!(
        a == b,
        "binary image filter: shape mismatch {:?} vs {:?}",
        a,
        b
    );
    Ok(())
}

// ── AddImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise addition of two images.
///
/// `out(x) = a(x) + b(x)`
///
/// # ITK Parity: `AddImageFilter`
#[derive(Debug, Clone, Default)]
pub struct AddImageFilter;

impl AddImageFilter {
    pub fn new() -> Self {
        Self
    }

    pub fn apply<B: Backend>(
        &self,
        a: &Image<B, 3>,
        b: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        check_shapes(a.shape(), b.shape())?;
        let (av, dims) = extract(a)?;
        let (bv, _) = extract(b)?;
        let out: Vec<f32> = av.iter().zip(bv.iter()).map(|(&x, &y)| x + y).collect();
        Ok(rebuild(out, dims, a))
    }
}

// ── SubtractImageFilter ───────────────────────────────────────────────────────

/// Pixelwise subtraction of two images.
///
/// `out(x) = a(x) − b(x)`
///
/// # ITK Parity: `SubtractImageFilter`
#[derive(Debug, Clone, Default)]
pub struct SubtractImageFilter;

impl SubtractImageFilter {
    pub fn new() -> Self {
        Self
    }

    pub fn apply<B: Backend>(
        &self,
        a: &Image<B, 3>,
        b: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        check_shapes(a.shape(), b.shape())?;
        let (av, dims) = extract(a)?;
        let (bv, _) = extract(b)?;
        let out: Vec<f32> = av.iter().zip(bv.iter()).map(|(&x, &y)| x - y).collect();
        Ok(rebuild(out, dims, a))
    }
}

// ── MultiplyImageFilter ───────────────────────────────────────────────────────

/// Pixelwise multiplication of two images.
///
/// `out(x) = a(x) × b(x)`
///
/// # ITK Parity: `MultiplyImageFilter`
#[derive(Debug, Clone, Default)]
pub struct MultiplyImageFilter;

impl MultiplyImageFilter {
    pub fn new() -> Self {
        Self
    }

    pub fn apply<B: Backend>(
        &self,
        a: &Image<B, 3>,
        b: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        check_shapes(a.shape(), b.shape())?;
        let (av, dims) = extract(a)?;
        let (bv, _) = extract(b)?;
        let out: Vec<f32> = av.iter().zip(bv.iter()).map(|(&x, &y)| x * y).collect();
        Ok(rebuild(out, dims, a))
    }
}

// ── DivideImageFilter ─────────────────────────────────────────────────────────

/// Pixelwise division of two images; division by zero yields 0.
///
/// `out(x) = a(x) / b(x)`   (returns 0 where `b(x) = 0`)
///
/// # ITK Parity: `DivideImageFilter`
#[derive(Debug, Clone, Default)]
pub struct DivideImageFilter;

impl DivideImageFilter {
    pub fn new() -> Self {
        Self
    }

    pub fn apply<B: Backend>(
        &self,
        a: &Image<B, 3>,
        b: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        check_shapes(a.shape(), b.shape())?;
        let (av, dims) = extract(a)?;
        let (bv, _) = extract(b)?;
        let out: Vec<f32> = av
            .iter()
            .zip(bv.iter())
            .map(|(&x, &y)| if y == 0.0 { 0.0 } else { x / y })
            .collect();
        Ok(rebuild(out, dims, a))
    }
}

// ── ImageMinFilter ────────────────────────────────────────────────────────────

/// Pixelwise minimum of two images.
///
/// `out(x) = min(a(x), b(x))`
///
/// # ITK Parity: `MinimumImageFilter`
#[derive(Debug, Clone, Default)]
pub struct ImageMinFilter;

impl ImageMinFilter {
    pub fn new() -> Self {
        Self
    }

    pub fn apply<B: Backend>(
        &self,
        a: &Image<B, 3>,
        b: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        check_shapes(a.shape(), b.shape())?;
        let (av, dims) = extract(a)?;
        let (bv, _) = extract(b)?;
        let out: Vec<f32> = av.iter().zip(bv.iter()).map(|(&x, &y)| x.min(y)).collect();
        Ok(rebuild(out, dims, a))
    }
}

// ── ImageMaxFilter ────────────────────────────────────────────────────────────

/// Pixelwise maximum of two images.
///
/// `out(x) = max(a(x), b(x))`
///
/// # ITK Parity: `MaximumImageFilter`
#[derive(Debug, Clone, Default)]
pub struct ImageMaxFilter;

impl ImageMaxFilter {
    pub fn new() -> Self {
        Self
    }

    pub fn apply<B: Backend>(
        &self,
        a: &Image<B, 3>,
        b: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        check_shapes(a.shape(), b.shape())?;
        let (av, dims) = extract(a)?;
        let (bv, _) = extract(b)?;
        let out: Vec<f32> = av.iter().zip(bv.iter()).map(|(&x, &y)| x.max(y)).collect();
        Ok(rebuild(out, dims, a))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    // --- AddImageFilter ------------------------------------------------------

    #[test]
    fn add_filter_computes_elementwise_sum() {
        let a = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let b = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
        let out = AddImageFilter::new().apply(&a, &b).unwrap();
        let v = voxels(&out);
        let expected = [11.0f32, 22.0, 33.0, 44.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "[{}] expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    #[test]
    fn add_filter_preserves_spatial_metadata() {
        let a = make_image(vec![1.0; 8], [2, 2, 2]);
        let b = make_image(vec![2.0; 8], [2, 2, 2]);
        let out = AddImageFilter::new().apply(&a, &b).unwrap();
        assert_eq!(out.shape(), a.shape());
        assert_eq!(out.spacing(), a.spacing());
    }

    #[test]
    fn add_filter_shape_mismatch_returns_error() {
        let a = make_image(vec![1.0; 4], [1, 2, 2]);
        let b = make_image(vec![1.0; 8], [2, 2, 2]);
        assert!(AddImageFilter::new().apply(&a, &b).is_err());
    }

    // --- SubtractImageFilter -------------------------------------------------

    #[test]
    fn subtract_filter_computes_elementwise_difference() {
        let a = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
        let b = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let out = SubtractImageFilter::new().apply(&a, &b).unwrap();
        let v = voxels(&out);
        let expected = [9.0f32, 18.0, 27.0, 36.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "[{}] expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    #[test]
    fn subtract_filter_self_minus_self_is_zero() {
        let a = make_image(vec![5.0, 3.0, 7.0, 2.0], [1, 2, 2]);
        let out = SubtractImageFilter::new().apply(&a, &a).unwrap();
        let v = voxels(&out);
        for (i, &val) in v.iter().enumerate() {
            assert!((val - 0.0).abs() < 1e-5, "[{}] expected 0, got {}", i, val);
        }
    }

    // --- MultiplyImageFilter -------------------------------------------------

    #[test]
    fn multiply_filter_computes_elementwise_product() {
        let a = make_image(vec![2.0, 3.0, 4.0, 5.0], [1, 2, 2]);
        let b = make_image(vec![3.0, 4.0, 5.0, 6.0], [1, 2, 2]);
        let out = MultiplyImageFilter::new().apply(&a, &b).unwrap();
        let v = voxels(&out);
        let expected = [6.0f32, 12.0, 20.0, 30.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "[{}] expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    #[test]
    fn multiply_filter_by_zero_image_yields_zeros() {
        let a = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let z = make_image(vec![0.0; 4], [1, 2, 2]);
        let out = MultiplyImageFilter::new().apply(&a, &z).unwrap();
        let v = voxels(&out);
        for &val in &v {
            assert!((val - 0.0).abs() < 1e-5, "expected 0, got {}", val);
        }
    }

    // --- DivideImageFilter ---------------------------------------------------

    #[test]
    fn divide_filter_computes_elementwise_quotient() {
        let a = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
        let b = make_image(vec![2.0, 4.0, 5.0, 8.0], [1, 2, 2]);
        let out = DivideImageFilter::new().apply(&a, &b).unwrap();
        let v = voxels(&out);
        let expected = [5.0f32, 5.0, 6.0, 5.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "[{}] expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    #[test]
    fn divide_filter_division_by_zero_yields_zero() {
        let a = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let b = make_image(vec![0.0, 1.0, 0.0, 2.0], [1, 2, 2]);
        let out = DivideImageFilter::new().apply(&a, &b).unwrap();
        let v = voxels(&out);
        assert!(
            (v[0] - 0.0).abs() < 1e-5,
            "div-by-zero at [0]: got {}",
            v[0]
        );
        assert!((v[1] - 2.0).abs() < 1e-5, "[1]: expected 2, got {}", v[1]);
        assert!(
            (v[2] - 0.0).abs() < 1e-5,
            "div-by-zero at [2]: got {}",
            v[2]
        );
        assert!((v[3] - 2.0).abs() < 1e-5, "[3]: expected 2, got {}", v[3]);
    }

    // --- ImageMinFilter / ImageMaxFilter -------------------------------------

    #[test]
    fn min_filter_returns_elementwise_minimum() {
        let a = make_image(vec![1.0, 5.0, 3.0, 7.0], [1, 2, 2]);
        let b = make_image(vec![4.0, 2.0, 6.0, 1.0], [1, 2, 2]);
        let out = ImageMinFilter::new().apply(&a, &b).unwrap();
        let v = voxels(&out);
        let expected = [1.0f32, 2.0, 3.0, 1.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "[{}] expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    #[test]
    fn max_filter_returns_elementwise_maximum() {
        let a = make_image(vec![1.0, 5.0, 3.0, 7.0], [1, 2, 2]);
        let b = make_image(vec![4.0, 2.0, 6.0, 1.0], [1, 2, 2]);
        let out = ImageMaxFilter::new().apply(&a, &b).unwrap();
        let v = voxels(&out);
        let expected = [4.0f32, 5.0, 6.0, 7.0];
        for (i, (&got, &exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "[{}] expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }
}
