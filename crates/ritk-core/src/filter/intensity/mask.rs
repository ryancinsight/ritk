//! Mask image filters for selective voxel zeroing.
//!
//! # Mathematical Specification
//!
//! Let `I : ℤ³ → ℝ` be the input image and `M : ℤ³ → ℝ` the mask image.
//!
//! **`MaskImageFilter`** (inside mask):
//! `out(x) = I(x)` if `M(x) > threshold`, else `outside_value`
//!
//! **`MaskNegatedImageFilter`** (outside mask):
//! `out(x) = I(x)` if `M(x) ≤ threshold`, else `outside_value`
//!
//! Default `threshold = 0.5`, `outside_value = 0.0`.
//! Spatial metadata (origin, spacing, direction) is taken from the input image `I`.
//!
//! # ITK / SimpleITK Parity
//!
//! | Filter                      | ITK class                    |
//! |-----------------------------|------------------------------|
//! | `MaskImageFilter`           | `MaskImageFilter`            |
//! | `MaskNegatedImageFilter`    | `MaskNegatedImageFilter`     |

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

fn check_shapes(a: [usize; 3], b: [usize; 3]) -> anyhow::Result<()> {
    anyhow::ensure!(
        a == b,
        "mask filter: shape mismatch image {:?} vs mask {:?}",
        a, b
    );
    Ok(())
}

fn extract<B: Backend>(img: &Image<B, 3>) -> anyhow::Result<Vec<f32>> {
    img.data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("mask filter requires f32 data: {:?}", e))
}

fn rebuild<B: Backend>(vals: Vec<f32>, dims: [usize; 3], src: &Image<B, 3>) -> Image<B, 3> {
    let device = src.data().device();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        *src.origin(),
        *src.spacing(),
        *src.direction(),
    )
}

// ── MaskImageFilter ───────────────────────────────────────────────────────────

/// Retain image values where the mask is active (> threshold); replace elsewhere.
///
/// `out(x) = image(x)` if `mask(x) > threshold`, else `outside_value`
///
/// # ITK Parity: `MaskImageFilter`
#[derive(Debug, Clone)]
pub struct MaskImageFilter {
    /// Foreground threshold for the mask image. Default: 0.5.
    pub threshold: f32,
    /// Value written where the mask is inactive. Default: 0.0.
    pub outside_value: f32,
}

impl Default for MaskImageFilter {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            outside_value: 0.0,
        }
    }
}

impl MaskImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    pub fn with_outside_value(mut self, v: f32) -> Self {
        self.outside_value = v;
        self
    }

    pub fn apply<B: Backend>(
        &self,
        image: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        check_shapes(dims, mask.shape())?;
        let iv = extract(image)?;
        let mv = extract(mask)?;
        let outside = self.outside_value;
        let thr = self.threshold;
        let out: Vec<f32> = iv
            .iter()
            .zip(mv.iter())
            .map(|(&img_val, &mask_val)| if mask_val > thr { img_val } else { outside })
            .collect();
        Ok(rebuild(out, dims, image))
    }
}

// ── MaskNegatedImageFilter ────────────────────────────────────────────────────

/// Retain image values where the mask is **inactive** (≤ threshold); replace elsewhere.
///
/// `out(x) = image(x)` if `mask(x) ≤ threshold`, else `outside_value`
///
/// # ITK Parity: `MaskNegatedImageFilter`
#[derive(Debug, Clone)]
pub struct MaskNegatedImageFilter {
    /// Foreground threshold for the mask image. Default: 0.5.
    pub threshold: f32,
    /// Value written where the mask is active. Default: 0.0.
    pub outside_value: f32,
}

impl Default for MaskNegatedImageFilter {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            outside_value: 0.0,
        }
    }
}

impl MaskNegatedImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    pub fn with_outside_value(mut self, v: f32) -> Self {
        self.outside_value = v;
        self
    }

    pub fn apply<B: Backend>(
        &self,
        image: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        check_shapes(dims, mask.shape())?;
        let iv = extract(image)?;
        let mv = extract(mask)?;
        let outside = self.outside_value;
        let thr = self.threshold;
        let out: Vec<f32> = iv
            .iter()
            .zip(mv.iter())
            .map(|(&img_val, &mask_val)| if mask_val <= thr { img_val } else { outside })
            .collect();
        Ok(rebuild(out, dims, image))
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

    #[test]
    fn mask_filter_passes_image_values_where_mask_active() {
        let img = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
        let mask = make_image(vec![1.0, 0.0, 1.0, 0.0], [1, 2, 2]);
        let out = MaskImageFilter::new().apply(&img, &mask).unwrap();
        let v = voxels(&out);
        assert!((v[0] - 10.0).abs() < 1e-5, "mask-active: expected 10, got {}", v[0]);
        assert!((v[1] - 0.0).abs() < 1e-5, "mask-inactive: expected 0, got {}", v[1]);
        assert!((v[2] - 30.0).abs() < 1e-5, "mask-active: expected 30, got {}", v[2]);
        assert!((v[3] - 0.0).abs() < 1e-5, "mask-inactive: expected 0, got {}", v[3]);
    }

    #[test]
    fn mask_filter_full_mask_is_identity() {
        let vals = vec![5.0f32, 6.0, 7.0, 8.0];
        let img = make_image(vals.clone(), [1, 2, 2]);
        let mask = make_image(vec![1.0; 4], [1, 2, 2]);
        let out = MaskImageFilter::new().apply(&img, &mask).unwrap();
        let v = voxels(&out);
        for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "[{}] expected {}, got {}", i, b, a);
        }
    }

    #[test]
    fn mask_filter_custom_outside_value() {
        let img = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
        let mask = make_image(vec![0.0, 1.0, 0.0, 1.0], [1, 2, 2]);
        let out = MaskImageFilter::new()
            .with_outside_value(-1.0)
            .apply(&img, &mask)
            .unwrap();
        let v = voxels(&out);
        assert!((v[0] - (-1.0)).abs() < 1e-5, "outside: expected -1, got {}", v[0]);
        assert!((v[1] - 20.0).abs() < 1e-5, "inside: expected 20, got {}", v[1]);
    }

    #[test]
    fn mask_filter_shape_mismatch_returns_error() {
        let img = make_image(vec![1.0; 4], [1, 2, 2]);
        let mask = make_image(vec![1.0; 8], [2, 2, 2]);
        assert!(MaskImageFilter::new().apply(&img, &mask).is_err());
    }

    #[test]
    fn mask_filter_preserves_spatial_metadata() {
        let img = make_image(vec![1.0; 8], [2, 2, 2]);
        let mask = make_image(vec![1.0; 8], [2, 2, 2]);
        let out = MaskImageFilter::new().apply(&img, &mask).unwrap();
        assert_eq!(out.shape(), img.shape());
        assert_eq!(out.spacing(), img.spacing());
    }

    #[test]
    fn mask_negated_filter_passes_values_where_mask_inactive() {
        let img = make_image(vec![10.0, 20.0, 30.0, 40.0], [1, 2, 2]);
        let mask = make_image(vec![1.0, 0.0, 1.0, 0.0], [1, 2, 2]);
        let out = MaskNegatedImageFilter::new().apply(&img, &mask).unwrap();
        let v = voxels(&out);
        // mask active → outside (0); mask inactive → pass through
        assert!((v[0] - 0.0).abs() < 1e-5, "mask-active zeroed: got {}", v[0]);
        assert!((v[1] - 20.0).abs() < 1e-5, "mask-inactive pass: got {}", v[1]);
        assert!((v[2] - 0.0).abs() < 1e-5, "mask-active zeroed: got {}", v[2]);
        assert!((v[3] - 40.0).abs() < 1e-5, "mask-inactive pass: got {}", v[3]);
    }

    #[test]
    fn mask_negated_full_mask_zeros_everything() {
        let img = make_image(vec![5.0, 6.0, 7.0, 8.0], [1, 2, 2]);
        let mask = make_image(vec![1.0; 4], [1, 2, 2]);
        let out = MaskNegatedImageFilter::new().apply(&img, &mask).unwrap();
        let v = voxels(&out);
        for &val in &v {
            assert!((val - 0.0).abs() < 1e-5, "expected 0, got {}", val);
        }
    }
}
