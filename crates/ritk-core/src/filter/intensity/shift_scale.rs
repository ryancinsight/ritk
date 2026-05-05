//! Shift-scale intensity filter.
//!
//! # Mathematical Specification
//!
//! For each voxel x in image I:
//!
//! `out(x) = (I(x) + shift) * scale`
//!
//! The operation is applied in f64 precision and cast back to f32.
//!
//! ## Invariants
//!
//! - Spatial metadata (shape, origin, spacing, direction) is preserved exactly.
//! - When `shift = 0` and `scale = 1`, `out = I` (identity).
//! - When `scale = 0`, `out = 0` everywhere regardless of shift.
//! - The transform is linear: `f(a + b) = f(a) + scale * b`.
//!
//! # ITK Parity
//!
//! `itk::ShiftScaleImageFilter` with `SetShift(s)` and `SetScale(k)`.
//! Output type defaults to f32 (matching ITK behaviour when input is float).

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Apply a linear shift-then-scale to every voxel.
///
/// `out(x) = (in(x) + shift) * scale`
///
/// # Example
///
/// ```no_run
/// # use ritk_core::filter::ShiftScaleImageFilter;
/// let filter = ShiftScaleImageFilter::new(-1024.0, 0.001);
/// // Converts Hounsfield units centred at –1024 to linear attenuation values
/// ```
#[derive(Debug, Clone)]
pub struct ShiftScaleImageFilter {
    /// Added to each voxel value before multiplication.
    pub shift: f32,
    /// Scale factor applied after shift.
    pub scale: f32,
}

impl Default for ShiftScaleImageFilter {
    fn default() -> Self {
        Self { shift: 0.0, scale: 1.0 }
    }
}

impl ShiftScaleImageFilter {
    /// Create a new filter with the given shift and scale.
    pub fn new(shift: f32, scale: f32) -> Self {
        Self { shift, scale }
    }

    /// Set shift value.
    pub fn with_shift(mut self, s: f32) -> Self {
        self.shift = s;
        self
    }

    /// Set scale value.
    pub fn with_scale(mut self, s: f32) -> Self {
        self.scale = s;
        self
    }

    /// Apply the filter to a 3-D image.
    ///
    /// Returns a new `Image` with identical spatial metadata and
    /// voxel values transformed by `(v + shift) * scale`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("ShiftScaleImageFilter: {:?}", e))?;

        let shift = self.shift as f64;
        let scale = self.scale as f64;

        let out_vals: Vec<f32> = vals
            .iter()
            .map(|&v| ((v as f64 + shift) * scale) as f32)
            .collect();

        let device = image.data().device();
        let out_td = TensorData::new(out_vals, Shape::new(dims));
        let out_tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            out_tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        use burn::tensor::{Shape, Tensor, TensorData};
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(shape));
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
    fn shift_scale_identity_zero_shift_unit_scale() {
        // shift=0, scale=1 → out = in
        let img = make_image(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2]);
        let out = ShiftScaleImageFilter::new(0.0, 1.0).apply(&img).unwrap();
        let v = voxels(&out);
        let expected = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (i, (&a, &b)) in v.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "voxel {} expected {} got {}", i, b, a);
        }
    }

    #[test]
    fn shift_scale_shift_only_adds_constant() {
        // shift=10, scale=1 → out = in + 10
        let img = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let out = ShiftScaleImageFilter::new(10.0, 1.0).apply(&img).unwrap();
        let v = voxels(&out);
        for (i, (&a, &b)) in v.iter().zip([11.0f32, 12.0, 13.0, 14.0].iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "voxel {}: expected {} got {}", i, b, a);
        }
    }

    #[test]
    fn shift_scale_scale_only_multiplies() {
        // shift=0, scale=2 → out = in * 2
        let img = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let out = ShiftScaleImageFilter::new(0.0, 2.0).apply(&img).unwrap();
        let v = voxels(&out);
        for (i, (&a, &b)) in v.iter().zip([2.0f32, 4.0, 6.0, 8.0].iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "voxel {}: expected {} got {}", i, b, a);
        }
    }

    #[test]
    fn shift_scale_combined_shift_then_scale() {
        // shift=-1024, scale=0.001 → HU to fractional
        // Input: 1024 → (1024 - 1024) * 0.001 = 0.0
        // Input: 0 → (0 - 1024) * 0.001 = -1.024
        let img = make_image(vec![1024.0, 0.0], [1, 1, 2]);
        let out = ShiftScaleImageFilter::new(-1024.0, 0.001).apply(&img).unwrap();
        let v = voxels(&out);
        assert!((v[0] - 0.0_f32).abs() < 1e-5, "expected 0.0 got {}", v[0]);
        assert!((v[1] - (-1.024_f32)).abs() < 1e-4, "expected -1.024 got {}", v[1]);
    }

    #[test]
    fn shift_scale_preserves_spatial_metadata() {
        let img = make_image(vec![1.0; 8], [2, 2, 2]);
        let out = ShiftScaleImageFilter::new(5.0, 3.0).apply(&img).unwrap();
        assert_eq!(out.shape(), img.shape());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.direction(), img.direction());
    }

    #[test]
    fn shift_scale_zero_scale_gives_zero() {
        let img = make_image(vec![5.0, 10.0, 15.0, 20.0], [1, 2, 2]);
        let out = ShiftScaleImageFilter::new(100.0, 0.0).apply(&img).unwrap();
        let v = voxels(&out);
        for (i, &x) in v.iter().enumerate() {
            assert!((x - 0.0).abs() < 1e-5, "voxel {} expected 0 got {}", i, x);
        }
    }
}
