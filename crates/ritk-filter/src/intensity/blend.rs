//! Alpha blending filter for pixel-level image fusion.
//!
//! # Mathematical Specification
//!
//! Let `A, B : ℤ³ → ℝ` be two images with identical shape `[nz, ny, nx]`,
//! and let `α ∈ [0, 1]` be a blending weight.
//!
//! `out(x) = (1 - α) * A(x) + α * B(x)`
//!
//! Spatial metadata (origin, spacing, direction) is taken from the **first** input image.
//! Both images must have identical shapes; a shape mismatch returns `Err`.
//!
//! # ITK Parity
//!
//! `itk::BlendImageFilter`

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_annotation::overlay::Opacity;
use ritk_image::tensor::Backend;
use ritk_image::{native::Image as NativeImage, Image};
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Linearly blend two co-registered images.
///
/// `out(x) = (1 - alpha) * a(x) + alpha * b(x)`
///
/// - `alpha = 0.0` yields exactly `a(x)`.
/// - `alpha = 1.0` yields exactly `b(x)`.
///
/// # ITK Parity: `BlendImageFilter`
#[derive(Debug, Clone)]
pub struct BlendImageFilter {
    pub alpha: Opacity,
}

impl Default for BlendImageFilter {
    fn default() -> Self {
        Self {
            alpha: Opacity::new(0.5),
        }
    }
}

impl BlendImageFilter {
    /// Create a new blend filter with the given alpha [0.0, 1.0].
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: Opacity::new(alpha),
        }
    }

    /// Set the alpha blending weight.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = Opacity::new(alpha);
        self
    }

    pub fn apply<B: Backend>(
        &self,
        a: &Image<B, 3>,
        b: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        anyhow::ensure!(
            a.shape() == b.shape(),
            "BlendImageFilter: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        );

        let (av_vec, dims) = extract_vec_infallible(a);
        let av = &av_vec;
        let (bv_vec, _) = extract_vec_infallible(b);
        let bv = &bv_vec;

        let alpha = self.alpha.get();
        let one_minus_alpha = 1.0 - alpha;

        let out: Vec<f32> = av
            .iter()
            .zip(bv.iter())
            .map(|(&x, &y)| one_minus_alpha * x + alpha * y)
            .collect();

        Ok(rebuild(out, dims, a))
    }

    /// Blend two co-registered Coeus-native images.
    pub fn apply_native<B>(
        &self,
        a: &NativeImage<f32, B, 3>,
        b: &NativeImage<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<NativeImage<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        anyhow::ensure!(
            a.shape() == b.shape(),
            "BlendImageFilter: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
        let alpha = self.alpha.get();
        let complement = 1.0 - alpha;
        let values = a
            .data_slice()?
            .iter()
            .zip(b.data_slice()?)
            .map(|(&left, &right)| complement * left + alpha * right)
            .collect();
        NativeImage::from_flat_on(
            values,
            a.shape(),
            *a.origin(),
            *a.spacing(),
            *a.direction(),
            backend,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use coeus_core::SequentialBackend;
    use ritk_image::native::Image as NativeImage;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use ritk_spatial::{Direction, Point, Spacing};

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
        img.data_slice().into_owned()
    }

    #[test]
    fn blend_filter_half_alpha() {
        let a = make_image(vec![0.0, 10.0, 20.0, 30.0], [1, 2, 2]);
        let b = make_image(vec![100.0, 100.0, 100.0, 100.0], [1, 2, 2]);
        let out = BlendImageFilter::new(0.5).apply(&a, &b).unwrap();
        let v = voxels(&out);
        let expected = [50.0f32, 55.0, 60.0, 65.0];
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
    fn blend_filter_zero_alpha() {
        let a = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let b = make_image(vec![10.0, 20.0, 30.0], [1, 1, 3]);
        let out = BlendImageFilter::new(0.0).apply(&a, &b).unwrap();
        let v = voxels(&out);
        for (i, &got) in v.iter().enumerate() {
            assert!((got - (i as f32 + 1.0)).abs() < 1e-5, "[{}] expected A", i);
        }
    }

    #[test]
    fn blend_filter_one_alpha() {
        let a = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let b = make_image(vec![10.0, 20.0, 30.0], [1, 1, 3]);
        let out = BlendImageFilter::new(1.0).apply(&a, &b).unwrap();
        let v = voxels(&out);
        for (i, &got) in v.iter().enumerate() {
            assert!(
                (got - ((i as f32 + 1.0) * 10.0)).abs() < 1e-5,
                "[{}] expected B",
                i
            );
        }
    }

    #[test]
    fn native_blend_preserves_values_and_first_image_metadata() {
        let image = |values, origin| {
            NativeImage::from_flat_on(
                values,
                [1, 1, 3],
                Point::new(origin),
                Spacing::new([0.5, 1.0, 2.0]),
                Direction::identity(),
                &SequentialBackend,
            )
            .expect("invariant: valid native image")
        };
        let a = image(vec![0.0, 10.0, 20.0], [1.0, 2.0, 3.0]);
        let b = image(vec![100.0, 100.0, 100.0], [9.0, 8.0, 7.0]);
        let output = BlendImageFilter::new(0.5)
            .apply_native(&a, &B::default(), &SequentialBackend)
            .expect("native blend succeeds");

        assert_eq!(
            output.data_slice().expect("invariant: contiguous storage"),
            &[50.0, 55.0, 60.0]
        );
        assert_eq!(output.origin(), a.origin());
        assert_eq!(output.spacing(), a.spacing());
        assert_eq!(output.direction(), a.direction());
    }
}
