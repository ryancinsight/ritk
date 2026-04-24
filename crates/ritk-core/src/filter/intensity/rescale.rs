//! Linear intensity rescaling filter.
//!
//! # Mathematical Specification
//!
//! Let I_min = min_{x} I(x), I_max = max_{x} I(x).
//! If I_min == I_max: output(x) = out_min for all x.
//! Else: output(x) = (I(x) - I_min) / (I_max - I_min) x (out_max - out_min) + out_min
//!
//! This is the unique affine bijection mapping [I_min, I_max] to [out_min, out_max].

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Linear rescale of image intensity to [out_min, out_max].
///
/// Computes the global minimum and maximum of the input image and maps the
/// intensity range [I_min, I_max] linearly to [out_min, out_max].
#[derive(Debug, Clone)]
pub struct RescaleIntensityFilter {
    /// Minimum output intensity value.
    pub out_min: f32,
    /// Maximum output intensity value.
    pub out_max: f32,
}

impl RescaleIntensityFilter {
    /// Construct with explicit output range.
    pub fn new(out_min: f32, out_max: f32) -> Self {
        Self { out_min, out_max }
    }

    /// Construct with unit output range [0.0, 1.0].
    pub fn unit() -> Self {
        Self { out_min: 0.0, out_max: 1.0 }
    }

    /// Apply the rescaling to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let i_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let i_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let out: Vec<f32> = if (i_max - i_min).abs() < f32::EPSILON {
            vec![self.out_min; vals.len()]
        } else {
            let scale = (self.out_max - self.out_min) / (i_max - i_min);
            vals.iter()
                .map(|&v| (v - i_min) * scale + self.out_min)
                .collect()
        };

        Ok(rebuild(out, dims, image))
    }
}

fn extract_vec<B: Backend>(image: &Image<B, 3>) -> anyhow::Result<(Vec<f32>, [usize; 3])> {
    let vals = image.data().clone().into_data()
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("RescaleIntensityFilter requires f32 data: {:?}", e))?;
    Ok((vals, image.shape()))
}

fn rebuild<B: Backend>(vals: Vec<f32>, dims: [usize; 3], src: &Image<B, 3>) -> Image<B, 3> {
    let device = src.data().device();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        src.origin().clone(),
        src.spacing().clone(),
        src.direction().clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>) -> Image<B, 3> {
        let n = vals.len();
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new([1, 1, n]));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(tensor, Point::new([0.0; 3]), Spacing::new([1.0; 3]), Direction::identity())
    }

    fn get_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    #[test]
    fn test_uniform_image_gives_out_min() {
        let img = make_image(vec![5.0_f32; 8]);
        let out = RescaleIntensityFilter::unit().apply(&img).unwrap();
        let vals = get_vals(&out);
        for &v in &vals {
            assert!((v - 0.0).abs() < 1e-6, "uniform image -> out_min=0.0, got {}", v);
        }
    }

    #[test]
    fn test_ramp_rescale_to_unit() {
        let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let img = make_image(vals);
        let out = RescaleIntensityFilter::unit().apply(&img).unwrap();
        let result = get_vals(&out);
        assert!((result[0] - 0.0).abs() < 1e-6, "min -> 0.0, got {}", result[0]);
        assert!((result[9] - 1.0).abs() < 1e-6, "max -> 1.0, got {}", result[9]);
        // Intermediate: (5 - 0) / (9 - 0) = 5/9
        let expected_mid = 5.0_f32 / 9.0;
        assert!((result[5] - expected_mid).abs() < 1e-5, "mid -> {}, got {}", expected_mid, result[5]);
    }

    #[test]
    fn test_custom_output_range() {
        let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let img = make_image(vals);
        let out = RescaleIntensityFilter::new(2.0, 5.0).apply(&img).unwrap();
        let result = get_vals(&out);
        assert!((result[0] - 2.0).abs() < 1e-5, "min -> 2.0, got {}", result[0]);
        assert!((result[9] - 5.0).abs() < 1e-5, "max -> 5.0, got {}", result[9]);
    }

    #[test]
    fn test_negative_values_rescaled() {
        let vals: Vec<f32> = (-5i32..=5).map(|i| i as f32).collect(); // -5..=5
        let img = make_image(vals);
        let out = RescaleIntensityFilter::unit().apply(&img).unwrap();
        let result = get_vals(&out);
        assert!((result[0] - 0.0).abs() < 1e-5, "min=-5 -> 0.0, got {}", result[0]);
        assert!((result[10] - 1.0).abs() < 1e-5, "max=5 -> 1.0, got {}", result[10]);
    }

    #[test]
    fn test_single_voxel_gives_out_min() {
        let img = make_image(vec![42.0_f32]);
        let out = RescaleIntensityFilter::new(3.0, 7.0).apply(&img).unwrap();
        let result = get_vals(&out);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.0).abs() < 1e-6, "single voxel -> out_min=3.0, got {}", result[0]);
    }
}