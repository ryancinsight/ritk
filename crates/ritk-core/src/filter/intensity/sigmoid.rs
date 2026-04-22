//! Sigmoid intensity transform filter.
//!
//! # Mathematical Specification
//!
//! output(x) = (max_output - min_output) / (1 + exp(-(I(x) - alpha) / beta)) + min_output
//!
//! Special case: if |beta| < 1e-12, use step function:
//!   output(x) = if I(x) >= alpha { max_output } else { min_output }
//!
//! At I(x) = alpha:        output = (max_output + min_output) / 2
//! At I(x) = alpha + beta: output = (max_output - min_output) / (1 + exp(-1)) + min_output
//!
//! Reference: Sethian (1996). The output is strictly bounded in (min_output, max_output)
//! for finite input and nonzero beta.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Pixel-wise sigmoid intensity transform.
///
/// Maps I(x) to (min_output, max_output) via the sigmoid function.
#[derive(Debug, Clone)]
pub struct SigmoidImageFilter {
    /// Inflection point (input value at which output = (max + min) / 2).
    pub alpha: f32,
    /// Width of the transition region.
    pub beta: f32,
    /// Minimum output intensity.
    pub min_output: f32,
    /// Maximum output intensity.
    pub max_output: f32,
}

impl SigmoidImageFilter {
    pub fn new(alpha: f32, beta: f32, min_output: f32, max_output: f32) -> Self {
        Self { alpha, beta, min_output, max_output }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let alpha = self.alpha;
        let beta = self.beta;
        let min_o = self.min_output;
        let max_o = self.max_output;
        let range = max_o - min_o;

        let out: Vec<f32> = if beta.abs() < 1e-12 {
            vals.iter().map(|&v| if v >= alpha { max_o } else { min_o }).collect()
        } else {
            vals.iter()
                .map(|&v| range / (1.0 + (-(v - alpha) / beta).exp()) + min_o)
                .collect()
        };

        Ok(rebuild(out, dims, image))
    }
}

fn extract_vec<B: Backend>(image: &Image<B, 3>) -> anyhow::Result<(Vec<f32>, [usize; 3])> {
    let td = image.data().clone().into_data();
    let vals = td.as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("SigmoidImageFilter requires f32 data: {:?}", e))?
        .to_vec();
    Ok((vals, image.shape()))
}

fn rebuild<B: Backend>(vals: Vec<f32>, dims: [usize; 3], src: &Image<B, 3>) -> Image<B, 3> {
    let device = src.data().device();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(tensor, src.origin().clone(), src.spacing().clone(), src.direction().clone())
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
        img.data().clone().into_data().as_slice::<f32>().unwrap().to_vec()
    }

    #[test]
    fn test_at_alpha_gives_midpoint() {
        // At I(x) = alpha, sigmoid(0) = 0.5, so output = range*0.5 + min = 0.5
        let img = make_image(vec![2.0_f32]); // alpha = 2.0
        let f = SigmoidImageFilter::new(2.0, 1.0, 0.0, 1.0);
        let result = get_vals(&f.apply(&img).unwrap());
        let expected = 0.5_f32;
        assert!((result[0] - expected).abs() < 1e-5, "at alpha -> 0.5, got {}", result[0]);
    }

    #[test]
    fn test_monotone_increasing_with_positive_beta() {
        let vals = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let img = make_image(vals);
        let f = SigmoidImageFilter::new(2.0, 1.0, 0.0, 1.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for i in 0..result.len() - 1 {
            assert!(result[i] < result[i + 1], "sigmoid must be monotone increasing, positions {} and {}", i, i+1);
        }
    }

    #[test]
    fn test_output_range_bounded() {
        let vals: Vec<f32> = (-50i32..=50).map(|i| i as f32).collect();
        let img = make_image(vals);
        let f = SigmoidImageFilter::new(0.0, 1.0, 0.0, 1.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for &v in &result {
            // In f32, exp(-50) < f32::EPSILON, so 1.0 + exp(-50) == 1.0 exactly.
            // The sigmoid is bounded in [0, 1] in f32; strict-open bound requires wider domain.
            assert!(v >= 0.0 && v <= 1.0, "sigmoid output must be in [0, 1], got {}", v);
        }
    }

    #[test]
    fn test_large_positive_input_approaches_max() {
        let img = make_image(vec![1e6_f32]);
        let f = SigmoidImageFilter::new(0.0, 1.0, 0.0, 1.0);
        let result = get_vals(&f.apply(&img).unwrap());
        assert!(result[0] > 0.9999, "large positive input should approach max_output=1.0, got {}", result[0]);
    }

    #[test]
    fn test_large_negative_input_approaches_min() {
        let img = make_image(vec![-1e6_f32]);
        let f = SigmoidImageFilter::new(0.0, 1.0, 0.0, 1.0);
        let result = get_vals(&f.apply(&img).unwrap());
        assert!(result[0] < 0.0001, "large negative input should approach min_output=0.0, got {}", result[0]);
    }
}
