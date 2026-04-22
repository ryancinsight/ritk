//! Threshold-based intensity suppression filter.
//!
//! # Mathematical Specification
//!
//! Three modes:
//! - Below:   output(x) = if I(x) < threshold { outside_value } else { I(x) }
//! - Above:   output(x) = if I(x) > threshold { outside_value } else { I(x) }
//! - Outside: output(x) = if I(x) < lower || I(x) > upper { outside_value } else { I(x) }

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Threshold mode controlling which pixels are replaced by outside_value.
#[derive(Debug, Clone)]
pub enum ThresholdMode {
    /// Replace pixels strictly below threshold with outside_value.
    Below { threshold: f32, outside_value: f32 },
    /// Replace pixels strictly above threshold with outside_value.
    Above { threshold: f32, outside_value: f32 },
    /// Replace pixels outside [lower, upper] with outside_value.
    Outside { lower: f32, upper: f32, outside_value: f32 },
}

/// Conditionally replaces pixel values based on a threshold condition.
#[derive(Debug, Clone)]
pub struct ThresholdImageFilter {
    pub mode: ThresholdMode,
}

impl ThresholdImageFilter {
    pub fn below(threshold: f32, outside_value: f32) -> Self {
        Self { mode: ThresholdMode::Below { threshold, outside_value } }
    }
    pub fn above(threshold: f32, outside_value: f32) -> Self {
        Self { mode: ThresholdMode::Above { threshold, outside_value } }
    }
    pub fn outside(lower: f32, upper: f32, outside_value: f32) -> Self {
        Self { mode: ThresholdMode::Outside { lower, upper, outside_value } }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out: Vec<f32> = match &self.mode {
            ThresholdMode::Below { threshold, outside_value } => {
                vals.iter().map(|&v| if v < *threshold { *outside_value } else { v }).collect()
            }
            ThresholdMode::Above { threshold, outside_value } => {
                vals.iter().map(|&v| if v > *threshold { *outside_value } else { v }).collect()
            }
            ThresholdMode::Outside { lower, upper, outside_value } => {
                vals.iter().map(|&v| if v < *lower || v > *upper { *outside_value } else { v }).collect()
            }
        };
        Ok(rebuild(out, dims, image))
    }
}

fn extract_vec<B: Backend>(image: &Image<B, 3>) -> anyhow::Result<(Vec<f32>, [usize; 3])> {
    let td = image.data().clone().into_data();
    let vals = td.as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("ThresholdImageFilter requires f32 data: {:?}", e))?
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
    fn test_below_zeros_low_values() {
        // values 0..9, threshold=5 -> pixels 0,1,2,3,4 become 0.0; 5..9 unchanged
        let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let img = make_image(vals);
        let f = ThresholdImageFilter::below(5.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for i in 0..5usize {
            assert_eq!(result[i], 0.0, "pixel {} (value {}) should be zeroed", i, i);
        }
        for i in 5..10usize {
            assert!((result[i] - i as f32).abs() < 1e-6, "pixel {} should be unchanged", i);
        }
    }

    #[test]
    fn test_above_zeros_high_values() {
        let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let img = make_image(vals);
        let f = ThresholdImageFilter::above(5.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for i in 0..=5usize {
            assert!((result[i] - i as f32).abs() < 1e-6, "pixel {} should be unchanged", i);
        }
        for i in 6..10usize {
            assert_eq!(result[i], 0.0, "pixel {} should be zeroed", i);
        }
    }

    #[test]
    fn test_outside_keeps_interior() {
        let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let img = make_image(vals);
        let f = ThresholdImageFilter::outside(3.0, 6.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for i in 0..3usize {
            assert_eq!(result[i], 0.0, "pixel {} outside [3,6] should be zeroed", i);
        }
        for i in 3..=6usize {
            assert!((result[i] - i as f32).abs() < 1e-6, "pixel {} inside [3,6] should be unchanged", i);
        }
        for i in 7..10usize {
            assert_eq!(result[i], 0.0, "pixel {} outside [3,6] should be zeroed", i);
        }
    }

    #[test]
    fn test_below_no_change_if_all_above() {
        let vals = vec![10.0_f32, 20.0, 30.0];
        let img = make_image(vals.clone());
        let f = ThresholdImageFilter::below(5.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for (a, b) in vals.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-6, "all above threshold: no change expected");
        }
    }

    #[test]
    fn test_above_no_change_if_all_below() {
        let vals = vec![1.0_f32, 2.0, 3.0];
        let img = make_image(vals.clone());
        let f = ThresholdImageFilter::above(10.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for (a, b) in vals.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-6, "all below threshold: no change expected");
        }
    }

    #[test]
    fn test_outside_uniform_inside() {
        let vals = vec![5.0_f32, 5.5, 6.0, 6.5, 7.0];
        let img = make_image(vals.clone());
        let f = ThresholdImageFilter::outside(5.0, 7.0, 0.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for (a, b) in vals.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-6, "all inside [5,7]: unchanged");
        }
    }
}
