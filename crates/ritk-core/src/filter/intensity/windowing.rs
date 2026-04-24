//! Intensity windowing filter (clamp-then-rescale).
//!
//! # Mathematical Specification
//!
//! Let f(x) = clamp(I(x), window_min, window_max).
//! If window_min == window_max: output(x) = out_min.
//! Else: output(x) = (f(x) - window_min) / (window_max - window_min) * (out_max - out_min) + out_min
//!
//! Pixels below window_min map to out_min; pixels above window_max map to out_max.
//! Interior pixels are mapped linearly.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Clamp input to [window_min, window_max], then rescale to [out_min, out_max].
#[derive(Debug, Clone)]
pub struct IntensityWindowingFilter {
    /// Lower bound of the intensity window.
    pub window_min: f32,
    /// Upper bound of the intensity window.
    pub window_max: f32,
    /// Minimum output value (maps from window_min).
    pub out_min: f32,
    /// Maximum output value (maps from window_max).
    pub out_max: f32,
}

impl IntensityWindowingFilter {
    /// Construct with explicit window and output ranges.
    pub fn new(window_min: f32, window_max: f32, out_min: f32, out_max: f32) -> Self {
        Self { window_min, window_max, out_min, out_max }
    }

    /// Apply windowing to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let wmin = self.window_min;
        let wmax = self.window_max;
        let omin = self.out_min;
        let omax = self.out_max;

        let out: Vec<f32> = if (wmax - wmin).abs() < f32::EPSILON {
            vec![omin; vals.len()]
        } else {
            let scale = (omax - omin) / (wmax - wmin);
            vals.iter()
                .map(|&v| {
                    let clamped = v.max(wmin).min(wmax);
                    (clamped - wmin) * scale + omin
                })
                .collect()
        };

        Ok(rebuild(out, dims, image))
    }
}

fn extract_vec<B: Backend>(image: &Image<B, 3>) -> anyhow::Result<(Vec<f32>, [usize; 3])> {
    let vals = image.data().clone().into_data()
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("IntensityWindowingFilter requires f32 data: {:?}", e))?;
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
    fn test_below_window_clamp_to_out_min() {
        // Values -10 are below window [0, 100] -> out_min = 0.0
        let img = make_image(vec![-10.0, -5.0, -1.0]);
        let f = IntensityWindowingFilter::new(0.0, 100.0, 0.0, 1.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for &v in &result {
            assert!((v - 0.0).abs() < 1e-6, "below window -> out_min=0.0, got {}", v);
        }
    }

    #[test]
    fn test_above_window_clamp_to_out_max() {
        let img = make_image(vec![200.0, 300.0, 1000.0]);
        let f = IntensityWindowingFilter::new(0.0, 100.0, 0.0, 1.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-6, "above window -> out_max=1.0, got {}", v);
        }
    }

    #[test]
    fn test_interior_linear_mapping() {
        // Value at midpoint of window -> midpoint of output
        let img = make_image(vec![50.0]); // midpoint of [0, 100]
        let f = IntensityWindowingFilter::new(0.0, 100.0, 0.0, 1.0);
        let result = get_vals(&f.apply(&img).unwrap());
        assert!((result[0] - 0.5).abs() < 1e-5, "midpoint -> 0.5, got {}", result[0]);
    }

    #[test]
    fn test_full_image_output_bounded() {
        let vals: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let img = make_image(vals);
        let f = IntensityWindowingFilter::new(20.0, 80.0, 0.0, 255.0);
        let result = get_vals(&f.apply(&img).unwrap());
        let min_out = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_out = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((min_out - 0.0).abs() < 1e-4, "min output = 0.0, got {}", min_out);
        assert!((max_out - 255.0).abs() < 1e-4, "max output = 255.0, got {}", max_out);
    }

    #[test]
    fn test_equal_window_bounds_gives_out_min() {
        let img = make_image(vec![50.0, 100.0, 200.0]);
        // window_min == window_max -> all pixels -> out_min
        let f = IntensityWindowingFilter::new(100.0, 100.0, 3.0, 7.0);
        let result = get_vals(&f.apply(&img).unwrap());
        for &v in &result {
            assert!((v - 3.0).abs() < 1e-6, "equal window -> out_min=3.0, got {}", v);
        }
    }
}
