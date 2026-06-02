use crate::filter::ops::{extract_vec_infallible as extract_vec, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;

/// Zero-mean, unit-variance intensity normalization filter.
///
/// # Mathematical Specification
///
/// Let `N = n_z · n_y · n_x` be the total voxel count.
/// Define:
///
///   `μ  = Σ_{x} in(x) / N`
///   `σ² = Σ_{x} (in(x) − μ)² / N`
///   `σ  = √σ²`
///
/// Then:
///
///   `out(x) = (in(x) − μ) / σ`      if σ > 0
///   `out(x) = 0`                      if σ = 0  (constant image)
///
/// # Properties
/// - `Σ out(x) / N = 0` (zero mean, exactly by construction).
/// - `Σ (out(x))² / N = 1` (unit variance, exactly by construction).
/// - Constant image → all-zero output (undefined normalisation → zero by convention).
///
/// # References
/// - ITK `itk::NormalizeImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NormalizeImageFilter;

impl NormalizeImageFilter {
    /// Construct a new `NormalizeImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply zero-mean unit-variance normalization to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let n = vals.len() as f64;
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
        let variance = vals
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = variance.sqrt() as f32;
        let mean_f = mean as f32;
        let out: Vec<f32> = if std < f32::EPSILON {
            vec![0.0_f32; vals.len()]
        } else {
            vals.into_iter().map(|v| (v - mean_f) / std).collect()
        };
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let t = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data_vec()
    }

    /// [1,1,3,3] → mean=2, std=1 → [-1,-1,1,1] exactly.
    ///
    /// # Derivation
    /// values = [1,1,3,3], N=4, mean=8/4=2.0
    /// variance = (1+1+1+1)/4 = 1.0, std = 1.0
    /// normalized = (v - 2.0) / 1.0 = [-1,-1,1,1]
    #[test]
    fn normalize_known_values() {
        let img = make_image(vec![1.0, 1.0, 3.0, 3.0], [1, 2, 2]);
        let out = NormalizeImageFilter::new().apply(&img);
        let v = vals(&out);
        let expected = [-1.0_f32, -1.0, 1.0, 1.0];
        for (a, b) in v.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "normalize [1,1,3,3]: got {a}, expected {b}"
            );
        }
    }

    /// Constant image → all zero (degenerate std=0 case).
    #[test]
    fn normalize_constant_image_all_zero() {
        let img = make_image(vec![5.0, 5.0, 5.0], [1, 1, 3]);
        let out = NormalizeImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "constant image with std=0 must map to 0");
        }
    }

    /// Output mean ≈ 0 and output variance ≈ 1 for non-constant input.
    ///
    /// # Derivation
    /// values = [0,2,4,6,8,10], N=6, mean=5, variance=35/3≈11.667, std≈3.416
    /// After normalization: mean of output = Σ(v-5)/std/6 = 0 (sum = Σ(v-5) = 0).
    /// Variance of output = Σ((v-5)/std)²/6 = Σ(v-5)²/(std²·6) = variance/variance = 1.
    #[test]
    fn normalize_output_mean_zero_variance_one() {
        let img = make_image(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0], [1, 2, 3]);
        let out = NormalizeImageFilter::new().apply(&img);
        let v = vals(&out);
        let n = v.len() as f64;
        let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        let variance: f64 = v
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 1e-5, "output mean must be ≈ 0; got {mean}");
        assert!(
            (variance - 1.0).abs() < 1e-4,
            "output variance must be ≈ 1; got {variance}"
        );
    }

    /// Two-element [0,2]: mean=1, std=1 → [-1,1].
    #[test]
    fn normalize_two_element() {
        let img = make_image(vec![0.0, 2.0], [1, 1, 2]);
        let out = NormalizeImageFilter::new().apply(&img);
        let v = vals(&out);
        assert!(
            (v[0] + 1.0).abs() < 1e-5,
            "normalize([0,2])[0] must be -1; got {}",
            v[0]
        );
        assert!(
            (v[1] - 1.0).abs() < 1e-5,
            "normalize([0,2])[1] must be +1; got {}",
            v[1]
        );
    }

    /// Spatial metadata is preserved.
    #[test]
    fn normalize_preserves_metadata() {
        let sp = Spacing::new([0.5, 1.0, 2.0]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![1.0_f32, 3.0], Shape::new([1usize, 1, 2]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(t, Point::new([0.0, 0.0, 0.0]), sp, Direction::identity());
        let out = NormalizeImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }
}
