//! Pixelwise arithmetic intensity transform filters.
//!
//! Each filter is a pure pixelwise map `f : f32 → f32` applied independently to
//! every voxel in a 3-D image. Spatial metadata (origin, spacing, direction) is
//! preserved identically in every output image.
//!
//! # ITK / ImageJ / SimpleITK parity
//!
//! | Filter                       | ITK class                          | ImageJ (Process > Math) |
//! |------------------------------|------------------------------------|-------------------------|
//! | `AbsImageFilter`             | `AbsImageFilter`                   | Abs                     |
//! | `InvertIntensityFilter`      | `InvertIntensityImageFilter`       | (Image > Adjust > Invert) |
//! | `NormalizeImageFilter`       | `NormalizeImageFilter`             | —                       |
//! | `SquareImageFilter`          | `SquareImageFilter`                | Square                  |
//! | `SqrtImageFilter`            | `SqrtImageFilter`                  | Square Root             |
//! | `LogImageFilter`             | `LogImageFilter`                   | Log                     |
//! | `ExpImageFilter`             | `ExpImageFilter`                   | Exp                     |

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Shared helpers ────────────────────────────────────────────────────────────

fn extract_vec<B: Backend>(image: &Image<B, 3>) -> (Vec<f32>, [usize; 3]) {
    let vals = image
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .expect("arithmetic filter requires f32 backend data");
    (vals, image.shape())
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

// ── AbsImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise absolute value filter.
///
/// # Mathematical Specification
///
/// `out(x) = |in(x)|`
///
/// # Properties
/// - Idempotent on non-negative images: `|f(x)| = f(x)` iff `f(x) ≥ 0`.
/// - Preserves spatial metadata.
/// - O(N) time, O(N) output space.
///
/// # References
/// - ITK `itk::AbsImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Abs.
#[derive(Debug, Clone, Copy, Default)]
pub struct AbsImageFilter;

impl AbsImageFilter {
    /// Construct a new `AbsImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the absolute-value transform to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::abs).collect();
        rebuild(out, dims, image)
    }
}

// ── InvertIntensityFilter ─────────────────────────────────────────────────────

/// Intensity inversion filter.
///
/// # Mathematical Specification
///
/// `out(x) = maximum - in(x)`
///
/// where `maximum` is either user-specified or derived from the image:
///   `maximum = max_{x} in(x)` (ITK default).
///
/// The mapping is an affine reflection of the intensity range about its midpoint.
/// The input's maximum voxel maps to `0.0`; the input's minimum voxel maps to
/// `maximum - min(in)`.
///
/// # Properties
/// - `InvertIntensity(InvertIntensity(I, M), M) = I` (involution when M is fixed).
/// - Constant image with value `c` maps to all-zero output (using auto maximum).
///
/// # References
/// - ITK `itk::InvertIntensityImageFilter<TImage>`.
/// - `SimpleITK::InvertIntensity(image, maximum)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct InvertIntensityFilter {
    /// Fixed inversion maximum.  When `None`, the image maximum is used.
    maximum: Option<f32>,
}

impl InvertIntensityFilter {
    /// Construct with automatic maximum (derived from the input image).
    pub fn new() -> Self {
        Self { maximum: None }
    }

    /// Construct with a fixed maximum value.
    ///
    /// `maximum` must be finite; the result is undefined if `maximum` is NaN or
    /// infinite (no error is raised; values saturate silently).
    pub fn with_maximum(maximum: f32) -> Self {
        Self {
            maximum: Some(maximum),
        }
    }

    /// Apply intensity inversion to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let max_val = self
            .maximum
            .unwrap_or_else(|| vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        let out: Vec<f32> = vals.into_iter().map(|v| max_val - v).collect();
        rebuild(out, dims, image)
    }
}

// ── NormalizeImageFilter ──────────────────────────────────────────────────────

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

// ── SquareImageFilter ─────────────────────────────────────────────────────────

/// Pixelwise square filter.
///
/// # Mathematical Specification
///
/// `out(x) = in(x)²`
///
/// # Properties
/// - Non-negative output for all real inputs.
/// - O(N) time.
///
/// # References
/// - ITK `itk::SquareImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Square.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquareImageFilter;

impl SquareImageFilter {
    /// Construct a new `SquareImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply pixelwise squaring to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(|v| v * v).collect();
        rebuild(out, dims, image)
    }
}

// ── SqrtImageFilter ───────────────────────────────────────────────────────────

/// Pixelwise square-root filter.
///
/// # Mathematical Specification
///
/// `out(x) = √in(x)`
///
/// For negative inputs, the IEEE 754 result is `NaN` — matching ITK behaviour.
/// In medical image contexts, inputs are expected to be non-negative; callers
/// requiring defined behaviour for negative inputs should precede this filter
/// with [`AbsImageFilter`].
///
/// # References
/// - ITK `itk::SqrtImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Square Root.
#[derive(Debug, Clone, Copy, Default)]
pub struct SqrtImageFilter;

impl SqrtImageFilter {
    /// Construct a new `SqrtImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply pixelwise square root to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::sqrt).collect();
        rebuild(out, dims, image)
    }
}

// ── LogImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise natural logarithm filter.
///
/// # Mathematical Specification
///
/// `out(x) = ln(in(x))`
///
/// For `in(x) ≤ 0`, the IEEE 754 result is `-∞` (`in(x) = 0`) or `NaN`
/// (`in(x) < 0`) — matching ITK behaviour. In medical image contexts
/// with non-negative intensities, `in(x) = 0` maps to `-∞`.
///
/// # References
/// - ITK `itk::LogImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Log.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogImageFilter;

impl LogImageFilter {
    /// Construct a new `LogImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply pixelwise natural logarithm to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::ln).collect();
        rebuild(out, dims, image)
    }
}

// ── ExpImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise natural exponential filter.
///
/// # Mathematical Specification
///
/// `out(x) = e^{in(x)}`
///
/// # Properties
/// - `out(x) > 0` for all finite inputs.
/// - `LogImageFilter(ExpImageFilter(I)) ≈ I` (composition recovers identity,
///   up to floating-point round-trip error).
///
/// # References
/// - ITK `itk::ExpImageFilter<TInputImage, TOutputImage>`.
/// - ImageJ Process > Math > Exp.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpImageFilter;

impl ExpImageFilter {
    /// Construct a new `ExpImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply pixelwise natural exponential to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::exp).collect();
        rebuild(out, dims, image)
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
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    // ── AbsImageFilter ────────────────────────────────────────────────────────

    /// Non-negative image: abs is identity.
    #[test]
    fn abs_nonneg_is_identity() {
        let img = make_image(vec![0.0, 1.0, 2.5, 10.0], [1, 2, 2]);
        let out = AbsImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(
            v,
            vec![0.0_f32, 1.0, 2.5, 10.0],
            "non-negative input must be unchanged by abs"
        );
    }

    /// Negative values become positive.
    #[test]
    fn abs_negates_negative_voxels() {
        let img = make_image(vec![-3.0, -1.0, 0.0, 2.0], [1, 2, 2]);
        let out = AbsImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(
            v,
            vec![3.0_f32, 1.0, 0.0, 2.0],
            "abs must negate each negative voxel: [-3,-1,0,2] → [3,1,0,2]"
        );
    }

    /// All-negative image: every output is the negation of input.
    #[test]
    fn abs_all_negative_all_positive() {
        let img = make_image(vec![-5.0, -2.0, -0.5], [1, 1, 3]);
        let out = AbsImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert!(v >= 0.0, "abs output must be non-negative; got {v}");
        }
    }

    /// Spatial metadata is preserved.
    #[test]
    fn abs_preserves_metadata() {
        let sp = Spacing::new([2.0, 3.0, 4.0]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![1.0_f32, -1.0], Shape::new([1usize, 1, 2]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = AbsImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    /// Constant positive image: unchanged.
    #[test]
    fn abs_constant_positive_unchanged() {
        let img = make_image(vec![7.0, 7.0, 7.0], [1, 1, 3]);
        let out = AbsImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 7.0_f32, "constant positive image unchanged by abs");
        }
    }

    // ── InvertIntensityFilter ─────────────────────────────────────────────────

    /// Auto maximum: [1,2,3] → max=3, out=[2,1,0].
    #[test]
    fn invert_auto_max() {
        let img = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let out = InvertIntensityFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(
            v,
            vec![2.0_f32, 1.0, 0.0],
            "[1,2,3] with auto max=3 must invert to [2,1,0]"
        );
    }

    /// Fixed maximum: [1,4,7] with max=10 → [9,6,3].
    #[test]
    fn invert_fixed_max() {
        let img = make_image(vec![1.0, 4.0, 7.0], [1, 1, 3]);
        let out = InvertIntensityFilter::with_maximum(10.0).apply(&img);
        let v = vals(&out);
        assert_eq!(
            v,
            vec![9.0_f32, 6.0, 3.0],
            "[1,4,7] inverted with max=10 must yield [9,6,3]"
        );
    }

    /// Minimum maps to (max - min), maximum maps to 0.
    #[test]
    fn invert_max_maps_to_zero_min_maps_to_range() {
        let img = make_image(vec![2.0, 5.0], [1, 1, 2]);
        let out = InvertIntensityFilter::new().apply(&img);
        let v = vals(&out);
        // auto max = 5.0; 5 - 5 = 0, 5 - 2 = 3
        assert_eq!(v[0], 3.0_f32, "minimum voxel 2 with max=5 → 5-2=3");
        assert_eq!(v[1], 0.0_f32, "maximum voxel 5 with max=5 → 5-5=0");
    }

    /// Constant image with auto max → all zero.
    #[test]
    fn invert_constant_auto_max_all_zero() {
        let img = make_image(vec![4.0, 4.0, 4.0], [1, 1, 3]);
        let out = InvertIntensityFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "constant image with auto max → 0 everywhere");
        }
    }

    /// Spatial metadata is preserved.
    #[test]
    fn invert_preserves_metadata() {
        let sp = Spacing::new([1.5, 2.5, 3.5]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![1.0_f32, 3.0], Shape::new([1usize, 1, 2]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = InvertIntensityFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    // ── NormalizeImageFilter ──────────────────────────────────────────────────

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
        let expected = vec![-1.0_f32, -1.0, 1.0, 1.0];
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
        let img = Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = NormalizeImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    // ── SquareImageFilter ─────────────────────────────────────────────────────

    /// [1,2,3] → [1,4,9].
    #[test]
    fn square_known_values() {
        let img = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let out = SquareImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(v, vec![1.0_f32, 4.0, 9.0], "[1,2,3]² must be [1,4,9]");
    }

    /// Zero voxel → zero output.
    #[test]
    fn square_zero_is_zero() {
        let img = make_image(vec![0.0, 0.0], [1, 1, 2]);
        let out = SquareImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "0² = 0");
        }
    }

    /// Negative inputs → positive output (square of negative is positive).
    #[test]
    fn square_negative_positive_output() {
        let img = make_image(vec![-2.0, -3.0], [1, 1, 2]);
        let out = SquareImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(v, vec![4.0_f32, 9.0], "(-2)² = 4, (-3)² = 9");
    }

    /// Spatial metadata is preserved.
    #[test]
    fn square_preserves_metadata() {
        let sp = Spacing::new([3.0, 3.0, 3.0]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![2.0_f32], Shape::new([1usize, 1, 1]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = SquareImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    /// Constant [3,3,3] → [9,9,9].
    #[test]
    fn square_constant() {
        let img = make_image(vec![3.0, 3.0, 3.0], [1, 1, 3]);
        let out = SquareImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 9.0_f32, "3² = 9 for each voxel");
        }
    }

    // ── SqrtImageFilter ───────────────────────────────────────────────────────

    /// [0,1,4,9] → [0,1,2,3].
    #[test]
    fn sqrt_perfect_squares() {
        let img = make_image(vec![0.0, 1.0, 4.0, 9.0], [1, 2, 2]);
        let out = SqrtImageFilter::new().apply(&img);
        let v = vals(&out);
        assert_eq!(v, vec![0.0_f32, 1.0, 2.0, 3.0], "sqrt of perfect squares");
    }

    /// All-zero image → all-zero output.
    #[test]
    fn sqrt_zero_is_zero() {
        let img = make_image(vec![0.0, 0.0, 0.0], [1, 1, 3]);
        let out = SqrtImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 0.0_f32, "sqrt(0) = 0");
        }
    }

    /// Constant [4,4] → [2,2].
    #[test]
    fn sqrt_constant() {
        let img = make_image(vec![4.0, 4.0], [1, 1, 2]);
        let out = SqrtImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert_eq!(v, 2.0_f32, "sqrt(4) = 2");
        }
    }

    /// Spatial metadata is preserved.
    #[test]
    fn sqrt_preserves_metadata() {
        let sp = Spacing::new([1.0, 2.0, 4.0]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![9.0_f32], Shape::new([1usize, 1, 1]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = SqrtImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    // ── LogImageFilter ────────────────────────────────────────────────────────

    /// ln(1) = 0.
    #[test]
    fn log_of_one_is_zero() {
        let img = make_image(vec![1.0, 1.0], [1, 1, 2]);
        let out = LogImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert!((v).abs() < 1e-6, "ln(1) = 0; got {v}");
        }
    }

    /// ln(e) ≈ 1.0.
    #[test]
    fn log_of_e_is_one() {
        let e = std::f32::consts::E;
        let img = make_image(vec![e], [1, 1, 1]);
        let out = LogImageFilter::new().apply(&img);
        let v = vals(&out)[0];
        assert!((v - 1.0).abs() < 1e-5, "ln(e) must be ≈ 1.0; got {v}");
    }

    /// ln(e²) ≈ 2.0 — verifies multiplicative correctness.
    ///
    /// # Derivation
    /// e² = exp(2) ≈ 7.389056 as f32. ln(e²) = 2.0 exactly in exact arithmetic;
    /// f32 round-trip introduces error < 1e-5.
    #[test]
    fn log_of_e_squared_is_two() {
        let e2 = (2.0_f32).exp(); // e² in f32
        let img = make_image(vec![e2], [1, 1, 1]);
        let out = LogImageFilter::new().apply(&img);
        let v = vals(&out)[0];
        assert!((v - 2.0).abs() < 1e-4, "ln(e²) must be ≈ 2.0; got {v}");
    }

    /// Spatial metadata is preserved.
    #[test]
    fn log_preserves_metadata() {
        let sp = Spacing::new([2.0, 2.0, 2.0]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![1.0_f32], Shape::new([1usize, 1, 1]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = LogImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    // ── ExpImageFilter ────────────────────────────────────────────────────────

    /// exp(0) = 1.
    #[test]
    fn exp_of_zero_is_one() {
        let img = make_image(vec![0.0, 0.0], [1, 1, 2]);
        let out = ExpImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert!((v - 1.0).abs() < 1e-6, "exp(0) = 1; got {v}");
        }
    }

    /// exp(1) ≈ e.
    #[test]
    fn exp_of_one_is_e() {
        let e = std::f32::consts::E;
        let img = make_image(vec![1.0], [1, 1, 1]);
        let out = ExpImageFilter::new().apply(&img);
        let v = vals(&out)[0];
        assert!((v - e).abs() < 1e-5, "exp(1) must be ≈ e = {e}; got {v}");
    }

    /// exp(2) ≈ e².
    #[test]
    fn exp_of_two_is_e_squared() {
        let e2 = std::f32::consts::E * std::f32::consts::E;
        let img = make_image(vec![2.0], [1, 1, 1]);
        let out = ExpImageFilter::new().apply(&img);
        let v = vals(&out)[0];
        assert!(
            (v - e2).abs() < 1e-3,
            "exp(2) must be ≈ e² ≈ {e2:.4}; got {v}"
        );
    }

    /// Output is always positive for finite inputs.
    #[test]
    fn exp_output_always_positive() {
        let img = make_image(vec![-5.0, -1.0, 0.0, 1.0, 5.0], [1, 1, 5]);
        let out = ExpImageFilter::new().apply(&img);
        for &v in vals(&out).iter() {
            assert!(v > 0.0, "exp(x) must be > 0 for finite x; got {v}");
        }
    }

    /// Spatial metadata is preserved.
    #[test]
    fn exp_preserves_metadata() {
        let sp = Spacing::new([0.5, 0.5, 0.5]);
        let device: burn_ndarray::NdArrayDevice = Default::default();
        let td = TensorData::new(vec![0.0_f32], Shape::new([1usize, 1, 1]));
        let t = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            t,
            Point::new([0.0, 0.0, 0.0]),
            sp.clone(),
            Direction::identity(),
        );
        let out = ExpImageFilter::new().apply(&img);
        assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
    }

    /// Log ∘ Exp ≈ identity (round-trip within f32 precision).
    #[test]
    fn log_exp_roundtrip() {
        let vals_in = vec![0.5_f32, 1.0, 2.0];
        let img = make_image(vals_in.clone(), [1, 1, 3]);
        let exp_out = ExpImageFilter::new().apply(&img);
        let log_out = LogImageFilter::new().apply(&exp_out);
        let v = vals(&log_out);
        for (a, b) in v.iter().zip(vals_in.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "ln(exp(x)) round-trip: got {a}, expected {b}"
            );
        }
    }
}
