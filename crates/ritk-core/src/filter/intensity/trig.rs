//! Pixelwise trigonometric and bounded-reciprocal intensity filters.
//!
//! Each filter is a pure pixelwise map `f : f32 → f32` applied independently to
//! every voxel in a 3-D image. Spatial metadata (origin, spacing, direction) is
//! preserved identically in every output image.
//!
//! All trigonometric arguments are interpreted as radians (matching ITK convention).
//!
//! # ITK / ImageJ / SimpleITK parity
//!
//! | Filter                          | ITK class                             | Note                    |
//! |---------------------------------|---------------------------------------|-------------------------|
//! | `AtanImageFilter`               | `AtanImageFilter`                     | atan(x) ∈ (−π/2, π/2)  |
//! | `SinImageFilter`                | `SinImageFilter`                      | sin(x), x in radians    |
//! | `CosImageFilter`                | `CosImageFilter`                      | cos(x), x in radians    |
//! | `TanImageFilter`                | `TanImageFilter`                      | tan(x), x in radians    |
//! | `AsinImageFilter`               | `AsinImageFilter`                     | asin(x), domain [−1,1]  |
//! | `AcosImageFilter`               | `AcosImageFilter`                     | acos(x), domain [−1,1]  |
//! | `BoundedReciprocalImageFilter`  | `BoundedReciprocalImageFilter`        | 1 / (1 + |x|)           |

use crate::filter::ops::{extract_vec_infallible as extract_vec, rebuild};
use crate::image::Image;
use burn::tensor::backend::Backend;


// ── AtanImageFilter ───────────────────────────────────────────────────────────

/// Pixelwise arctangent filter.
///
/// # Mathematical Specification
///
/// `out(x) = atan(in(x))`
///
/// Range: `(−π/2, π/2)` for all finite inputs.
///
/// # Properties
/// - Bijective on ℝ with range `(−π/2, π/2)`.
/// - Odd function: `atan(−x) = −atan(x)`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::AtanImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AtanImageFilter;

impl AtanImageFilter {
    /// Construct a new `AtanImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the arctangent transform to every voxel.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::atan).collect();
        rebuild(out, dims, image)
    }
}

// ── SinImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise sine filter.
///
/// # Mathematical Specification
///
/// `out(x) = sin(in(x))`
///
/// Input interpreted as radians. Range: `[−1, 1]` for real inputs.
///
/// # Properties
/// - `sin(0) = 0`, `sin(π/2) = 1`, `sin(π) ≈ 0`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::SinImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SinImageFilter;

impl SinImageFilter {
    /// Construct a new `SinImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the sine transform to every voxel (input in radians).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::sin).collect();
        rebuild(out, dims, image)
    }
}

// ── CosImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise cosine filter.
///
/// # Mathematical Specification
///
/// `out(x) = cos(in(x))`
///
/// Input interpreted as radians. Range: `[−1, 1]` for real inputs.
///
/// # Properties
/// - `cos(0) = 1`, `cos(π/2) ≈ 0`, `cos(π) = −1`.
/// - Even function: `cos(−x) = cos(x)`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::CosImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosImageFilter;

impl CosImageFilter {
    /// Construct a new `CosImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the cosine transform to every voxel (input in radians).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::cos).collect();
        rebuild(out, dims, image)
    }
}

// ── TanImageFilter ────────────────────────────────────────────────────────────

/// Pixelwise tangent filter.
///
/// # Mathematical Specification
///
/// `out(x) = tan(in(x))`
///
/// Input interpreted as radians. Undefined at `x = π/2 + nπ` (produces `±∞` or `NaN`).
///
/// # Properties
/// - `tan(0) = 0`, `tan(π/4) = 1`.
/// - Odd function: `tan(−x) = −tan(x)`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::TanImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TanImageFilter;

impl TanImageFilter {
    /// Construct a new `TanImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the tangent transform to every voxel (input in radians).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::tan).collect();
        rebuild(out, dims, image)
    }
}

// ── AsinImageFilter ───────────────────────────────────────────────────────────

/// Pixelwise arcsine filter.
///
/// # Mathematical Specification
///
/// `out(x) = asin(in(x))`
///
/// Domain: `[−1, 1]`. Range: `[−π/2, π/2]`. Values outside `[−1, 1]` produce `NaN`.
///
/// # Properties
/// - `asin(0) = 0`, `asin(1) = π/2`, `asin(−1) = −π/2`.
/// - Odd function: `asin(−x) = −asin(x)`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::AsinImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AsinImageFilter;

impl AsinImageFilter {
    /// Construct a new `AsinImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the arcsine transform to every voxel.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::asin).collect();
        rebuild(out, dims, image)
    }
}

// ── AcosImageFilter ───────────────────────────────────────────────────────────

/// Pixelwise arccosine filter.
///
/// # Mathematical Specification
///
/// `out(x) = acos(in(x))`
///
/// Domain: `[−1, 1]`. Range: `[0, π]`. Values outside `[−1, 1]` produce `NaN`.
///
/// # Properties
/// - `acos(1) = 0`, `acos(0) = π/2`, `acos(−1) = π`.
/// - Complement relation: `asin(x) + acos(x) = π/2` for `x ∈ [−1, 1]`.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::AcosImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcosImageFilter;

impl AcosImageFilter {
    /// Construct a new `AcosImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the arccosine transform to every voxel.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals.into_iter().map(f32::acos).collect();
        rebuild(out, dims, image)
    }
}

// ── BoundedReciprocalImageFilter ──────────────────────────────────────────────

/// Pixelwise bounded-reciprocal filter.
///
/// # Mathematical Specification
///
/// `out(x) = 1 / (1 + |in(x)|)`
///
/// Range: `(0, 1]` for all finite inputs. Achieves maximum `1.0` at `in(x) = 0`.
///
/// # Properties
/// - Monotone decreasing on `[0, ∞)`: larger `|x|` → smaller output.
/// - At zero: `out(0) = 1`.
/// - Asymptote: `lim_{|x|→∞} out(x) = 0`.
/// - Bounded: `out(x) ∈ (0, 1]` for all finite inputs.
/// - Preserves spatial metadata.
/// - O(N) time and output space.
///
/// # References
/// - ITK `itk::BoundedReciprocalImageFilter<TInputImage, TOutputImage>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundedReciprocalImageFilter;

impl BoundedReciprocalImageFilter {
    /// Construct a new `BoundedReciprocalImageFilter`.
    pub fn new() -> Self {
        Self
    }

    /// Apply the bounded-reciprocal transform `1 / (1 + |x|)` to every voxel.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec(image);
        let out: Vec<f32> = vals
            .into_iter()
            .map(|x| 1.0_f32 / (1.0 + x.abs()))
            .collect();
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

    // ── AtanImageFilter ───────────────────────────────────────────────────────

    /// Proof: atan(0) = 0 exactly; atan(1) = π/4; atan(−1) = −π/4.
    #[test]
    fn atan_zero_is_zero() {
        let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let out = AtanImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert_eq!(x, 0.0f32, "atan(0) must equal 0 exactly");
        }
    }

    /// atan(1) = π/4 ≈ 0.7854.
    #[test]
    fn atan_one_is_pi_over_four() {
        let img = make_image(vec![1.0f32; 8], [2, 2, 2]);
        let out = AtanImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        let expected = (1.0f32).atan();
        for &x in v {
            assert!(
                (x - expected).abs() < 1e-6,
                "atan(1) = {x}, expected {expected}"
            );
        }
    }

    /// Odd function: atan(−x) = −atan(x).
    #[test]
    fn atan_odd_function() {
        let img = make_image(vec![2.5f32; 8], [2, 2, 2]);
        let pos = AtanImageFilter::new().apply(&img);
        let img_neg = make_image(vec![-2.5f32; 8], [2, 2, 2]);
        let neg = AtanImageFilter::new().apply(&img_neg);
        let pos_td = pos.data().clone().into_data();
        let neg_td = neg.data().clone().into_data();
        let pv: &[f32] = pos_td.as_slice::<f32>().unwrap();
        let nv: &[f32] = neg_td.as_slice::<f32>().unwrap();
        for (&p, &n) in pv.iter().zip(nv.iter()) {
            assert!((p + n).abs() < 1e-6, "atan not odd: {p} + {n} ≠ 0");
        }
    }

    /// Range is strictly within (−π/2, π/2).
    #[test]
    fn atan_range_bounded() {
        let vals: Vec<f32> = (-100..=100).map(|i| i as f32).collect();
        let n = vals.len();
        let img = make_image(vals, [1, 1, n]);
        let out = AtanImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        let pi_half = std::f32::consts::FRAC_PI_2;
        for &x in v {
            assert!(x > -pi_half && x < pi_half, "atan out of range: {x}");
        }
    }

    /// Spatial metadata is preserved exactly.
    #[test]
    fn atan_preserves_metadata() {
        let img = make_image(vec![1.0f32; 27], [3, 3, 3]);
        let out = AtanImageFilter::new().apply(&img);
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.direction(), img.direction());
        assert_eq!(out.shape(), img.shape());
    }

    // ── SinImageFilter ────────────────────────────────────────────────────────

    /// sin(0) = 0 exactly.
    #[test]
    fn sin_zero_is_zero() {
        let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let out = SinImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert!((x - 0.0f32).abs() < 1e-6, "sin(0) must equal 0");
        }
    }

    /// sin(π/2) = 1.0 exactly.
    #[test]
    fn sin_pi_over_2_is_one() {
        let pi_half = std::f32::consts::FRAC_PI_2;
        let img = make_image(vec![pi_half; 8], [2, 2, 2]);
        let out = SinImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert!((x - 1.0f32).abs() < 1e-6, "sin(π/2) = {x}");
        }
    }

    /// Output always in [−1, 1] for real inputs.
    #[test]
    fn sin_range_bounded() {
        let vals: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let n = vals.len();
        let img = make_image(vals, [1, 1, n]);
        let out = SinImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert!(
                x >= -1.0 - 1e-6 && x <= 1.0 + 1e-6,
                "sin out of [−1,1]: {x}"
            );
        }
    }

    // ── CosImageFilter ────────────────────────────────────────────────────────

    /// cos(0) = 1.0 exactly.
    #[test]
    fn cos_zero_is_one() {
        let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let out = CosImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert!((x - 1.0f32).abs() < 1e-6, "cos(0) = {x}");
        }
    }

    /// cos(π) = −1.0.
    #[test]
    fn cos_pi_is_minus_one() {
        let pi = std::f32::consts::PI;
        let img = make_image(vec![pi; 8], [2, 2, 2]);
        let out = CosImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert!((x - (-1.0f32)).abs() < 1e-5, "cos(π) = {x}");
        }
    }

    /// sin²(x) + cos²(x) = 1 (Pythagorean identity).
    #[test]
    fn sin_cos_pythagorean_identity() {
        let vals: Vec<f32> = (0..8)
            .map(|i| i as f32 * std::f32::consts::PI / 8.0)
            .collect();
        let img = make_image(vals.clone(), [2, 2, 2]);
        let img2 = make_image(vals, [2, 2, 2]);
        let sins = SinImageFilter::new().apply(&img);
        let coss = CosImageFilter::new().apply(&img2);
        let s_td = sins.data().clone().into_data();
        let c_td = coss.data().clone().into_data();
        let sv: &[f32] = s_td.as_slice::<f32>().unwrap();
        let cv: &[f32] = c_td.as_slice::<f32>().unwrap();
        for (&s, &c) in sv.iter().zip(cv.iter()) {
            let sum = s * s + c * c;
            assert!((sum - 1.0f32).abs() < 1e-5, "sin²+cos²={sum}");
        }
    }

    // ── TanImageFilter ────────────────────────────────────────────────────────

    /// tan(0) = 0 exactly.
    #[test]
    fn tan_zero_is_zero() {
        let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let out = TanImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert!((x - 0.0f32).abs() < 1e-6, "tan(0) = {x}");
        }
    }

    /// tan(π/4) = 1.0.
    #[test]
    fn tan_pi_over_4_is_one() {
        let pi_4 = std::f32::consts::FRAC_PI_4;
        let img = make_image(vec![pi_4; 8], [2, 2, 2]);
        let out = TanImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert!((x - 1.0f32).abs() < 1e-5, "tan(π/4) = {x}");
        }
    }

    // ── AsinImageFilter ───────────────────────────────────────────────────────

    /// asin(0) = 0, asin(1) = π/2, asin(−1) = −π/2.
    #[test]
    fn asin_boundary_values() {
        let pi_half = std::f32::consts::FRAC_PI_2;
        let img = make_image(
            vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0],
            [2, 2, 2],
        );
        let out = AsinImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        assert!((v[0] - 0.0f32).abs() < 1e-6, "asin(0) = {}", v[0]);
        assert!((v[1] - pi_half).abs() < 1e-5, "asin(1) = {}", v[1]);
        assert!((v[2] - (-pi_half)).abs() < 1e-5, "asin(-1) = {}", v[2]);
    }

    /// asin + acos = π/2 (complement identity).
    #[test]
    fn asin_acos_complement_identity() {
        let pi_half = std::f32::consts::FRAC_PI_2;
        let vals = vec![0.0f32, 0.5, -0.5, 0.8, -0.8, 1.0, -1.0, 0.0];
        let img = make_image(vals.clone(), [2, 2, 2]);
        let img2 = make_image(vals, [2, 2, 2]);
        let asins = AsinImageFilter::new().apply(&img);
        let acoss = AcosImageFilter::new().apply(&img2);
        let as_td = asins.data().clone().into_data();
        let ac_td = acoss.data().clone().into_data();
        let asv: &[f32] = as_td.as_slice::<f32>().unwrap();
        let acv: &[f32] = ac_td.as_slice::<f32>().unwrap();
        for (&a, &b) in asv.iter().zip(acv.iter()) {
            assert!((a + b - pi_half).abs() < 1e-5, "asin+acos={} ≠ π/2", a + b);
        }
    }

    // ── AcosImageFilter ───────────────────────────────────────────────────────

    /// acos(1) = 0, acos(0) = π/2, acos(−1) = π.
    #[test]
    fn acos_boundary_values() {
        let pi = std::f32::consts::PI;
        let pi_half = std::f32::consts::FRAC_PI_2;
        let img = make_image(vec![1.0f32, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0], [2, 2, 2]);
        let out = AcosImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        assert!((v[0] - 0.0f32).abs() < 1e-5, "acos(1) = {}", v[0]);
        assert!((v[1] - pi_half).abs() < 1e-5, "acos(0) = {}", v[1]);
        assert!((v[2] - pi).abs() < 1e-5, "acos(-1) = {}", v[2]);
    }

    // ── BoundedReciprocalImageFilter ──────────────────────────────────────────

    /// At zero: out = 1.0.
    #[test]
    fn bounded_reciprocal_zero_is_one() {
        let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let out = BoundedReciprocalImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert_eq!(x, 1.0f32, "1/(1+|0|) must equal 1.0");
        }
    }

    /// At x=1: out = 0.5. At x=−1: out = 0.5 (symmetric in |x|).
    #[test]
    fn bounded_reciprocal_at_one_is_half() {
        let img = make_image(
            vec![1.0f32, -1.0f32, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [2, 2, 2],
        );
        let out = BoundedReciprocalImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert!((x - 0.5f32).abs() < 1e-6, "1/(1+1) = {x}");
        }
    }

    /// Output strictly in (0, 1] for arbitrary inputs.
    #[test]
    fn bounded_reciprocal_range() {
        let vals: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
        let n = vals.len();
        let img = make_image(vals, [1, 1, n]);
        let out = BoundedReciprocalImageFilter::new().apply(&img);
        let td = out.data().clone().into_data();
        let v: &[f32] = td.as_slice::<f32>().unwrap();
        for &x in v {
            assert!(x > 0.0 && x <= 1.0, "bounded reciprocal out of (0,1]: {x}");
        }
    }

    /// Monotone decreasing: |x| < |y| ⟹ out(x) > out(y).
    #[test]
    fn bounded_reciprocal_monotone() {
        let img_small = make_image(vec![1.0f32; 8], [2, 2, 2]);
        let img_large = make_image(vec![10.0f32; 8], [2, 2, 2]);
        let out_small = BoundedReciprocalImageFilter::new().apply(&img_small);
        let out_large = BoundedReciprocalImageFilter::new().apply(&img_large);
        let s_td = out_small.data().clone().into_data();
        let l_td = out_large.data().clone().into_data();
        let sv: &[f32] = s_td.as_slice::<f32>().unwrap();
        let lv: &[f32] = l_td.as_slice::<f32>().unwrap();
        for (&s, &l) in sv.iter().zip(lv.iter()) {
            assert!(s > l, "bounded_reciprocal not monotone: {s} <= {l}");
        }
    }

    /// Spatial metadata preserved by BoundedReciprocal.
    #[test]
    fn bounded_reciprocal_preserves_metadata() {
        let img = make_image(vec![3.0f32; 27], [3, 3, 3]);
        let out = BoundedReciprocalImageFilter::new().apply(&img);
        assert_eq!(out.origin(), img.origin());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.direction(), img.direction());
        assert_eq!(out.shape(), img.shape());
    }
}
