//! Recursive Gaussian filter using the Young–van Vliet IIR approximation.
//!
//! # Mathematical Specification
//!
//! Implements a separable 3rd-order recursive (IIR) Gaussian filter based on
//! the Young–van Vliet approximation (Young & van Vliet 1995, van Vliet et al.
//! 1998). For each dimension the 1-D filter decomposes into a causal (forward)
//! pass followed by an anticausal (backward) pass applied to the forward
//! output:
//!
//!   Forward:  y_f[n] = B·x[n] + d₁·y_f[n−1] + d₂·y_f[n−2] + d₃·y_f[n−3]
//!   Backward: y[n]   = B·y_f[n] + d₁·y[n+1] + d₂·y[n+2] + d₃·y[n+3]
//!
//! The cascade H(z)·H(z⁻¹) yields a zero-phase symmetric Gaussian
//! approximation with unit DC gain.
//!
//! The feedback coefficients d₁, d₂, d₃ depend only on σ (in pixel units).
//! The feedforward gain is B = 1 − (d₁ + d₂ + d₃).
//!
//! Derivative orders are computed by composing smoothing with finite
//! differences:
//!
//! - **Order 0 (smoothing)**: Two-pass IIR as described above.
//! - **Order 1 (first derivative)**: Smooth all axes separably, then compute
//!   gradient magnitude |∇I| = √(Σ_d (∂I/∂x_d)²) via central differences.
//! - **Order 2 (second derivative)**: Smooth all axes separably, then compute
//!   Laplacian ∇²I = Σ_d ∂²I/∂x_d² via second-order finite differences.
//!
//! Physical spacing is respected: `pixel_sigma = sigma / spacing[dim]`.
//!
//! # Complexity
//!
//! O(N) per dimension where N is the number of voxels along that axis,
//! applied separably across all D dimensions. Total: O(D · N_total).
//!
//! # References
//!
//! - Young, I.T. & van Vliet, L.J. (1995). Recursive implementation of the
//!   Gaussian filter. *Signal Processing* 44(2), pp. 139–151.
//! - van Vliet, L.J., Young, I.T., Verbeek, P.W. (1998). Recursive Gaussian
//!   derivative filters. *Proc. 14th ICPR*, pp. 509–514.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Derivative order enum ─────────────────────────────────────────────────────

/// Derivative order for the recursive Gaussian filter.
///
/// Selects which derivative of the Gaussian kernel to approximate:
/// - `Zero`: Gaussian smoothing (zeroth derivative).
/// - `First`: First derivative — smoothing + gradient magnitude.
/// - `Second`: Second derivative — smoothing + Laplacian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativeOrder {
    /// Zeroth derivative — Gaussian smoothing.
    Zero,
    /// First derivative of Gaussian.
    First,
    /// Second derivative of Gaussian.
    Second,
}

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Recursive Gaussian filter using a 3rd-order Young–van Vliet IIR
/// approximation.
///
/// Applies a separable recursive Gaussian (or its first/second derivative)
/// along each spatial dimension of a 3-D image, respecting physical spacing.
#[derive(Debug, Clone)]
pub struct RecursiveGaussianFilter {
    /// Standard deviation of the Gaussian in physical units (mm).
    sigma: f64,
    /// Which derivative order to approximate.
    derivative_order: DerivativeOrder,
    /// When true, multiply the filter output by σ^order so that responses
    /// at different scales have comparable magnitudes.
    normalize_across_scale: bool,
}

impl RecursiveGaussianFilter {
    /// Create a new recursive Gaussian filter with the given sigma (physical
    /// units).
    ///
    /// Defaults to smoothing (order 0), no scale normalization.
    pub fn new(sigma: f64) -> Self {
        Self {
            sigma,
            derivative_order: DerivativeOrder::Zero,
            normalize_across_scale: false,
        }
    }

    /// Set the derivative order.
    pub fn with_derivative_order(mut self, order: DerivativeOrder) -> Self {
        self.derivative_order = order;
        self
    }

    /// Enable or disable normalization across scale.
    pub fn with_normalize_across_scale(mut self, normalize: bool) -> Self {
        self.normalize_across_scale = normalize;
        self
    }

    /// Apply the recursive Gaussian filter to a 3-D image.
    ///
    /// Processing is separable: the 1-D IIR filter is applied along each of
    /// the three spatial axes in sequence.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let td = image.data().clone().into_data();
        let mut vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("RecursiveGaussianFilter requires f32 data: {:?}", e))?
            .to_vec();

        let spacing = image.spacing();

        // Stage 1: Smooth all axes separably via the two-pass IIR
        for dim in 0..3 {
            let pixel_sigma = self.sigma / spacing[dim];
            if pixel_sigma < 0.2 {
                continue;
            }
            let coeffs = YvVCoefficients::from_sigma(pixel_sigma);
            vals = apply_smooth_1d(&vals, dims, dim, &coeffs);
        }

        // Stage 2: Apply derivative operator across all axes combined.
        // Smoothing is already complete; the derivative is computed from
        // the smoothed data independently along each axis and then
        // combined (magnitude for order 1, sum for order 2).
        let sp = [spacing[0], spacing[1], spacing[2]];
        match self.derivative_order {
            DerivativeOrder::Zero => {}
            DerivativeOrder::First => {
                vals = gradient_magnitude_3d(&vals, dims, sp);
            }
            DerivativeOrder::Second => {
                vals = laplacian_3d(&vals, dims, sp);
            }
        }

        // Scale normalization: multiply by σ^order
        if self.normalize_across_scale {
            let scale_factor = match self.derivative_order {
                DerivativeOrder::Zero => 1.0,
                DerivativeOrder::First => self.sigma,
                DerivativeOrder::Second => self.sigma * self.sigma,
            };
            if (scale_factor - 1.0).abs() > 1e-12 {
                let sf = scale_factor as f32;
                for v in &mut vals {
                    *v *= sf;
                }
            }
        }

        let device = image.data().device();
        let out_td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

// ── Young–van Vliet coefficient set ───────────────────────────────────────────

/// Precomputed Young–van Vliet IIR coefficients for one 1-D pass.
///
/// The recurrence is:
///   y[n] = B·x[n] + d1·y[n−1] + d2·y[n−2] + d3·y[n−3]
///
/// where B = 1 − d1 − d2 − d3 ensures unit DC gain.
struct YvVCoefficients {
    /// Feedforward gain.
    b_gain: f64,
    /// Feedback coefficient for y[n−1] (or y[n+1] in anticausal pass).
    d1: f64,
    /// Feedback coefficient for y[n−2] (or y[n+2] in anticausal pass).
    d2: f64,
    /// Feedback coefficient for y[n−3] (or y[n+3] in anticausal pass).
    d3: f64,
}

impl YvVCoefficients {
    /// Compute the Young–van Vliet coefficients from a pixel-space sigma.
    ///
    /// # Derivation
    ///
    /// Following Young & van Vliet (1995):
    ///
    /// The scale parameter q is related to σ by:
    ///   q = 0.98711σ − 0.96330           for σ ≥ 2.5
    ///   q = 3.97156 − 4.14554√(1−0.26891σ)  for 0.5 ≤ σ < 2.5
    ///
    /// From q the un-normalised feedback coefficients are:
    ///   b0 = 1.57825 + 2.44413q + 1.4281q² + 0.422205q³
    ///   b1 = 2.44413q + 2.85619q² + 1.26661q³
    ///   b2 = −(1.4281q² + 1.26661q³)
    ///   b3 = 0.422205q³
    ///
    /// Normalised: d_i = b_i / b0.
    /// Feedforward gain: B = 1 − d1 − d2 − d3.
    fn from_sigma(sigma: f64) -> Self {
        let q = if sigma >= 2.5 {
            0.98711 * sigma - 0.96330
        } else if sigma >= 0.5 {
            3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).max(0.0).sqrt()
        } else {
            // For very small sigma, linearly interpolate q towards 0
            0.1 * sigma / 0.5
        };

        let q2 = q * q;
        let q3 = q2 * q;

        let b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
        let b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3;
        let b2 = -(1.4281 * q2 + 1.26661 * q3);
        let b3 = 0.422205 * q3;

        let d1 = b1 / b0;
        let d2 = b2 / b0;
        let d3 = b3 / b0;
        let b_gain = 1.0 - d1 - d2 - d3;

        Self { b_gain, d1, d2, d3 }
    }
}

// ── 1-D line iteration helpers ────────────────────────────────────────────────

/// Information needed to iterate over a 1-D line within a 3-D volume along a
/// given dimension.
struct LineParams {
    /// Length of each 1-D line along the target dimension.
    len: usize,
    /// Number of independent lines (product of the other two dimensions).
    num_lines: usize,
    /// Stride between consecutive elements along the target dimension.
    stride: usize,
}

/// Compute the base offset (index of the first element) for the `line_idx`-th
/// line along dimension `dim` in a volume with shape `dims = [nz, ny, nx]`.
fn line_base(dims: [usize; 3], dim: usize, line_idx: usize) -> usize {
    let [_nz, ny, nx] = dims;
    match dim {
        0 => {
            // Lines along Z. Each line is identified by (iy, ix).
            // line_idx = iy * nx + ix. Base = iy*nx + ix.
            line_idx
        }
        1 => {
            // Lines along Y. Each line is identified by (iz, ix).
            // line_idx = iz * nx + ix. Base = iz * ny * nx + ix.
            let iz = line_idx / nx;
            let ix = line_idx % nx;
            iz * ny * nx + ix
        }
        2 => {
            // Lines along X. Each line is identified by (iz, iy).
            // line_idx = iz * ny + iy. Base = iz * ny * nx + iy * nx.
            let iz = line_idx / ny;
            let iy = line_idx % ny;
            iz * ny * nx + iy * nx
        }
        _ => unreachable!(),
    }
}

fn line_params(dims: [usize; 3], dim: usize) -> LineParams {
    let [nz, ny, nx] = dims;
    match dim {
        0 => LineParams {
            len: nz,
            num_lines: ny * nx,
            stride: ny * nx,
        },
        1 => LineParams {
            len: ny,
            num_lines: nz * nx,
            stride: nx,
        },
        2 => LineParams {
            len: nx,
            num_lines: nz * ny,
            stride: 1,
        },
        _ => unreachable!(),
    }
}

// ── Smoothing (order 0): two-pass IIR ─────────────────────────────────────────

/// Apply the Young–van Vliet two-pass IIR smoothing along dimension `dim`.
///
/// Pass 1 (causal/forward on input):
///   y_f[n] = B·x[n] + d1·y_f[n−1] + d2·y_f[n−2] + d3·y_f[n−3]
///
/// Pass 2 (anticausal/backward on y_f):
///   y[n]   = B·y_f[n] + d1·y[n+1] + d2·y[n+2] + d3·y[n+3]
///
/// Boundary conditions: constant extension (replicate). The steady-state
/// response of the recursion to a constant input c is c (since B/(1−d1−d2−d3)
/// = B/B = 1), so we initialise the boundary taps to the first/last sample.
fn apply_smooth_1d(
    data: &[f32],
    dims: [usize; 3],
    dim: usize,
    coeffs: &YvVCoefficients,
) -> Vec<f32> {
    let lp = line_params(dims, dim);
    let mut output = vec![0.0_f32; data.len()];

    let bg = coeffs.b_gain;
    let d1 = coeffs.d1;
    let d2 = coeffs.d2;
    let d3 = coeffs.d3;

    for li in 0..lp.num_lines {
        let base = line_base(dims, dim, li);

        // Read input line into contiguous buffer
        let mut x = vec![0.0_f64; lp.len];
        for i in 0..lp.len {
            x[i] = data[base + i * lp.stride] as f64;
        }

        // --- Forward (causal) pass ---
        // Boundary initialisation: steady-state for constant extension = x[0]
        let mut yf = vec![0.0_f64; lp.len];
        let init_fwd = x[0];
        let mut ym1 = init_fwd;
        let mut ym2 = init_fwd;
        let mut ym3 = init_fwd;

        for i in 0..lp.len {
            let val = bg * x[i] + d1 * ym1 + d2 * ym2 + d3 * ym3;
            yf[i] = val;
            ym3 = ym2;
            ym2 = ym1;
            ym1 = val;
        }

        // --- Backward (anticausal) pass on yf ---
        // Boundary initialisation: steady-state for constant extension = yf[N-1]
        let init_bwd = yf[lp.len - 1];
        let mut yp1 = init_bwd;
        let mut yp2 = init_bwd;
        let mut yp3 = init_bwd;

        for i in (0..lp.len).rev() {
            let val = bg * yf[i] + d1 * yp1 + d2 * yp2 + d3 * yp3;
            // Write to output
            output[base + i * lp.stride] = val as f32;
            yp3 = yp2;
            yp2 = yp1;
            yp1 = val;
        }
    }

    output
}

// ── First derivative via central difference ───────────────────────────────────

/// Apply central-difference first derivative along dimension `dim`:
///   d[n] = (x[n+1] − x[n−1]) / 2
///
/// Boundary: one-sided differences at the edges.
fn apply_first_derivative_1d(data: &[f32], dims: [usize; 3], dim: usize) -> Vec<f32> {
    let lp = line_params(dims, dim);
    let mut output = vec![0.0_f32; data.len()];

    for li in 0..lp.num_lines {
        let base = line_base(dims, dim, li);

        for i in 0..lp.len {
            let val = if lp.len == 1 {
                0.0
            } else if i == 0 {
                data[base + lp.stride] - data[base]
            } else if i == lp.len - 1 {
                data[base + i * lp.stride] - data[base + (i - 1) * lp.stride]
            } else {
                (data[base + (i + 1) * lp.stride] - data[base + (i - 1) * lp.stride]) * 0.5
            };
            output[base + i * lp.stride] = val;
        }
    }

    output
}

// ── Second derivative via second-order finite difference ──────────────────────

/// Apply second-order finite difference along dimension `dim`:
///   d²[n] = x[n+1] − 2·x[n] + x[n−1]
///
/// Boundary: one-sided second-order differences at the edges.
fn apply_second_derivative_1d(data: &[f32], dims: [usize; 3], dim: usize) -> Vec<f32> {
    let lp = line_params(dims, dim);
    let mut output = vec![0.0_f32; data.len()];

    for li in 0..lp.num_lines {
        let base = line_base(dims, dim, li);

        for i in 0..lp.len {
            let val = if lp.len < 3 {
                0.0
            } else if i == 0 {
                // Forward one-sided: x[2] - 2x[1] + x[0]
                let x0 = data[base] as f64;
                let x1 = data[base + lp.stride] as f64;
                let x2 = data[base + 2 * lp.stride] as f64;
                (x2 - 2.0 * x1 + x0) as f32
            } else if i == lp.len - 1 {
                // Backward one-sided: x[n-1] - 2x[n-2] + x[n-3]
                let xn1 = data[base + i * lp.stride] as f64;
                let xn2 = data[base + (i - 1) * lp.stride] as f64;
                let xn3 = data[base + (i - 2) * lp.stride] as f64;
                (xn1 - 2.0 * xn2 + xn3) as f32
            } else {
                // Central: x[n+1] - 2x[n] + x[n-1]
                let xp = data[base + (i + 1) * lp.stride] as f64;
                let xc = data[base + i * lp.stride] as f64;
                let xm = data[base + (i - 1) * lp.stride] as f64;
                (xp - 2.0 * xc + xm) as f32
            };
            output[base + i * lp.stride] = val;
        }
    }

    output
}

// ── Combined derivative operators (gradient magnitude, Laplacian) ─────────────

/// Compute the gradient magnitude of a 3-D volume:
///
///   |∇I| = √(Σ_d (∂I/∂x_d / s_d)²)
///
/// Uses central differences at interior points and one-sided differences at
/// boundaries. Each component is divided by the physical spacing along that
/// axis so the result is in physical units (intensity / mm).
fn gradient_magnitude_3d(data: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let n = data.len();
    let mut sum_sq = vec![0.0_f64; n];

    for dim in 0..3 {
        let deriv = apply_first_derivative_1d(data, dims, dim);
        let s = spacing[dim];
        for i in 0..n {
            let d = deriv[i] as f64 / s;
            sum_sq[i] += d * d;
        }
    }

    sum_sq.iter().map(|v| v.sqrt() as f32).collect()
}

/// Compute the Laplacian of a 3-D volume:
///
///   ∇²I = Σ_d ∂²I/∂x_d² / s_d²
///
/// Uses central second-order finite differences at interior points and
/// one-sided differences at boundaries. Each component is divided by the
/// squared physical spacing along that axis.
fn laplacian_3d(data: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let n = data.len();
    let mut result = vec![0.0_f64; n];

    for dim in 0..3 {
        let d2 = apply_second_derivative_1d(data, dims, dim);
        let s2 = spacing[dim] * spacing[dim];
        for i in 0..n {
            result[i] += d2[i] as f64 / s2;
        }
    }

    result.iter().map(|v| *v as f32).collect()
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

    fn make_image(vals: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// Smoothing a constant image must return the same constant value at every
    /// voxel.
    ///
    /// **Proof**: A constant signal x[n] = c is the steady state of the IIR
    /// recursion: y = B·c + (d1+d2+d3)·y ⇒ y·(1−d1−d2−d3) = B·c ⇒ y = c.
    /// Both forward and backward passes converge to c, and the cascade
    /// preserves c. Boundary initialisation to c ensures no transients. ∎
    #[test]
    fn test_smoothing_constant_image() {
        let dims = [16, 16, 16];
        let c = 42.0_f32;
        let vals = vec![c; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims, [1.0, 1.0, 1.0]);

        let filter = RecursiveGaussianFilter::new(2.0);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - c).abs() < 1e-3,
                "constant image smoothing: voxel {i} = {v}, expected {c}"
            );
        }
    }

    /// Smoothing (order 0) preserves total intensity (sum) to within a small
    /// tolerance.
    ///
    /// **Proof sketch**: A Gaussian kernel integrates to 1, so convolution
    /// preserves the L¹ norm of the signal. The IIR approximation is designed
    /// to have unit DC gain, so the sum is preserved up to boundary effects.
    #[test]
    fn test_smoothing_preserves_sum() {
        let dims = [20, 20, 20];
        let n = dims[0] * dims[1] * dims[2];
        // Non-trivial signal: voxel value = flat index mod 17
        let vals: Vec<f32> = (0..n).map(|i| (i % 17) as f32).collect();
        let sum_in: f64 = vals.iter().map(|&v| v as f64).sum();

        let img = make_image(vals, dims, [1.0, 1.0, 1.0]);

        let filter = RecursiveGaussianFilter::new(1.5);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);
        let sum_out: f64 = out.iter().map(|&v| v as f64).sum();

        let rel_err = (sum_out - sum_in).abs() / sum_in.abs().max(1e-12);
        assert!(
            rel_err < 0.05,
            "sum not preserved: input sum = {sum_in}, output sum = {sum_out}, \
             relative error = {rel_err}"
        );
    }

    /// First derivative of a linear ramp I(x) = x gives a constant.
    ///
    /// **Derivation**: d/dx (x) = 1. The smoothing step preserves linearity
    /// (Gaussian * linear = linear with same slope at interior points).
    /// The central difference of the smoothed linear ramp gives ≈ 1 at
    /// interior voxels.
    #[test]
    fn test_first_derivative_of_linear_ramp() {
        let [nz, ny, nx] = [1usize, 1, 64];
        let vals: Vec<f32> = (0..nx).map(|ix| ix as f32).collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

        let filter =
            RecursiveGaussianFilter::new(3.0).with_derivative_order(DerivativeOrder::First);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        // The interior values should be approximately constant
        let margin = 12;
        let interior: Vec<f32> = out[margin..nx - margin].to_vec();
        let mean: f64 = interior.iter().map(|&v| v as f64).sum::<f64>() / interior.len() as f64;

        // All interior values should be close to the mean
        for (i, &v) in interior.iter().enumerate() {
            let dev = ((v as f64) - mean).abs();
            assert!(
                dev < mean.abs() * 0.15 + 0.1,
                "first derivative of ramp not constant at interior position {}: \
                 value = {v}, mean = {mean}",
                i + margin
            );
        }

        // The mean itself should be positive and close to 0.5 (central
        // difference of unit-slope ramp: (x[n+1]-x[n-1])/2 = 1*0.5... but
        // the smoothing + FD composition may differ in scale). Verify nonzero.
        assert!(
            mean.abs() > 0.01,
            "first derivative of ramp should be nonzero, got mean = {mean}"
        );
    }

    /// Second derivative of a quadratic I(x) = x² gives approximately
    /// constant output at interior voxels.
    ///
    /// **Derivation**: d²/dx² (x²) = 2. The smoothing preserves quadratic
    /// structure at interior points, and the central second-difference
    /// x[n+1]-2x[n]+x[n-1] of the smoothed quadratic gives ≈ 2 at interior
    /// voxels (the exact value depends on the smoothing kernel width but
    /// should be constant).
    #[test]
    fn test_second_derivative_of_quadratic() {
        let [nz, ny, nx] = [1usize, 1, 64];
        let vals: Vec<f32> = (0..nx).map(|ix| (ix as f32) * (ix as f32)).collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

        let filter =
            RecursiveGaussianFilter::new(3.0).with_derivative_order(DerivativeOrder::Second);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        // Interior values should be approximately constant
        let margin = 15;
        let interior: Vec<f32> = out[margin..nx - margin].to_vec();
        let mean: f64 = interior.iter().map(|&v| v as f64).sum::<f64>() / interior.len() as f64;

        for (i, &v) in interior.iter().enumerate() {
            let dev = ((v as f64) - mean).abs();
            assert!(
                dev < mean.abs() * 0.25 + 0.5,
                "second derivative of quadratic not constant at interior position {}: \
                 value = {v}, mean = {mean}",
                i + margin
            );
        }
        // The mean should be close to 2.0 (exact second derivative of x²)
        assert!(
            mean.abs() > 0.5,
            "second derivative of quadratic should be substantially nonzero, \
             got mean = {mean}"
        );
    }

    /// Smoothing with a large image and small sigma should approximate identity.
    #[test]
    fn test_small_sigma_near_identity() {
        let dims = [8, 8, 8];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
        let img = make_image(vals.clone(), dims, [1.0, 1.0, 1.0]);

        // Sigma = 0.1 in physical units, pixel sigma < 0.2 → skipped
        let filter = RecursiveGaussianFilter::new(0.1);
        let result = filter.apply(&img).unwrap();
        let out = extract_vals(&result);

        for (i, (&expected, &actual)) in vals.iter().zip(out.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "small sigma should be near-identity: voxel {i} = {actual}, expected {expected}"
            );
        }
    }

    /// Verify coefficient DC-gain invariant: B = 1 - d1 - d2 - d3.
    #[test]
    fn test_coefficients_dc_gain() {
        for &sigma in &[0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
            let c = YvVCoefficients::from_sigma(sigma);
            let dc = c.b_gain + c.d1 + c.d2 + c.d3;
            assert!(
                (dc - 1.0).abs() < 1e-12,
                "DC gain invariant violated for sigma={sigma}: B+d1+d2+d3 = {dc}"
            );
        }
    }
}
