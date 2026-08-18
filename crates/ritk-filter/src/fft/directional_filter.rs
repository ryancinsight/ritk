//! Directional 1-D frequency-domain filter over N-D images.
//!
//! Applies a transfer function along one axis of a real-valued N-D image,
//! using a 1-D FFT + multiply + IFFT pipeline. Unlike the radial
//! `FrequencyDomainFilter` (which uses H(r) over the full N-D spectrum),
//! this filter applies H(f_k) independently along one axis, leaving all
//! other axes unchanged.
//!
//! Typical use: bandpass the axial axis of an ultrasound RF image.
//!
//! # Normalisation
//!
//! Frequencies are normalised to `[0, 0.5]` before being passed to the
//! transfer function (`0.5` = Nyquist).
//!
//! # References
//!
//! - ITK `itkFFT1DImageFilter` — axis-indexed 1-D FFT pipeline.
//! - Butterworth, S. (1930). "On the theory of filter amplifiers."
//!   *Wireless Engineer* 7, 536–541.

use anyhow::{bail, Result};
use coeus_core::Backend;
use eunomia::Complex;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;
use ritk_tensor_ops::rebuild;

// ── Transfer function trait ───────────────────────────────────────────────────

/// Trait for 1-D directional frequency-response functions.
///
/// Receives normalised frequency `f ∈ [0, 0.5]` and returns the
/// transfer-function magnitude in `[0, 1]`.
pub trait DirectionalResponse: Send + Sync {
    /// Transfer-function magnitude at normalised frequency `f ∈ [0, 0.5]`.
    fn response(&self, f: f64) -> f32;
}

// ── Butterworth variants ──────────────────────────────────────────────────────

/// Butterworth bandpass: passes the band `(low_cutoff, high_cutoff)`.
///
/// H(f) = LP(f, high_cutoff) · (1 − LP(f, low_cutoff))
/// where LP(f, c) = 1 / (1 + (f/c)^(2n)).
///
/// Both cutoffs in `(0, 0.5]`, `order >= 1`.
#[derive(Debug, Clone, Copy)]
pub struct ButterworthBandpass {
    low_cutoff: f64,
    high_cutoff: f64,
    order: u32,
}

impl ButterworthBandpass {
    /// Construct a Butterworth bandpass filter.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid.
    pub fn new(low_cutoff: f64, high_cutoff: f64, order: u32) -> Result<Self> {
        if order == 0 {
            bail!("ButterworthBandpass: order must be >= 1");
        }
        if !low_cutoff.is_finite()
            || !high_cutoff.is_finite()
            || low_cutoff <= 0.0
            || high_cutoff > 0.5
            || low_cutoff >= high_cutoff
        {
            bail!(
                "ButterworthBandpass: need 0 < low < high <= 0.5, got [{}, {}]",
                low_cutoff,
                high_cutoff
            );
        }
        Ok(Self {
            low_cutoff,
            high_cutoff,
            order,
        })
    }

    fn lp(f: f64, c: f64, n: u32) -> f64 {
        1.0 / (1.0 + (f / c).powi(2 * n as i32))
    }
}

impl DirectionalResponse for ButterworthBandpass {
    fn response(&self, f: f64) -> f32 {
        (Self::lp(f, self.high_cutoff, self.order)
            * (1.0 - Self::lp(f, self.low_cutoff, self.order))) as f32
    }
}

/// Butterworth low-pass filter.
#[derive(Debug, Clone, Copy)]
pub struct ButterworthLowpass {
    cutoff: f64,
    order: u32,
}

impl ButterworthLowpass {
    /// Construct a Butterworth low-pass filter.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid.
    pub fn new(cutoff: f64, order: u32) -> Result<Self> {
        if order == 0 {
            bail!("ButterworthLowpass: order must be >= 1");
        }
        if !cutoff.is_finite() || cutoff <= 0.0 || cutoff > 0.5 {
            bail!("ButterworthLowpass: need 0 < cutoff <= 0.5, got {cutoff}");
        }
        Ok(Self { cutoff, order })
    }
}

impl DirectionalResponse for ButterworthLowpass {
    fn response(&self, f: f64) -> f32 {
        (1.0 / (1.0 + (f / self.cutoff).powi(2 * self.order as i32))) as f32
    }
}

/// Butterworth high-pass filter.
#[derive(Debug, Clone, Copy)]
pub struct ButterworthHighpass {
    cutoff: f64,
    order: u32,
}

impl ButterworthHighpass {
    /// Construct a Butterworth high-pass filter.
    ///
    /// # Errors
    /// Returns an error when inputs are invalid.
    pub fn new(cutoff: f64, order: u32) -> Result<Self> {
        if order == 0 {
            bail!("ButterworthHighpass: order must be >= 1");
        }
        if !cutoff.is_finite() || cutoff <= 0.0 || cutoff > 0.5 {
            bail!("ButterworthHighpass: need 0 < cutoff <= 0.5, got {cutoff}");
        }
        Ok(Self { cutoff, order })
    }
}

impl DirectionalResponse for ButterworthHighpass {
    fn response(&self, f: f64) -> f32 {
        (1.0 - 1.0 / (1.0 + (f / self.cutoff).powi(2 * self.order as i32))) as f32
    }
}

// ── Directional filter ────────────────────────────────────────────────────────

/// Apply a directional 1-D frequency-domain filter along one image axis.
///
/// For each 1-D line along `axis`:
/// 1. Forward FFT via `apollo_fft`.
/// 2. Multiply bin `k` by `response.response(k / n)`.
/// 3. Inverse FFT.
///
/// # Errors
///
/// Returns an error when `axis >= D` or the axis has length 0.
pub fn apply_directional_filter<R, B, const D: usize>(
    image: &Image<f32, B, D>,
    axis: usize,
    response: &R,
) -> Result<Image<f32, B, D>>
where
    R: DirectionalResponse,
    B: Backend,
{
    let dims: [usize; D] = image.shape();
    if axis >= D {
        bail!("apply_directional_filter: axis {axis} >= D ({D})");
    }
    let n = dims[axis];
    if n == 0 {
        bail!("apply_directional_filter: axis {axis} has length 0");
    }

    // Precompute weights for each bin k = 0..n.
    // For a length-n FFT: bin k → normalised frequency k/n for k <= n/2,
    // and (n-k)/n for k > n/2 (negative half-plane, same magnitude).
    let weights: Vec<f32> = (0..n)
        .map(|k| {
            let f = if k <= n / 2 {
                k as f64 / n as f64
            } else {
                (n - k) as f64 / n as f64
            };
            response.response(f)
        })
        .collect();

    let (flat, _) = extract_vec(image)?;
    let total: usize = dims.iter().product();
    debug_assert_eq!(flat.len(), total);

    let mut output = flat.clone();

    // Row-major strides for each dimension.
    let mut strides = [0usize; D];
    strides[D - 1] = 1;
    for i in (0..D - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    let axis_stride = strides[axis];

    // Number of 1-D lines (product of all orthogonal dimensions).
    let n_lines: usize = total / n;

    // For each line, we need the flat base offset into the image.
    // We enumerate over the orthogonal axes to compute this.
    let ortho_dims: Vec<usize> = (0..D).filter(|&a| a != axis).map(|a| dims[a]).collect();
    let ortho_strides: Vec<usize> = (0..D).filter(|&a| a != axis).map(|a| strides[a]).collect();
    let n_ortho = ortho_dims.len();

    let mut line = vec![Complex::new(0.0_f32, 0.0_f32); n];
    let norm = 1.0_f32 / n as f32;

    for line_idx in 0..n_lines {
        // Decode line_idx into per-axis coordinates for the orthogonal axes.
        let mut base = 0usize;
        let mut remaining = line_idx;
        for oi in 0..n_ortho {
            let stride_ortho: usize = if oi + 1 < n_ortho {
                ortho_dims[oi + 1..].iter().product()
            } else {
                1
            };
            let coord = remaining / stride_ortho;
            remaining %= stride_ortho;
            base += coord * ortho_strides[oi];
        }

        // Collect the 1-D line into a complex buffer.
        for k in 0..n {
            line[k] = Complex::new(flat[base + k * axis_stride], 0.0);
        }

        // Forward FFT in-place.
        apollo_fft::application::execution::kernel::fft_forward(&mut line);

        // Apply frequency weights.
        for k in 0..n {
            let w = weights[k];
            line[k] = Complex::new(line[k].re * w, line[k].im * w);
        }

        // Inverse FFT in-place (unnormalized).
        apollo_fft::application::execution::kernel::fft_inverse_unnorm(&mut line);

        // Write real part back, normalised by 1/n.
        for k in 0..n {
            output[base + k * axis_stride] = line[k].re * norm;
        }
    }

    Ok(rebuild(output, dims, image))
}
