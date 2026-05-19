//! Forward FFT filter.
//!
//! Transforms a real-valued image to its complex frequency-domain representation.
//!
//! # Mathematical specification
//!
//! For a 2-D image f of shape [H, W], the forward DFT is:
//!
//!   F(u, v) = Σ_{x=0}^{H-1} Σ_{y=0}^{W-1} f(x,y) · e^{-2πi(ux/H + vy/W)}
//!
//! Applied separably: row-wise DFT first, then column-wise DFT.
//! No normalization is applied in the forward direction (ITK convention).
//!
//! # Output format
//!
//! For a 2-D input of shape [H, W], the output is shape [H, 2*W]:
//!   data[r * 2*W + 2*c]     = Re(F[r, c])
//!   data[r * 2*W + 2*c + 1] = Im(F[r, c])
//!
//! For a 3-D input of shape [D, H, W], the output is shape [D, H, 2*W]:
//!   data[d*H*2*W + r*2*W + 2*c]     = Re(F[d, r, c])
//!   data[d*H*2*W + r*2*W + 2*c + 1] = Im(F[d, r, c])
//!
//! # Complexity
//! O(N log N), N = product of image dimensions.

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::Result;
use burn::tensor::backend::Backend;
use rustfft::{num_complex::Complex, FftPlanner};

/// Forward Fast Fourier Transform filter.
///
/// Transforms a real-valued image into its frequency-domain complex representation.
/// Preserves spatial metadata (origin, spacing, direction) from the source image.
///
/// # Output layout
///
/// 2-D input `[H, W]` → output `[H, 2*W]`: each row contains `W` interleaved `(Re, Im)` pairs.
/// 3-D input `[D, H, W]` → output `[D, H, 2*W]`: same layout per depth slice.
///
/// DC (zero-frequency) component is at index `[0, 0]`.
/// Use [`super::shift::FftShiftFilter`] to move it to the centre.
///
/// # Complexity
/// O(N log N), N = product of image dimensions.
pub struct ForwardFftFilter;

impl ForwardFftFilter {
    /// Create a new `ForwardFftFilter`.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Apply forward FFT to a 2-D real image.
    ///
    /// Input shape `[H, W]` → output shape `[H, 2*W]`.
    ///
    /// At row `r` and frequency column `c`:
    ///   `out[r * 2*W + 2*c]     = Re(F[r, c])`
    ///   `out[r * 2*W + 2*c + 1] = Im(F[r, c])`
    pub fn apply_2d<B: Backend>(&self, image: &Image<B, 2>) -> Result<Image<B, 2>> {
        let [h, w] = image.shape();
        let (vals, _) = extract_vec(image)?;

        // Load real data into complex buffer; imaginary parts are zero.
        let mut buf: Vec<Complex<f32>> = vals.into_iter().map(|v| Complex::new(v, 0.0)).collect();

        let mut planner = FftPlanner::<f32>::new();

        // Step 1: Row-wise DFT. Each row occupies a contiguous slice of length w.
        let fft_row = planner.plan_fft_forward(w);
        for r in 0..h {
            fft_row.process(&mut buf[r * w..(r + 1) * w]);
        }

        // Step 2: Column-wise DFT. Gather each column into a temporary buffer,
        // transform it, and scatter the result back.
        let fft_col = planner.plan_fft_forward(h);
        let mut col_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); h];
        for c in 0..w {
            for r in 0..h {
                col_buf[r] = buf[r * w + c];
            }
            fft_col.process(&mut col_buf);
            for r in 0..h {
                buf[r * w + c] = col_buf[r];
            }
        }

        // Interleave (Re, Im) pairs into a flat output buffer; output shape [h, 2*w].
        let mut out = Vec::with_capacity(h * 2 * w);
        for r in 0..h {
            for c in 0..w {
                out.push(buf[r * w + c].re);
                out.push(buf[r * w + c].im);
            }
        }

        Ok(rebuild(out, [h, 2 * w], image))
    }

    /// Apply forward FFT to a 3-D real image.
    ///
    /// Input shape `[D, H, W]` → output shape `[D, H, 2*W]`.
    ///
    /// At depth `d`, row `r`, and frequency column `c`:
    ///   `out[d*H*2*W + r*2*W + 2*c]     = Re(F[d, r, c])`
    ///   `out[d*H*2*W + r*2*W + 2*c + 1] = Im(F[d, r, c])`
    ///
    /// The separable transform is applied row-wise, then column-wise (within each
    /// depth slice), then depth-wise (across all slices at each (row, col) position).
    pub fn apply_3d<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let [depth, h, w] = image.shape();
        let (vals, _) = extract_vec(image)?;

        let slice_size = h * w;
        let mut buf: Vec<Complex<f32>> = vals.into_iter().map(|v| Complex::new(v, 0.0)).collect();

        let mut planner = FftPlanner::<f32>::new();

        // Step 1: Row-wise DFT over all (depth, row) pairs.
        let fft_row = planner.plan_fft_forward(w);
        for d in 0..depth {
            for r in 0..h {
                let start = d * slice_size + r * w;
                fft_row.process(&mut buf[start..start + w]);
            }
        }

        // Step 2: Column-wise DFT within each depth slice.
        let fft_col = planner.plan_fft_forward(h);
        let mut col_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); h];
        for d in 0..depth {
            for c in 0..w {
                for r in 0..h {
                    col_buf[r] = buf[d * slice_size + r * w + c];
                }
                fft_col.process(&mut col_buf);
                for r in 0..h {
                    buf[d * slice_size + r * w + c] = col_buf[r];
                }
            }
        }

        // Step 3: Depth-wise DFT over each (row, col) position across all slices.
        let fft_depth = planner.plan_fft_forward(depth);
        let mut depth_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); depth];
        for r in 0..h {
            for c in 0..w {
                for d in 0..depth {
                    depth_buf[d] = buf[d * slice_size + r * w + c];
                }
                fft_depth.process(&mut depth_buf);
                for d in 0..depth {
                    buf[d * slice_size + r * w + c] = depth_buf[d];
                }
            }
        }

        // Interleave (Re, Im) pairs; output shape [depth, h, 2*w].
        let mut out = Vec::with_capacity(depth * h * 2 * w);
        for d in 0..depth {
            for r in 0..h {
                for c in 0..w {
                    out.push(buf[d * slice_size + r * w + c].re);
                    out.push(buf[d * slice_size + r * w + c].im);
                }
            }
        }

        Ok(rebuild(out, [depth, h, 2 * w], image))
    }
}

impl Default for ForwardFftFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests_forward.rs"]
mod tests;
