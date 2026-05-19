//! Inverse Fast Fourier Transform filter.
//!
//! Transforms a frequency-domain complex image back to its spatial representation.
//! This is the inverse of [`super::forward::ForwardFftFilter`].
//!
//! # Mathematical specification
//!
//! For a 2-D complex image F(u, v) with dimensions (H, W):
//!
//! ```text
//! f(x, y) = (1 / H·W) · Σ_{u=0}^{H-1} Σ_{v=0}^{W-1} F(u,v) · e^{+2πi(ux/H + vy/W)}
//! ```
//!
//! The transform is applied separably: 1-D IFFT along rows, then along columns.
//! For 3-D images an additional 1-D IFFT is applied along the depth axis.
//!
//! `rustfft` computes the unnormalized IFFT:
//!
//! ```text
//! IFFT_unnorm(F)[n] = Σ_{k} F[k] · e^{+2πi·k·n/N}
//! ```
//!
//! All IFFT passes are completed first; a single normalization by `1/N`
//! (N = product of all spatial dimensions) is applied afterwards.  This
//! satisfies the round-trip identity `inverse(forward(f)) ≈ f` to within
//! f32 rounding error.
//!
//! # Input format (shared with ForwardFftFilter)
//!
//! Complex images are stored with interleaved (Re, Im) pairs in the last
//! dimension:
//!
//! - 2-D input shape `[H, 2·W]`:
//!   Re at flat index `r·2W + 2c`, Im at `r·2W + 2c + 1`
//! - 3-D input shape `[D, H, 2·W]`:
//!   Re at `d·H·2W + r·2W + 2c`, Im at `+1`

use crate::filter::ops::{extract_vec, rebuild};
use crate::image::Image;
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use rustfft::{num_complex::Complex, FftPlanner};

// ── Struct ────────────────────────────────────────────────────────────────────

/// Inverse Fast Fourier Transform filter.
///
/// Transforms a complex-valued frequency-domain image (produced by
/// [`super::forward::ForwardFftFilter`]) back to the spatial domain.
/// Spatial metadata (origin, spacing, direction) is preserved from the
/// complex input image.
///
/// # Output
///
/// - 2-D: shape `[H, W]`, real-valued, normalized by `1/(H·W)`.
/// - 3-D: shape `[D, H, W]`, real-valued, normalized by `1/(D·H·W)`.
///
/// # Complexity
///
/// O(N log N) where N = product of spatial dimensions.
pub struct InverseFftFilter;

// ── impl ──────────────────────────────────────────────────────────────────────

impl InverseFftFilter {
    /// Create a new `InverseFftFilter`.
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Apply inverse FFT to a 2-D complex image.
    ///
    /// # Input
    ///
    /// Shape `[H, 2·W]` — Re at `r·2W + 2c`, Im at `r·2W + 2c + 1`.
    ///
    /// # Output
    ///
    /// Shape `[H, W]` — real-valued spatial image, normalized by `1/(H·W)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the last dimension is odd (not a valid complex
    /// interleaved layout) or when the backend tensor cannot be converted to
    /// `f32`.
    pub fn apply_2d<B: Backend>(&self, image: &Image<B, 2>) -> Result<Image<B, 2>> {
        let [h, cw] = image.shape();
        if cw % 2 != 0 {
            return Err(anyhow!(
                "InverseFftFilter: 2D input last dimension must be even \
                 (got cw={}); expected interleaved complex layout [H, 2*W]",
                cw
            ));
        }
        let w = cw / 2;

        let (vals, _) = extract_vec(image)?;
        // vals has length h * cw = h * 2 * w, row-major.

        // Build complex buffer: buf[r*w + c] = F[r, c].
        let mut buf: Vec<Complex<f32>> = Vec::with_capacity(h * w);
        for r in 0..h {
            for c in 0..w {
                buf.push(Complex::new(vals[r * cw + 2 * c], vals[r * cw + 2 * c + 1]));
            }
        }

        let mut planner = FftPlanner::<f32>::new();

        // Row-wise IFFT along the W axis (length w per row).
        let ifft_row = planner.plan_fft_inverse(w);
        for r in 0..h {
            ifft_row.process(&mut buf[r * w..(r + 1) * w]);
        }

        // Column-wise IFFT along the H axis (length h per column).
        let ifft_col = planner.plan_fft_inverse(h);
        let mut col_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); h];
        for c in 0..w {
            for r in 0..h {
                col_buf[r] = buf[r * w + c];
            }
            ifft_col.process(&mut col_buf);
            for r in 0..h {
                buf[r * w + c] = col_buf[r];
            }
        }

        // Normalize after all IFFT passes.
        // rustfft's IFFT is unnormalized; the factor 1/(H*W) accounts for
        // both the row pass (factor W) and column pass (factor H).
        let scale = 1.0 / (h * w) as f32;
        let out: Vec<f32> = buf.iter().map(|z| z.re * scale).collect();

        Ok(rebuild(out, [h, w], image))
    }

    /// Apply inverse FFT to a 3-D complex image.
    ///
    /// # Input
    ///
    /// Shape `[D, H, 2·W]` — Re at `d·H·2W + r·2W + 2c`, Im at `+1`.
    ///
    /// # Output
    ///
    /// Shape `[D, H, W]` — real-valued spatial volume, normalized by
    /// `1/(D·H·W)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the last dimension is odd or the backend tensor
    /// cannot be converted to `f32`.
    pub fn apply_3d<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let [depth, h, cw] = image.shape();
        if cw % 2 != 0 {
            return Err(anyhow!(
                "InverseFftFilter: 3D input last dimension must be even \
                 (got cw={}); expected interleaved complex layout [D, H, 2*W]",
                cw
            ));
        }
        let w = cw / 2;

        let (vals, _) = extract_vec(image)?;
        // vals has length depth * h * cw = depth * h * 2 * w.

        // Build complex buffer: buf[d*h*w + r*w + c] = F[d, r, c].
        let mut buf: Vec<Complex<f32>> = Vec::with_capacity(depth * h * w);
        for d in 0..depth {
            for r in 0..h {
                for c in 0..w {
                    let src = d * h * cw + r * cw + 2 * c;
                    buf.push(Complex::new(vals[src], vals[src + 1]));
                }
            }
        }

        let mut planner = FftPlanner::<f32>::new();

        // Row-wise IFFT per depth slice (along the W axis, length w).
        let ifft_row = planner.plan_fft_inverse(w);
        for d in 0..depth {
            for r in 0..h {
                let start = d * h * w + r * w;
                ifft_row.process(&mut buf[start..start + w]);
            }
        }

        // Column-wise IFFT per depth slice (along the H axis, length h).
        let ifft_col = planner.plan_fft_inverse(h);
        let mut col_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); h];
        for d in 0..depth {
            for c in 0..w {
                for r in 0..h {
                    col_buf[r] = buf[d * h * w + r * w + c];
                }
                ifft_col.process(&mut col_buf);
                for r in 0..h {
                    buf[d * h * w + r * w + c] = col_buf[r];
                }
            }
        }

        // Depth-wise IFFT for each (r, c) pair (along the D axis, length depth).
        let ifft_depth = planner.plan_fft_inverse(depth);
        let mut depth_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); depth];
        for r in 0..h {
            for c in 0..w {
                for d in 0..depth {
                    depth_buf[d] = buf[d * h * w + r * w + c];
                }
                ifft_depth.process(&mut depth_buf);
                for d in 0..depth {
                    buf[d * h * w + r * w + c] = depth_buf[d];
                }
            }
        }

        // Normalize after all IFFT passes: 1/(D*H*W).
        let scale = 1.0 / (depth * h * w) as f32;
        let out: Vec<f32> = buf.iter().map(|z| z.re * scale).collect();

        Ok(rebuild(out, [depth, h, w], image))
    }
}

impl Default for InverseFftFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_inverse.rs"]
mod tests;
