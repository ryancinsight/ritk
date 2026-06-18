//! Laplacian sharpening filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Ports `itk::LaplacianSharpeningImageFilter`. The output sharpens the input by
//! subtracting its Laplacian after rescaling the Laplacian into the input's
//! intensity range, then restores the original mean and clamps to the input
//! range. All intermediate computation is in `f64` (ITK's `RealType` for a
//! floating-point input), matching the reference exactly.
//!
//! Let `I` be the input, `L = ∇²I` its Laplacian (ZeroFluxNeumann boundary; the
//! axis stencil is `[1, −2, 1]·s_a²`, with `s_a = 1/spacing_a` when
//! `use_image_spacing`, else `s_a = 1`). With
//!
//! ```text
//! i_shift = min I,   i_scale = max I − min I,
//! f_shift = min L,   f_scale = max L − min L,
//! ```
//!
//! the combined image is
//!
//! ```text
//! C = I − ( (L − f_shift)·(i_scale / f_scale) + i_shift )
//! ```
//!
//! and the output restores the mean and clamps to the input range:
//!
//! ```text
//! O = clamp( C − mean(C) + mean(I),  min I,  max I ).
//! ```
//!
//! # ITK parity
//!
//! Corresponds to `itk::LaplacianSharpeningImageFilter` with default
//! `UseImageSpacing = true`. The Laplacian operator is ITK's `LaplacianOperator`
//! with derivative scalings `1/spacing` (so the axis coefficient is `1/spacing²`)
//! under `ZeroFluxNeumannBoundaryCondition`.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Laplacian sharpening filter (`itk::LaplacianSharpeningImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct LaplacianSharpeningFilter {
    /// Divide each axis second-derivative by `spacing²` when `true` (ITK default),
    /// else use unit scalings.
    pub use_image_spacing: bool,
}

impl Default for LaplacianSharpeningFilter {
    fn default() -> Self {
        Self {
            use_image_spacing: true,
        }
    }
}

impl LaplacianSharpeningFilter {
    /// Construct with explicit spacing handling.
    pub fn new(use_image_spacing: bool) -> Self {
        Self { use_image_spacing }
    }

    /// Apply Laplacian sharpening to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let n = vals.len();

        // Per-axis inverse-spacing-squared scalings (s_a²) for the Laplacian.
        let inv2 = |s: f64| 1.0 / (s * s);
        let scal = if self.use_image_spacing {
            [
                inv2(image.spacing()[0]),
                inv2(image.spacing()[1]),
                inv2(image.spacing()[2]),
            ]
        } else {
            [1.0, 1.0, 1.0]
        };

        // Input statistics in f64.
        let mut i_min = f64::INFINITY;
        let mut i_max = f64::NEG_INFINITY;
        let mut i_sum = 0.0f64;
        for &v in &vals {
            let v = v as f64;
            i_min = i_min.min(v);
            i_max = i_max.max(v);
            i_sum += v;
        }
        let i_mean = i_sum / n as f64;
        let i_shift = i_min;
        let i_scale = i_max - i_min;

        // Laplacian in f64 (ZeroFluxNeumann), tracking its own min/max.
        let lap = laplacian_f64(&vals, dims, scal);
        let mut f_min = f64::INFINITY;
        let mut f_max = f64::NEG_INFINITY;
        for &l in &lap {
            f_min = f_min.min(l);
            f_max = f_max.max(l);
        }
        let f_shift = f_min;
        let f_scale = f_max - f_shift;
        let gain = i_scale / f_scale;

        // Combined image and its mean.
        let mut combined = vec![0.0f64; n];
        let mut c_sum = 0.0f64;
        for i in 0..n {
            let c = vals[i] as f64 - ((lap[i] - f_shift) * gain + i_shift);
            combined[i] = c;
            c_sum += c;
        }
        let c_mean = c_sum / n as f64;

        // Restore mean and clamp to the input range.
        let out: Vec<f32> = combined
            .iter()
            .map(|&c| (c - c_mean + i_mean).clamp(i_min, i_max) as f32)
            .collect();
        rebuild(out, dims, image)
    }
}

/// Discrete Laplacian `Σ_a s_a²·(I[+a] − 2I + I[−a])` in `f64` with
/// ZeroFluxNeumann boundary (out-of-range neighbour clamped to the edge), `scal`
/// the per-axis `s_a²` in `[z, y, x]` order. Mirrors `LaplacianFilter` but in the
/// `RealType = f64` precision `LaplacianSharpeningImageFilter` computes in.
fn laplacian_f64(data: &[f32], dims: [usize; 3], scal: [f64; 3]) -> Vec<f64> {
    let [nz, ny, nx] = dims;
    let mut out = vec![0.0f64; nz * ny * nx];
    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);
                let center = data[flat] as f64;
                let (zlo, zhi) = (iz.saturating_sub(1), (iz + 1).min(nz - 1));
                let (ylo, yhi) = (iy.saturating_sub(1), (iy + 1).min(ny - 1));
                let (xlo, xhi) = (ix.saturating_sub(1), (ix + 1).min(nx - 1));
                let d2z = (data[idx(zhi, iy, ix)] as f64 - 2.0 * center
                    + data[idx(zlo, iy, ix)] as f64)
                    * scal[0];
                let d2y = (data[idx(iz, yhi, ix)] as f64 - 2.0 * center
                    + data[idx(iz, ylo, ix)] as f64)
                    * scal[1];
                let d2x = (data[idx(iz, iy, xhi)] as f64 - 2.0 * center
                    + data[idx(iz, iy, xlo)] as f64)
                    * scal[2];
                out[flat] = d2z + d2y + d2x;
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "tests_laplacian_sharpening.rs"]
mod tests_laplacian_sharpening;
