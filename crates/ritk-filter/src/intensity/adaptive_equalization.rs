//! Stark adaptive histogram equalization
//! (`itk::AdaptiveHistogramEqualizationImageFilter` / `sitk.AdaptiveHistogramEqualization`).
//!
//! # Mathematical Specification
//!
//! Stark's (2000) generalized local equalization, parameterized by `α, β ∈ [0, 1]`
//! that interpolate between classic adaptive histogram equalization
//! (`α = 0`) and unsharp masking / identity. Intensities are normalized to
//! `[−0.5, 0.5]` via the global range `iscale = max − min`:
//!
//! ```text
//! u(x) = (I(x) − min)/iscale − 0.5
//! cumf(u, v) = 0.5·sgn(u−v)·|2(u−v)|^α − 0.5·β·sgn(u−v)·|2(u−v)| + β·u
//! out(x) = iscale·( 0.5 + (1/k(x))·Σ_{y ∈ W(x)∩image} cumf(u(x), u(y)) ) + min
//! ```
//!
//! where `W(x)` is the box window of the given per-axis radius and `k(x)` is the
//! count of in-image window voxels (the window shrinks at the border, matching
//! ITK's boundary handling). `sgn(0) = 0`. Internal arithmetic is `f64`; the
//! result is float-exact to SimpleITK (output is `f32`).

use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Stark adaptive histogram equalization filter.
#[derive(Debug, Clone)]
pub struct AdaptiveHistogramEqualizationFilter {
    /// Box-window radius per tensor axis `[z, y, x]` (ITK/sitk default `[5, 5, 5]`).
    pub radius: [usize; 3],
    /// `α` — equalization exponent (sitk default `0.3`).
    pub alpha: f64,
    /// `β` — unsharp/identity blend (sitk default `0.3`).
    pub beta: f64,
}

impl Default for AdaptiveHistogramEqualizationFilter {
    fn default() -> Self {
        Self {
            radius: [5, 5, 5],
            alpha: 0.3,
            beta: 0.3,
        }
    }
}

/// ITK's `Math::sgn`: `-1`, `0`, or `1` (note `f64::signum` returns `±1` at zero).
#[inline]
fn sgn(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl AdaptiveHistogramEqualizationFilter {
    /// Construct with the given window radius (α, β default to `0.3`).
    pub fn new(radius: [usize; 3]) -> Self {
        Self {
            radius,
            ..Self::default()
        }
    }

    /// Apply the adaptive equalization.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let (rz, ry, rx) = (self.radius[0], self.radius[1], self.radius[2]);
        let (alpha, beta) = (self.alpha, self.beta);

        let (min_f32, max_f32) = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
            n,
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(mn, mx), i| {
                let v = vals[i];
                (mn.min(v), mx.max(v))
            },
            |(a_mn, a_mx), (b_mn, b_mx)| (a_mn.min(b_mn), a_mx.max(b_mx)),
        );
        let min = min_f32 as f64;
        let max = max_f32 as f64;
        let iscale = max - min;
        if iscale == 0.0 {
            // Constant image: the equalization is the identity.
            return Ok(rebuild(vals, dims, image));
        }
        let norm = |i: usize| (vals[i] as f64 - min) / iscale - 0.5;

        let cumf = |u: f64, v: f64| -> f64 {
            let s = sgn(u - v);
            let ad = (2.0 * (u - v)).abs();
            0.5 * s * ad.powf(alpha) - beta * 0.5 * s * ad + beta * u
        };

        let out: Vec<f32> = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |i| {
            let z = i / (ny * nx);
            let y = (i % (ny * nx)) / nx;
            let x = i % nx;
            let u = norm(i);
            let mut sum = 0.0f64;
            let mut k = 0u32;
            let z0 = z.saturating_sub(rz);
            let z1 = (z + rz).min(nz - 1);
            let y0 = y.saturating_sub(ry);
            let y1 = (y + ry).min(ny - 1);
            let x0 = x.saturating_sub(rx);
            let x1 = (x + rx).min(nx - 1);
            for zz in z0..=z1 {
                for yy in y0..=y1 {
                    let base = (zz * ny + yy) * nx;
                    for xx in x0..=x1 {
                        sum += cumf(u, norm(base + xx));
                        k += 1;
                    }
                }
            }
            (iscale * (sum / k as f64 + 0.5) + min) as f32
        });

        Ok(rebuild(out, dims, image))
    }
}

#[cfg(test)]
#[path = "tests_adaptive_equalization.rs"]
mod tests_adaptive_equalization;
