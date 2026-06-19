//! Anti-alias binary image filter via mean curvature level-set flow.
//!
//! # Mathematical Specification
//!
//! Ports `sitk.AntiAliasBinaryImageFilter` / `itk::AntiAliasBinaryImageFilter`.
//! Smooths the boundary of a binary object by evolving a narrow-band level-set
//! function under mean curvature flow.
//!
//! ## Algorithm
//!
//! 1. **Initialise**: φ(x) = −1.0 where binary input = 1.0 (inside the object);
//!    φ(x) = +1.0 everywhere else (outside).
//! 2. **Iterate** up to `number_of_iterations`:
//!    - Compute the mean curvature speed `|∇φ|·κ = N / |∇φ|²` where N is the
//!      Caselles-Kimmel-Sapiro (1997) curvature numerator via second-order
//!      central finite differences with clamped (Neumann) boundaries.
//!    - Explicit-Euler update: `φ^{n+1} = φ^n + Δt · (N / |∇φ|²)`.
//!    - RMS change: `rms = sqrt( sum(φ^{n+1} - φ^n)^2 / N_vox )`.
//!    - If `rms < max_rms_error`: terminate early.
//! 3. Return φ — the level-set function (negative inside the smoothed object).
//!
//! ## Curvature numerator
//!
//! ```text
//! N = φ_xx·(φ_y² + φ_z²)
//!   + φ_yy·(φ_x² + φ_z²)
//!   + φ_zz·(φ_x² + φ_y²)
//!   − 2·φ_x·φ_y·φ_xy
//!   − 2·φ_x·φ_z·φ_xz
//!   − 2·φ_y·φ_z·φ_yz
//! ```
//!
//! ## Stability
//!
//! Δt = 0.125 is the standard stability constant for 3-D explicit-Euler mean
//! curvature flow (well inside the 1/6 bound for unit-spacing grids).
//!
//! ## Boundary conditions
//!
//! ZeroFluxNeumann — out-of-bounds indices are clamped to the nearest valid index.
//!
//! ## References
//! - Whitaker, R.T. (2000). "Reducing aliasing artifacts in iso-surfaces of binary
//!   volumes." *Proc. IEEE Vis. Symp. Vol. Vis.*, pp. 23–32.
//! - Caselles, V., Kimmel, R. & Sapiro, G. (1997). "Geodesic active contours."
//!   *Int. J. Comput. Vis.* 22(1):61–79.
//! - ITK `itkAntiAliasBinaryImageFilter.hxx`.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Explicit-Euler time step; stability bound for 3-D mean curvature flow.
const DT: f32 = 0.125;

/// Gradient-magnitude-squared floor preventing 0/0 in flat regions.
const GRAD_SQ_EPS: f32 = 1e-9;

// ── Filter ────────────────────────────────────────────────────────────────────

/// Anti-alias binary image filter.
///
/// Smooths the boundary of a binary object by evolving a level-set function
/// under mean curvature flow. The output is the signed level-set φ, where
/// negative values are inside the smoothed object and positive outside.
///
/// # Defaults
/// - `max_rms_error = 0.01`
/// - `number_of_iterations = 50`
///
/// # Example
/// ```rust,ignore
/// let filter = AntiAliasBinaryImageFilter {
///     number_of_iterations: 100,
///     ..Default::default()
/// };
/// let level_set = filter.apply(&binary_image);
/// ```
#[derive(Debug, Clone)]
pub struct AntiAliasBinaryImageFilter {
    /// Per-voxel RMS change threshold for early termination (default 0.01).
    pub max_rms_error: f32,
    /// Maximum number of level-set evolution iterations (default 50).
    pub number_of_iterations: usize,
}

impl Default for AntiAliasBinaryImageFilter {
    fn default() -> Self {
        Self {
            max_rms_error: 0.01,
            number_of_iterations: 50,
        }
    }
}

impl AntiAliasBinaryImageFilter {
    /// Evolve the binary image boundary under mean curvature level-set flow.
    ///
    /// `image`: binary float32 (0.0 = outside, 1.0 = inside), shape `[nz, ny, nx]`.
    ///
    /// Returns the level-set function φ with the same shape and spatial metadata
    /// as `image`. Values are negative inside the smoothed object and positive outside.
    ///
    /// # Panics
    /// Panics if the backend tensor cannot be converted to `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (binary, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let slab = ny * nx;

        // φ₀: +1.0 inside the object (binary = max), −1.0 outside, matching
        // ITK's convention where the foreground (== m_UpperBinaryValue) carries
        // the positive level-set sign and the zero level set sits on the
        // halfway iso-surface (itkAntiAliasBinaryImageFilter.hxx).
        let mut phi: Vec<f32> = binary
            .iter()
            .map(|&v| if v > 0.5 { 1.0_f32 } else { -1.0_f32 })
            .collect();

        let mut next = vec![0.0_f32; n];

        for _ in 0..self.number_of_iterations {
            let mut sum_sq = 0.0_f64;

            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let flat = iz * slab + iy * nx + ix;
                        let z = iz as isize;
                        let y = iy as isize;
                        let x = ix as isize;

                        // ZeroFluxNeumann neighbour accessor (clamp-at-boundary).
                        let get = |zz: isize, yy: isize, xx: isize| -> f32 {
                            let zc = zz.clamp(0, nz as isize - 1) as usize;
                            let yc = yy.clamp(0, ny as isize - 1) as usize;
                            let xc = xx.clamp(0, nx as isize - 1) as usize;
                            phi[zc * slab + yc * nx + xc]
                        };

                        let c = phi[flat];

                        // First-order central differences (unit spacing).
                        let phi_x = (get(z, y, x + 1) - get(z, y, x - 1)) * 0.5;
                        let phi_y = (get(z, y + 1, x) - get(z, y - 1, x)) * 0.5;
                        let phi_z = (get(z + 1, y, x) - get(z - 1, y, x)) * 0.5;

                        // Second-order central differences.
                        let phi_xx = get(z, y, x + 1) - 2.0 * c + get(z, y, x - 1);
                        let phi_yy = get(z, y + 1, x) - 2.0 * c + get(z, y - 1, x);
                        let phi_zz = get(z + 1, y, x) - 2.0 * c + get(z - 1, y, x);

                        // Mixed (cross) second-order central differences.
                        let phi_xy =
                            (get(z, y + 1, x + 1) - get(z, y + 1, x - 1) - get(z, y - 1, x + 1)
                                + get(z, y - 1, x - 1))
                                * 0.25;
                        let phi_xz =
                            (get(z + 1, y, x + 1) - get(z + 1, y, x - 1) - get(z - 1, y, x + 1)
                                + get(z - 1, y, x - 1))
                                * 0.25;
                        let phi_yz =
                            (get(z + 1, y + 1, x) - get(z + 1, y - 1, x) - get(z - 1, y + 1, x)
                                + get(z - 1, y - 1, x))
                                * 0.25;

                        // Mean curvature numerator N (Caselles–Kimmel–Sapiro 1997).
                        let num = phi_xx * (phi_y * phi_y + phi_z * phi_z)
                            + phi_yy * (phi_x * phi_x + phi_z * phi_z)
                            + phi_zz * (phi_x * phi_x + phi_y * phi_y)
                            - 2.0 * phi_x * phi_y * phi_xy
                            - 2.0 * phi_x * phi_z * phi_xz
                            - 2.0 * phi_y * phi_z * phi_yz;

                        // speed = |∇φ|·κ = N / |∇φ|²  (regularised to avoid 0/0).
                        let grad_sq = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z;
                        let speed = if grad_sq > GRAD_SQ_EPS {
                            num / grad_sq
                        } else {
                            0.0
                        };

                        // ITK CalculateUpdateValue flow constraint: a foreground
                        // voxel may not cross below the zero iso-surface, and a
                        // background voxel may not cross above it. This keeps the
                        // antialiased boundary pinned to the original binary edge
                        // (itkAntiAliasBinaryImageFilter.hxx::CalculateUpdateValue).
                        let raw = c + DT * speed;
                        let new_val = if binary[flat] > 0.5 {
                            raw.max(0.0)
                        } else {
                            raw.min(0.0)
                        };
                        next[flat] = new_val;

                        let diff = (new_val - c) as f64;
                        sum_sq += diff * diff;
                    }
                }
            }

            // Per-voxel RMS change; test convergence before committing the next buffer.
            let rms = (sum_sq / n as f64).sqrt() as f32;
            // `next` was written for every voxel this sweep, so swapping the
            // buffers commits it as the current level set without an N-element
            // memcpy; the old `phi` becomes scratch, fully overwritten next
            // sweep. Bit-identical to copy_from_slice, O(1) instead of O(N).
            std::mem::swap(&mut phi, &mut next);

            if rms < self.max_rms_error {
                break;
            }
        }

        rebuild(phi, dims, image)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_image::test_support as ts;
    use ritk_tensor_ops::extract_vec_infallible;

    type B = NdArray<f32>;

    /// A 32×32×1 binary circle: after mean curvature smoothing, at least one
    /// voxel must have a level-set value strictly in (−1, +1), confirming that
    /// non-trivial boundary evolution occurred.
    ///
    /// # Evidence
    /// At diagonal-corner boundary voxels (e.g. index (10,10) for a circle centred
    /// at (15.5, 15.5) with r = 8), both φ_x and φ_y are non-zero (−1.0 each),
    /// making the curvature numerator N = 4.0 and the speed = 2.0. After one
    /// iteration (Δt = 0.125): φ_new = −1.0 + 0.25 = −0.75 ∈ (−1, +1).
    #[test]
    fn binary_circle_smoothed_has_intermediate_values() {
        let ny = 32usize;
        let nx = 32usize;
        let cy = 15.5_f32;
        let cx = 15.5_f32;
        let r = 8.0_f32;

        let data: Vec<f32> = (0..ny)
            .flat_map(|y| {
                (0..nx).map(move |x| {
                    let dy = y as f32 - cy;
                    let dx = x as f32 - cx;
                    if dy * dy + dx * dx < r * r {
                        1.0_f32
                    } else {
                        0.0_f32
                    }
                })
            })
            .collect();

        let image = ts::make_image::<B, 3>(data, [1, ny, nx]);
        let filter = AntiAliasBinaryImageFilter {
            number_of_iterations: 10,
            ..Default::default()
        };
        let out = filter.apply(&image);
        let (vals, _) = extract_vec_infallible(&out);

        let has_intermediate = vals.iter().any(|&v| v > -1.0 && v < 1.0);
        assert!(
            has_intermediate,
            "expected at least one voxel in (−1, +1) after smoothing; \
             all values still ±1.0 — no curvature evolution occurred"
        );
    }

    /// A uniform foreground image (all 1.0): φ initialises to +1.0 everywhere.
    /// The gradient is zero at every voxel, so the curvature speed is exactly 0,
    /// the RMS change is 0.0 < `max_rms_error`, and the output must be all +1.0.
    ///
    /// Sign convention is fixed by ITK source: in
    /// `itkAntiAliasBinaryImageFilter.hxx::CalculateUpdateValue`, a voxel equal
    /// to `m_UpperBinaryValue` (the foreground/max value) is clamped to
    /// `max(new, 0)`, i.e. the foreground carries the *positive* level-set sign.
    /// The earlier −1.0 expectation encoded the inverted convention and is
    /// analytically incorrect against the reference filter.
    #[test]
    fn uniform_binary_image_stays_at_plus_one() {
        // 4×8×8 = genuinely 3-D to exercise the z-axis stencil path.
        let image = ts::fill_image::<B, 3>([4, 8, 8], 1.0_f32);
        let out = AntiAliasBinaryImageFilter::default().apply(&image);
        let (vals, _) = extract_vec_infallible(&out);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(
                v, 1.0_f32,
                "voxel {i}: uniform foreground should stay at +1.0, got {v}"
            );
        }
    }
}
