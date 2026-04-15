//! Sato multi-scale line filter for curvilinear structure detection.
//!
//! # Mathematical Specification
//!
//! **Reference:** Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
//! Koller, T., Gerig, G. & Kikinis, R. (1998). Three-dimensional multi-scale line
//! filter for segmentation and visualization of curvilinear structures in medical
//! images. *Medical Image Analysis* 2(2):143–168.
//!
//! At each Gaussian scale σ the normalised Hessian `H_σ = σ² · H(I_σ)` is
//! computed, where `I_σ = G_σ ∗ I`.  The three eigenvalues
//! `λ₁, λ₂, λ₃` are sorted by absolute value so that `|λ₁| ≤ |λ₂| ≤ |λ₃|`.
//!
//! For a bright tubular structure on a dark background the expected pattern is:
//!
//! ```text
//!   λ₁ ≈ 0,  λ₂ < 0,  λ₃ < 0   (two strongly negative, one near zero)
//! ```
//!
//! **Line response function** (Sato 1998, eq. 5–7):
//!
//! If `λ₂ < 0` AND `λ₃ < 0`:
//!
//! ```text
//!   V(λ₁,λ₂,λ₃) = |λ₃| · (λ₂/λ₃)^α · f(λ₁,λ₂)
//!
//!   where  f(λ₁,λ₂) = 1                                  if λ₁ ≤ 0
//!                     = exp(−λ₁² / (2·(α·λ₂)²))           if λ₁ > 0
//! ```
//!
//! `α` (default 0.5) controls cross-section anisotropy tolerance.
//! Higher α → more permissive to elliptical cross-sections.
//!
//! For dark tubes on a bright background set `bright_tubes = false`, which
//! inverts the sign convention: `λ₂ > 0` and `λ₃ > 0` are required and the
//! response is computed using `|λ₂|` and `|λ₃|` with the same formula.
//!
//! The final output is the **maximum** response over all scales σ.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

use super::frangi::gaussian_blur_vec;
use super::hessian::{compute_hessian_3d, symmetric_3x3_eigenvalues};

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for the Sato line filter.
#[derive(Debug, Clone)]
pub struct SatoConfig {
    /// Gaussian scale values σ (physical units, e.g. mm) at which to evaluate
    /// the filter.  The output is the per-voxel maximum over all scales.
    pub scales: Vec<f64>,
    /// Cross-section anisotropy exponent.  Controls how strongly the ratio
    /// `λ₂/λ₃` is penalised.  Typical range: [0.5, 2.0].  Default: 0.5.
    pub alpha: f64,
    /// When `true`, detect bright structures on a dark background
    /// (requires `λ₂ < 0, λ₃ < 0`).  When `false`, detect dark structures
    /// on a bright background (requires `λ₂ > 0, λ₃ > 0`).
    pub bright_tubes: bool,
}

impl Default for SatoConfig {
    fn default() -> Self {
        Self {
            scales: vec![1.0, 2.0, 4.0],
            alpha: 0.5,
            bright_tubes: true,
        }
    }
}

/// Multi-scale Sato line filter.
///
/// Produces a line-probability map in `[0, ∞)` (not normalised to 1 because the
/// raw Hessian eigenvalue magnitudes carry scale information useful for
/// downstream thresholding).  The output has the same shape and spatial metadata
/// as the input image.
///
/// # Example
/// ```rust,ignore
/// let config = SatoConfig { scales: vec![1.0, 2.0], ..Default::default() };
/// let filter = SatoLineFilter::new(config);
/// let line_map = filter.apply(&image)?;
/// ```
#[derive(Debug, Clone)]
pub struct SatoLineFilter {
    /// Algorithm configuration.
    pub config: SatoConfig,
}

impl SatoLineFilter {
    /// Construct with explicit configuration.
    pub fn new(config: SatoConfig) -> Self {
        Self { config }
    }

    /// Apply the filter to a 3-D image.
    ///
    /// The input tensor must have `f32` element type.
    ///
    /// # Errors
    /// Returns an error if the image tensor cannot be converted to `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("SatoLineFilter requires f32 image data: {:?}", e))?
            .to_vec();

        let dims = image.shape();
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];

        let response = compute_sato_multiscale(&vals, dims, spacing, &self.config);

        let device = image.data().device();
        let td2 = TensorData::new(response, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td2, &device);
        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Compute per-voxel maximum Sato line response over all scales.
fn compute_sato_multiscale(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    config: &SatoConfig,
) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    let mut max_response = vec![0.0_f32; n];

    for &sigma in &config.scales {
        // Gaussian-blur the image at scale σ.
        let blurred = gaussian_blur_vec(data, dims, sigma, spacing);

        // Compute physical-space Hessian components at all voxels.
        let hessian = compute_hessian_3d(&blurred, dims, spacing);

        // Per-voxel line response (scale-normalised by σ²).
        let sigma2 = (sigma * sigma) as f32;
        for (i, h) in hessian.iter().enumerate() {
            // Scale-normalise Hessian (σ² · H_σ convention).
            let h_scaled = [
                h[0] * sigma2,
                h[1] * sigma2,
                h[2] * sigma2,
                h[3] * sigma2,
                h[4] * sigma2,
                h[5] * sigma2,
            ];
            let eigs = symmetric_3x3_eigenvalues(h_scaled);
            let v = sato_response(eigs, config.alpha, config.bright_tubes);
            if v > max_response[i] {
                max_response[i] = v;
            }
        }
    }

    max_response
}

/// Compute the Sato line response for a single voxel given its three Hessian
/// eigenvalues in arbitrary order.
///
/// # Algorithm
/// 1. Sort eigenvalues by absolute value: `|λ₁| ≤ |λ₂| ≤ |λ₃|`.
/// 2. For bright tubes: require `λ₂ < 0` and `λ₃ < 0`.
///    For dark tubes:  require `λ₂ > 0` and `λ₃ > 0`.
///    Inversion for dark tubes: negate all eigenvalues before the test.
/// 3. Compute `V = |λ₃| · (λ₂/λ₃)^α · f(λ₁,λ₂)`.
#[inline]
fn sato_response(eigenvalues: [f32; 3], alpha: f64, bright_tubes: bool) -> f32 {
    // Sort by absolute value (bubble-sort on 3 elements; branchless-friendly).
    let mut e = eigenvalues;
    if e[0].abs() > e[1].abs() {
        e.swap(0, 1);
    }
    if e[1].abs() > e[2].abs() {
        e.swap(1, 2);
    }
    if e[0].abs() > e[1].abs() {
        e.swap(0, 1);
    }
    // Now |e[0]| ≤ |e[1]| ≤ |e[2]|  (λ₁, λ₂, λ₃).
    let [lam1, lam2, lam3] = e;

    // For dark tubes invert all signs so that the bright-tube gate applies.
    let (l1, l2, l3) = if bright_tubes {
        (lam1, lam2, lam3)
    } else {
        (-lam1, -lam2, -lam3)
    };

    // Gate: both l2 and l3 must be strictly negative.
    if l2 >= 0.0 || l3 >= 0.0 {
        return 0.0;
    }

    // Avoid numerical instability: |l3| must be non-trivial.
    let abs_l3 = l3.abs();
    if abs_l3 < f32::EPSILON {
        return 0.0;
    }

    // Ratio λ₂/λ₃ ∈ (0, 1] (both negative → positive ratio).
    let ratio = l2 / l3; // ∈ (0,1] since |l2| ≤ |l3| and both negative.

    // Shape anisotropy term: (λ₂/λ₃)^α
    let shape_term = (ratio as f64).powf(alpha) as f32;

    // Perpendicular modulation: f(λ₁, λ₂)
    let perp_term = if l1 <= 0.0 {
        1.0_f32
    } else {
        // exp(−λ₁² / (2·(α·λ₂)²))
        let denom = 2.0 * (alpha * l2 as f64) * (alpha * l2 as f64);
        if denom < 1e-30 {
            0.0
        } else {
            (-(l1 as f64 * l1 as f64) / denom).exp() as f32
        }
    };

    abs_l3 * shape_term * perp_term
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    // Re-import using the crate's own paths (within ritk-core).
    use crate::image::Image as CoreImage;
    use crate::spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, dims: [usize; 3]) -> CoreImage<B, 3> {
        let _device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let td = TensorData::new(data, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &_device);
        CoreImage::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Build a 3-D volume with a bright cylinder of radius `r` centred at
    /// (cy, cx) running along the full z-axis.
    fn make_tube(nz: usize, ny: usize, nx: usize, cy: f32, cx: f32, r: f32) -> Vec<f32> {
        (0..nz * ny * nx)
            .map(|fi| {
                let ix = (fi % nx) as f32;
                let iy = ((fi / nx) % ny) as f32;
                let dist = ((ix - cx).powi(2) + (iy - cy).powi(2)).sqrt();
                if dist <= r {
                    1.0_f32
                } else {
                    0.0_f32
                }
            })
            .collect()
    }

    /// Build a bright sphere centred at (cz, cy, cx) with radius `r`.
    fn make_sphere(nz: usize, ny: usize, nx: usize, cz: f32, cy: f32, cx: f32, r: f32) -> Vec<f32> {
        (0..nz * ny * nx)
            .map(|fi| {
                let ix = (fi % nx) as f32;
                let iy = ((fi / nx) % ny) as f32;
                let iz = (fi / (ny * nx)) as f32;
                let dist = ((ix - cx).powi(2) + (iy - cy).powi(2) + (iz - cz).powi(2)).sqrt();
                if dist <= r {
                    1.0_f32
                } else {
                    0.0_f32
                }
            })
            .collect()
    }

    // ── Test 1 ────────────────────────────────────────────────────────────────

    /// A bright cylinder along the z-axis must produce a high Sato response
    /// at its centre compared to the background.
    #[test]
    fn test_cylindrical_tube_detects_line() {
        const N: usize = 32;
        let cy = 16.0_f32;
        let cx = 16.0_f32;
        let r = 3.0_f32;
        let data = make_tube(N, N, N, cy, cx, r);
        let image = make_image(data, [N, N, N]);

        let config = SatoConfig {
            scales: vec![1.5, 3.0],
            alpha: 0.5,
            bright_tubes: true,
        };
        let filter = SatoLineFilter::new(config);
        let result = filter.apply(&image).expect("apply failed");

        let _device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let out: Vec<f32> = result
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        // Centre column: z = any, y = 16, x = 16 (flat index = z*N*N + 16*N + 16).
        let mut centre_responses: Vec<f32> =
            (0..N).map(|iz| out[iz * N * N + 16 * N + 16]).collect();
        centre_responses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_centre = centre_responses[N / 2];

        // Far background voxels (corner strip x=0..2, y=0..2).
        let background: Vec<f32> = (0..N).map(|iz| out[iz * N * N + 0 * N + 0]).collect();
        let mut bg = background.clone();
        bg.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_bg = bg[N / 2];

        assert!(
            median_centre > 0.1,
            "centre-line median Sato response should be > 0.1, got {median_centre:.6}"
        );
        assert!(
            median_centre > median_bg * 3.0,
            "centre response ({median_centre:.6}) should exceed background ({median_bg:.6}) by 3×"
        );
    }

    // ── Test 2 ────────────────────────────────────────────────────────────────

    /// A bright sphere has a lower peak Sato response than the cylinder tube
    /// produced by test 1, because spheres are blob-like, not line-like.
    #[test]
    fn test_sphere_lower_response_than_tube() {
        const N: usize = 32;
        let config = SatoConfig {
            scales: vec![1.5, 3.0],
            alpha: 0.5,
            bright_tubes: true,
        };

        let tube_data = make_tube(N, N, N, 16.0, 16.0, 3.0);
        let sphere_data = make_sphere(N, N, N, 16.0, 16.0, 16.0, 4.0);

        let filter = SatoLineFilter::new(config);

        let tube_img = make_image(tube_data, [N, N, N]);
        let sphere_img = make_image(sphere_data, [N, N, N]);

        let tube_out: Vec<f32> = filter
            .apply(&tube_img)
            .unwrap()
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let sphere_out: Vec<f32> = filter
            .apply(&sphere_img)
            .unwrap()
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let tube_peak = tube_out.iter().cloned().fold(0.0_f32, f32::max);
        let sphere_peak = sphere_out.iter().cloned().fold(0.0_f32, f32::max);

        assert!(
            tube_peak > sphere_peak,
            "tube peak ({tube_peak:.6}) should exceed sphere peak ({sphere_peak:.6})"
        );
    }

    // ── Test 3 ────────────────────────────────────────────────────────────────

    /// With `bright_tubes = true`, a dark tube (intensity 0, background 1)
    /// must produce near-zero response everywhere.
    #[test]
    fn test_dark_tube_rejected_by_bright_gate() {
        const N: usize = 24;
        // Invert: background = 1, tube = 0.
        let tube_mask = make_tube(N, N, N, 12.0, 12.0, 2.5);
        let dark_tube: Vec<f32> = tube_mask.iter().map(|&v| 1.0 - v).collect();

        let config = SatoConfig {
            scales: vec![1.5],
            alpha: 0.5,
            bright_tubes: true, // bright gate — should reject the dark tube
        };
        let filter = SatoLineFilter::new(config);
        let result = filter.apply(&make_image(dark_tube, [N, N, N])).unwrap();
        let out: Vec<f32> = result
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let max_resp = out.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max_resp < 0.05,
            "bright-gate must reject dark tube; max response = {max_resp:.6}"
        );
    }

    // ── Test 4 ────────────────────────────────────────────────────────────────

    /// A uniform image has zero Hessian everywhere, so all Sato responses are zero.
    #[test]
    fn test_uniform_image_zero_response() {
        const N: usize = 16;
        let data = vec![0.5_f32; N * N * N];
        let config = SatoConfig::default();
        let filter = SatoLineFilter::new(config);
        let result = filter.apply(&make_image(data, [N, N, N])).unwrap();
        let out: Vec<f32> = result
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let max_resp = out.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max_resp < 1e-5,
            "uniform image must give zero response; max = {max_resp:.2e}"
        );
    }

    // ── Test 5 ────────────────────────────────────────────────────────────────

    /// All output voxels must be finite for any non-trivial input.
    #[test]
    fn test_response_all_finite() {
        const N: usize = 20;
        // Sinusoidal phantom: exercises all code paths including the perp term.
        let data: Vec<f32> = (0..N * N * N)
            .map(|fi| {
                let ix = fi % N;
                let iy = (fi / N) % N;
                let iz = fi / (N * N);
                let v = (std::f32::consts::PI * ix as f32 / N as f32).sin()
                    * (std::f32::consts::PI * iy as f32 / N as f32).cos()
                    * (0.5 + iz as f32 / N as f32);
                v
            })
            .collect();

        let config = SatoConfig {
            scales: vec![1.0, 2.0],
            alpha: 1.0,
            bright_tubes: true,
        };
        let filter = SatoLineFilter::new(config);
        let result = filter.apply(&make_image(data, [N, N, N])).unwrap();
        let out: Vec<f32> = result
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "voxel {i} is non-finite: {v}");
        }
    }
}
