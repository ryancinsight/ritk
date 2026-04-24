//! Frangi multi-scale vesselness filter (Frangi et al., MICCAI 1998).
//!
//! # Mathematical specification
//!
//! Given a 3-D image `I`, for each scale σ:
//! 1. Smooth `I` with a Gaussian of standard deviation σ (physical units).
//! 2. Compute the Hessian `H` at every voxel via second-order finite differences.
//! 3. Compute eigenvalues `|λ₁| ≤ |λ₂| ≤ |λ₃|` of `H`.
//! 4. Apply the Frangi vesselness measure:
//!
//! ```text
//!   R_A = |λ₂| / |λ₃|                          (cross-section anisotropy)
//!   R_B = |λ₁| / √(|λ₂| · |λ₃|)               (blobness)
//!   S   = √(λ₁² + λ₂² + λ₃²)                  (structureness / Frobenius norm)
//!
//!   V(σ) = (1 − exp(−R_A²/(2α²)))
//!         · exp(−R_B²/(2β²))
//!         · (1 − exp(−S²/(2γ²)))
//! ```
//!
//! For bright vessels (`bright_vessels = true`): `V = 0` if `λ₂ ≥ 0` or `λ₃ ≥ 0`.
//! For dark  vessels (`bright_vessels = false`): `V = 0` if `λ₂ ≤ 0` or `λ₃ ≤ 0`.
//!
//! 5. The final vesselness map is the **maximum** over all σ: `V*(p) = max_σ V(σ, p)`.
//!
//! # Reference
//! Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998).
//! Multiscale vessel enhancement filtering. MICCAI, LNCS 1496, 130–137.

use super::hessian::{compute_hessian_3d, symmetric_3x3_eigenvalues};
use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, TensorData};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration parameters for the Frangi vesselness filter.
///
/// Default values match the recommendations in Frangi et al. (1998).
#[derive(Debug, Clone)]
pub struct FrangiConfig {
    /// Scale values σ (in physical units, e.g. mm) at which to evaluate the filter.
    /// The output is the maximum vesselness over all scales.
    pub scales: Vec<f64>,
    /// Plate-like vs. line-like anisotropy threshold (controls sensitivity of R_A).
    /// Typical value: 0.5.
    pub alpha: f64,
    /// Blobness threshold (controls sensitivity of R_B).
    /// Typical value: 0.5.
    pub beta: f64,
    /// Noise / background structureness threshold (controls sensitivity of S).
    /// Typical value: 15.0 (half the maximum Frobenius norm of the Hessian for
    /// the application image intensity range).
    pub gamma: f64,
    /// If `true`, detect bright structures on a dark background (e.g. vessels in
    /// MRA); requires `λ₂ < 0` and `λ₃ < 0`.
    /// If `false`, detect dark structures on a bright background; requires
    /// `λ₂ > 0` and `λ₃ > 0`.
    pub bright_vessels: bool,
}

impl Default for FrangiConfig {
    fn default() -> Self {
        Self {
            scales: vec![1.0, 2.0, 4.0],
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
            bright_vessels: true,
        }
    }
}

// ── Filter ────────────────────────────────────────────────────────────────────

/// Multi-scale Frangi vesselness filter for 3-D medical images.
///
/// Produces a vesselness probability map in `[0, 1]` with the same shape and
/// spatial metadata as the input image.
///
/// # Example
/// ```rust,ignore
/// let config = FrangiConfig { scales: vec![1.0, 2.0], ..Default::default() };
/// let filter = FrangiVesselnessFilter { config };
/// let vesselness_image = filter.apply(&image)?;
/// ```
#[derive(Debug, Clone)]
pub struct FrangiVesselnessFilter {
    pub config: FrangiConfig,
}

impl FrangiVesselnessFilter {
    /// Construct with explicit configuration.
    pub fn new(config: FrangiConfig) -> Self {
        Self { config }
    }

    /// Apply the filter to a 3-D image.
    ///
    /// The input tensor must have `f32` element type.
    ///
    /// # Errors
    /// Returns an error if the tensor dtype is not `f32` or if the input is
    /// degenerate (fewer than 1 voxel per dimension).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        // ── Extract voxel data ────────────────────────────────────────────────
        let vals: Vec<f32> = image
            .data()
            .clone()
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("f32 required: {:?}", e))?;

        let shape = image.shape(); // [nz, ny, nx]
        let dims = [shape[0], shape[1], shape[2]];
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];

        let n = dims[0] * dims[1] * dims[2];
        let mut vesselness_max = vec![0.0f32; n];

        // ── Max over scales ───────────────────────────────────────────────────
        for &sigma in &self.config.scales {
            // 1. Gaussian-blur the image at the current scale.
            let blurred = gaussian_blur_vec(&vals, dims, sigma, spacing);

            // 2. Compute Hessian at every voxel.
            let hessians = compute_hessian_3d(&blurred, dims, spacing);

            // 3. Compute Frangi vesselness at every voxel.
            for i in 0..n {
                let [lambda1, lambda2, lambda3] = symmetric_3x3_eigenvalues(hessians[i]);
                let v = self.voxel_vesselness(lambda1, lambda2, lambda3);
                if v > vesselness_max[i] {
                    vesselness_max[i] = v;
                }
            }
        }

        // ── Rebuild image ─────────────────────────────────────────────────────
        let td2 = TensorData::new(vesselness_max, Shape::new(shape));
        let tensor = burn::tensor::Tensor::<B, 3>::from_data(td2, &image.data().device());
        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }

    /// Evaluate the Frangi vesselness measure for a single voxel.
    ///
    /// Precondition: eigenvalues are sorted by absolute value ascending,
    /// i.e. `|lambda1| ≤ |lambda2| ≤ |lambda3|`.
    #[inline]
    fn voxel_vesselness(&self, lambda1: f32, lambda2: f32, lambda3: f32) -> f32 {
        // Vessel polarity gate.
        if self.config.bright_vessels {
            // Bright vessel on dark background: both transverse eigenvalues must
            // be negative (concave in cross-sectional directions).
            if lambda2 >= 0.0 || lambda3 >= 0.0 {
                return 0.0;
            }
        } else {
            // Dark vessel on bright background: both must be positive.
            if lambda2 <= 0.0 || lambda3 <= 0.0 {
                return 0.0;
            }
        }

        let l3_abs = lambda3.abs();
        if l3_abs < 1e-10 {
            return 0.0;
        }

        // R_A = |λ₂| / |λ₃|  — cross-section anisotropy (plate vs. line).
        let r_a = lambda2.abs() / l3_abs;

        // R_B = |λ₁| / √(|λ₂| · |λ₃|)  — blobness (blob vs. line/plate).
        let l2_l3_sqrt = (lambda2.abs() * l3_abs).sqrt();
        let r_b = if l2_l3_sqrt < 1e-20 {
            0.0f32
        } else {
            lambda1.abs() / l2_l3_sqrt
        };

        // S = ‖H‖_F = √(λ₁² + λ₂² + λ₃²)  — structureness.
        let s = (lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3).sqrt();
        if s < 1e-10 {
            return 0.0;
        }

        let alpha = self.config.alpha as f32;
        let beta = self.config.beta as f32;
        let gamma = self.config.gamma as f32;

        let v = (1.0 - (-(r_a * r_a) / (2.0 * alpha * alpha)).exp())
            * (-(r_b * r_b) / (2.0 * beta * beta)).exp()
            * (1.0 - (-(s * s) / (2.0 * gamma * gamma)).exp());

        v.max(0.0)
    }
}

// ── Separable Gaussian blur on Vec<f32> ───────────────────────────────────────

/// Apply separable 3-D Gaussian smoothing to a flat voxel buffer.
///
/// Convolves sequentially along the Z, Y, and X axes.  Each axis uses a
/// normalised Gaussian kernel of radius `⌈3·σ_px⌉` voxels where
/// `σ_px = sigma_mm / spacing[axis]`.
///
/// Boundary condition: replicate (clamp-to-edge).
pub(crate) fn gaussian_blur_vec(
    data: &[f32],
    dims: [usize; 3],
    sigma_mm: f64,
    spacing: [f64; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut buf = data.to_vec();

    for axis in 0..3usize {
        let sigma_px = sigma_mm / spacing[axis];
        if sigma_px < 1e-9 {
            continue;
        }

        let radius = (3.0 * sigma_px).ceil() as usize;
        let kernel = build_gaussian_kernel(sigma_px, radius);
        let mut scratch = vec![0.0f32; n];

        match axis {
            0 => {
                // Z-axis: stride = ny * nx
                for iy in 0..ny {
                    for ix in 0..nx {
                        for iz in 0..nz {
                            let flat = iz * ny * nx + iy * nx + ix;
                            let mut acc = 0.0f32;
                            for (ki, &kv) in kernel.iter().enumerate() {
                                let src_iz = (iz as isize + ki as isize - radius as isize)
                                    .clamp(0, nz as isize - 1)
                                    as usize;
                                acc += kv * buf[src_iz * ny * nx + iy * nx + ix];
                            }
                            scratch[flat] = acc;
                        }
                    }
                }
            }
            1 => {
                // Y-axis: stride = nx
                for iz in 0..nz {
                    for ix in 0..nx {
                        for iy in 0..ny {
                            let flat = iz * ny * nx + iy * nx + ix;
                            let mut acc = 0.0f32;
                            for (ki, &kv) in kernel.iter().enumerate() {
                                let src_iy = (iy as isize + ki as isize - radius as isize)
                                    .clamp(0, ny as isize - 1)
                                    as usize;
                                acc += kv * buf[iz * ny * nx + src_iy * nx + ix];
                            }
                            scratch[flat] = acc;
                        }
                    }
                }
            }
            _ => {
                // X-axis: stride = 1
                for iz in 0..nz {
                    for iy in 0..ny {
                        for ix in 0..nx {
                            let flat = iz * ny * nx + iy * nx + ix;
                            let mut acc = 0.0f32;
                            for (ki, &kv) in kernel.iter().enumerate() {
                                let src_ix = (ix as isize + ki as isize - radius as isize)
                                    .clamp(0, nx as isize - 1)
                                    as usize;
                                acc += kv * buf[iz * ny * nx + iy * nx + src_ix];
                            }
                            scratch[flat] = acc;
                        }
                    }
                }
            }
        }

        buf = scratch;
    }

    buf
}

/// Build a normalised 1-D Gaussian kernel.
///
/// `kernel[i] = exp(−(i − radius)² / (2 · sigma_px²))`, normalised to sum = 1.
fn build_gaussian_kernel(sigma_px: f64, radius: usize) -> Vec<f32> {
    let width = 2 * radius + 1;
    let inv_two_s2 = 1.0 / (2.0 * sigma_px * sigma_px);
    let mut k: Vec<f32> = (0..width)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-(x * x) * inv_two_s2).exp() as f32
        })
        .collect();
    let sum: f32 = k.iter().sum();
    if sum > f32::EPSILON {
        for v in &mut k {
            *v /= sum;
        }
    }
    k
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    // ── gaussian_blur_vec ─────────────────────────────────────────────────────

    /// Blurring a uniform image must return the same uniform image.
    #[test]
    fn test_gaussian_blur_uniform_invariant() {
        let dims = [8usize, 8, 8];
        let data = vec![5.0f32; 8 * 8 * 8];
        let blurred = gaussian_blur_vec(&data, dims, 1.0, [1.0, 1.0, 1.0]);
        for (i, &v) in blurred.iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-4,
                "uniform blur: voxel {i} expected 5.0, got {v}"
            );
        }
    }

    /// Gaussian blur must reduce peak intensity (smoothing spreads energy).
    #[test]
    fn test_gaussian_blur_smooths_peak() {
        let dims = [9usize, 9, 9];
        let n = 9 * 9 * 9;
        let mut data = vec![0.0f32; n];
        // Single bright voxel at centre.
        data[4 * 9 * 9 + 4 * 9 + 4] = 1000.0;
        let blurred = gaussian_blur_vec(&data, dims, 1.5, [1.0, 1.0, 1.0]);
        let peak = blurred.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(peak < 1000.0, "blur must reduce peak; peak = {peak}");
        // Energy (sum) must be conserved under replicate boundary.
        let sum_orig: f32 = data.iter().sum();
        let sum_blur: f32 = blurred.iter().sum();
        // Sum is not necessarily preserved under clamp-to-edge blurring when
        // the image is mostly zero.  Check only that no energy is generated.
        assert!(
            sum_blur <= sum_orig * 1.01,
            "blur must not create energy: orig={sum_orig}, blurred={sum_blur}"
        );
    }

    // ── FrangiVesselnessFilter ────────────────────────────────────────────────

    /// **Test 1 — Cylindrical tube phantom.**
    ///
    /// 20×20×20 image.  All voxels whose cross-sectional distance from the
    /// z-axis centre (y = 9.5, x = 9.5) is < 3 voxels are set to 100.0;
    /// the background is 0.0.
    ///
    /// Invariants verified:
    /// - Tube centre (iz ∈ {9,10}, iy ∈ {9,10}, ix ∈ {9,10}): V > 0.05.
    /// - Corners (0,0,0) and (19,19,19): V < 0.02.
    #[test]
    fn test_frangi_cylindrical_tube() {
        const N: usize = 20;
        let mut vals = vec![0.0f32; N * N * N];
        let centre = (N as f64 - 1.0) / 2.0; // 9.5
        for iz in 0..N {
            for iy in 0..N {
                for ix in 0..N {
                    let dy = iy as f64 - centre;
                    let dx = ix as f64 - centre;
                    if (dy * dy + dx * dx).sqrt() < 3.0 {
                        vals[iz * N * N + iy * N + ix] = 100.0;
                    }
                }
            }
        }

        let image = make_image(vals, [N, N, N]);
        let config = FrangiConfig {
            scales: vec![1.0, 2.0],
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
            bright_vessels: true,
        };
        let filter = FrangiVesselnessFilter::new(config);
        let out = filter.apply(&image).expect("frangi apply failed");

        let v: Vec<f32> = out.data().clone().into_data().into_vec::<f32>().unwrap();

        let get = |iz: usize, iy: usize, ix: usize| v[iz * N * N + iy * N + ix];

        // Tube-centre voxels: vesselness must be clearly positive.
        for iz in [9usize, 10] {
            for iy in [9usize, 10] {
                for ix in [9usize, 10] {
                    let val = get(iz, iy, ix);
                    assert!(
                        val > 0.05,
                        "tube centre ({iz},{iy},{ix}): expected > 0.05, got {val}"
                    );
                }
            }
        }

        // Corner voxels: far from tube, vesselness must be near zero.
        for (iz, iy, ix) in [(0usize, 0usize, 0usize), (N - 1, N - 1, N - 1)] {
            let val = get(iz, iy, ix);
            assert!(
                val < 0.02,
                "corner ({iz},{iy},{ix}): expected < 0.02, got {val}"
            );
        }
    }

    /// **Test 2 — Uniform image.**
    ///
    /// All second derivatives are zero at every voxel → S = 0 → V = 0.
    #[test]
    fn test_frangi_uniform_image_zero_vesselness() {
        const N: usize = 10;
        let vals = vec![42.0f32; N * N * N];
        let image = make_image(vals, [N, N, N]);
        let filter = FrangiVesselnessFilter::new(FrangiConfig::default());
        let out = filter.apply(&image).expect("frangi apply failed");

        let v: Vec<f32> = out.data().clone().into_data().into_vec::<f32>().unwrap();
        for (i, &val) in v.iter().enumerate() {
            assert!(val < 1e-6, "uniform image: voxel {i} expected 0, got {val}");
        }
    }

    /// **Test 3 — Spherical blob.**
    ///
    /// A bright sphere of radius 5 centred in a 30×30×30 image is a
    /// blob-like structure, not a vessel.  With `bright_vessels = true` and
    /// the Frangi measure, a perfect sphere satisfies the polarity gate
    /// (both λ₂ and λ₃ are negative) but the blobness term
    /// `exp(−R_B² / (2β²))` with R_B ≈ 1 and β = 0.5 evaluates to ≈ 0.135,
    /// yielding a vesselness value substantially lower than that of a tube
    /// (where R_B ≈ 0 → blobness term ≈ 1.0).
    ///
    /// Invariant: sphere centre vesselness < tube scale factor × 0.3.
    /// Concretely: V_sphere_centre < 0.4.
    #[test]
    fn test_frangi_sphere_low_vesselness() {
        const N: usize = 30;
        let mut vals = vec![0.0f32; N * N * N];
        let centre = (N as f64 - 1.0) / 2.0; // 14.5
        for iz in 0..N {
            for iy in 0..N {
                for ix in 0..N {
                    let dz = iz as f64 - centre;
                    let dy = iy as f64 - centre;
                    let dx = ix as f64 - centre;
                    if (dz * dz + dy * dy + dx * dx).sqrt() < 5.0 {
                        vals[iz * N * N + iy * N + ix] = 100.0;
                    }
                }
            }
        }

        let image = make_image(vals, [N, N, N]);
        let config = FrangiConfig {
            scales: vec![1.0, 2.0],
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
            bright_vessels: true,
        };
        let filter = FrangiVesselnessFilter::new(config);
        let out = filter.apply(&image).expect("frangi apply failed");

        let v: Vec<f32> = out.data().clone().into_data().into_vec::<f32>().unwrap();

        // Sphere-centre voxel.
        let c = 14usize; // floor(14.5)
        let centre_idx = c * N * N + c * N + c;
        let val = v[centre_idx];

        // A sphere has R_B ≈ 1 → blobness term ≈ exp(-2) ≈ 0.135, so
        // vesselness is suppressed relative to a tube.  Threshold: < 0.4.
        assert!(
            val < 0.4,
            "sphere centre vesselness: expected < 0.4 (blob suppression), got {val}"
        );
    }

    /// **Test 4 — Tube vs. sphere discrimination.**
    ///
    /// The vesselness at the tube centre must exceed the vesselness at the
    /// sphere centre.  This directly validates that the Frangi measure
    /// discriminates tubular from blob-like structures.
    #[test]
    fn test_frangi_tube_exceeds_sphere() {
        const N: usize = 20;
        // ── Tube phantom ──────────────────────────────────────────────────────
        let mut tube_vals = vec![0.0f32; N * N * N];
        let centre = (N as f64 - 1.0) / 2.0;
        for iz in 0..N {
            for iy in 0..N {
                for ix in 0..N {
                    let dy = iy as f64 - centre;
                    let dx = ix as f64 - centre;
                    if (dy * dy + dx * dx).sqrt() < 3.0 {
                        tube_vals[iz * N * N + iy * N + ix] = 100.0;
                    }
                }
            }
        }

        // ── Sphere phantom ────────────────────────────────────────────────────
        let mut sphere_vals = vec![0.0f32; N * N * N];
        for iz in 0..N {
            for iy in 0..N {
                for ix in 0..N {
                    let dz = iz as f64 - centre;
                    let dy = iy as f64 - centre;
                    let dx = ix as f64 - centre;
                    if (dz * dz + dy * dy + dx * dx).sqrt() < 3.0 {
                        sphere_vals[iz * N * N + iy * N + ix] = 100.0;
                    }
                }
            }
        }

        let config = FrangiConfig {
            scales: vec![1.0, 2.0],
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
            bright_vessels: true,
        };

        let tube_image = make_image(tube_vals, [N, N, N]);
        let sphere_image = make_image(sphere_vals, [N, N, N]);

        let filter = FrangiVesselnessFilter::new(config);
        let tube_out = filter.apply(&tube_image).unwrap();
        let sphere_out = filter.apply(&sphere_image).unwrap();

        let tube_v: Vec<f32> = tube_out.data().clone().into_data().into_vec::<f32>().unwrap();
        let sphere_v: Vec<f32> = sphere_out.data().clone().into_data().into_vec::<f32>().unwrap();

        // Centre index for 20×20×20 image.
        let c = 9usize;
        let tube_centre = tube_v[c * N * N + c * N + c];
        let sphere_centre = sphere_v[c * N * N + c * N + c];

        assert!(
            tube_centre > sphere_centre,
            "tube centre ({tube_centre:.4}) must exceed sphere centre ({sphere_centre:.4})"
        );
    }

    /// **Test 5 — Dark vessel polarity gate.**
    ///
    /// With `bright_vessels = false`, a bright tube returns zero vesselness
    /// everywhere (all eigenvalues are negative, gate requires positive).
    #[test]
    fn test_frangi_dark_vessel_gate_rejects_bright_tube() {
        const N: usize = 12;
        let mut vals = vec![0.0f32; N * N * N];
        let centre = (N as f64 - 1.0) / 2.0;
        for iz in 0..N {
            for iy in 0..N {
                for ix in 0..N {
                    let dy = iy as f64 - centre;
                    let dx = ix as f64 - centre;
                    if (dy * dy + dx * dx).sqrt() < 2.5 {
                        vals[iz * N * N + iy * N + ix] = 100.0;
                    }
                }
            }
        }

        let image = make_image(vals, [N, N, N]);
        let config = FrangiConfig {
            scales: vec![1.5],
            bright_vessels: false, // dark-vessel mode
            ..Default::default()
        };
        let filter = FrangiVesselnessFilter::new(config);
        let out = filter.apply(&image).expect("frangi apply failed");

        let v: Vec<f32> = out.data().clone().into_data().into_vec::<f32>().unwrap();

        let c = N / 2;
        let centre_val = v[c * N * N + c * N + c];
        assert!(
            centre_val < 1e-6,
            "dark-vessel mode must reject bright tube; centre V = {centre_val}"
        );
    }
}
