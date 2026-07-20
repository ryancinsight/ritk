//! Frangi multi-scale vesselness filter (Frangi et al., MICCAI 1998).
//!
//! # Mathematical specification
//!
//! Given a 3-D image `I`, for each scale σ:
//! 1. Smooth `I` with a Gaussian of standard deviation σ (physical units).
//! 2. Compute the Hessian `H` at every voxel via the second-order Deriche IIR
//!    recursion (matching ITK HessianRecursiveGaussianImageFilter).
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

use super::hessian::symmetric_3x3_eigenvalues;
use super::VesselPolarity;
use crate::recursive_gaussian::compute_hessian_iir;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

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
    /// Blobness threshold (controls sensitivity of R_B). Typical value: 0.5.
    pub beta: f64,
    /// Noise / background structureness threshold (controls sensitivity of S).
    /// Typical value: 15.0 (half the maximum Frobenius norm of the Hessian for
    /// the application image intensity range).
    pub gamma: f64,
    /// Vessel polarity: detect bright structures on a dark background
    /// (e.g. vessels in MRA) or dark structures on a bright background.
    pub polarity: VesselPolarity,
}

impl Default for FrangiConfig {
    fn default() -> Self {
        Self {
            scales: vec![1.0, 2.0, 4.0],
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
            polarity: VesselPolarity::Bright,
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
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        let vesselness_max = self.compute(&vals_vec, dims, spacing);
        Ok(rebuild(vesselness_max, dims, image))
    }

    /// Coeus-native sister of [`FrangiVesselnessFilter::apply`].
    ///
    /// Runs the identical multi-scale Frangi vesselness (recursive-Gaussian
    /// Hessian + eigen-analysis, max over scales) via the shared `compute`
    /// host core on the image's contiguous host buffer, so the result is
    /// bitwise-identical to the Coeus path. No Coeus tensor is constructed.
    /// Spatial metadata is preserved.
    ///
    /// `compute`: FrangiVesselnessFilter::compute
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            self.compute(vals, dims, spacing)
        })
    }

    /// Substrate-agnostic host core: per-voxel maximum Frangi vesselness over all
    /// configured scales, on a flat `[nz, ny, nx]` buffer. Shared single source
    /// of truth for the Coeus [`apply`](Self::apply) and Coeus-native
    /// [`apply_native`](Self::apply_native) paths.
    fn compute(&self, vals: &[f32], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
        let n = dims[0] * dims[1] * dims[2];
        let mut vesselness_max = vec![0.0f32; n];

        for &sigma in &self.config.scales {
            // Compute Hessian via second-order Deriche IIR recursion —
            // matching ITK HessianRecursiveGaussianImageFilter.
            let hessians = compute_hessian_iir(vals, dims, spacing, sigma);

            let hessians_ref = &hessians;
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                &mut vesselness_max,
                4096,
                |chunk_idx, slice| {
                    let start_idx = chunk_idx * 4096;
                    for (offset, max_val) in slice.iter_mut().enumerate() {
                        let i = start_idx + offset;
                        let [lambda1, lambda2, lambda3] =
                            symmetric_3x3_eigenvalues(hessians_ref[i]);
                        let v = self.voxel_vesselness(lambda1, lambda2, lambda3);
                        if v > *max_val {
                            *max_val = v;
                        }
                    }
                },
            );
        }
        vesselness_max
    }

    /// Evaluate the Frangi vesselness measure for a single voxel.
    ///
    /// Precondition: eigenvalues are sorted by absolute value ascending,
    /// i.e. `|lambda1| ≤ |lambda2| ≤ |lambda3|`.
    #[inline]
    fn voxel_vesselness(&self, lambda1: f32, lambda2: f32, lambda3: f32) -> f32 {
        // Vessel polarity gate.
        match self.config.polarity {
            VesselPolarity::Bright => {
                // Bright vessel on dark background: both transverse eigenvalues must
                // be negative (concave in cross-sectional directions).
                if lambda2 >= 0.0 || lambda3 >= 0.0 {
                    return 0.0;
                }
            }
            VesselPolarity::Dark => {
                // Dark vessel on bright background: both must be positive.
                if lambda2 <= 0.0 || lambda3 <= 0.0 {
                    return 0.0;
                }
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

        // S = —–H—–_F = √(λ₁² + λ₂² + λ₃²)  — structureness.
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

// ── Separable Gaussian blur on Vec<f32> (test helper) ─────────────────────

/// Apply separable 3-D Gaussian smoothing to a flat voxel buffer.
///
/// Convolves sequentially along the Z, Y, and X axes.  Each axis uses a
/// normalised Gaussian kernel of radius `⌈ 3·σ_px⌉` voxels where
/// `σ_px = sigma_mm / spacing[axis]`.
///
/// Boundary condition: replicate (clamp-to-edge).
#[cfg(test)]
pub(crate) fn gaussian_blur_vec(
    data: &[f32],
    dims: [usize; 3],
    sigma_mm: f64,
    spacing: [f64; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut buf = data.to_vec();

    for (axis, &sp) in spacing.iter().enumerate() {
        let sigma_px = sigma_mm / sp;
        if sigma_px < 1e-9 {
            continue;
        }

        let radius = (3.0 * sigma_px).ceil() as usize;
        let kernel = crate::gaussian_kernel::<f32>(sigma_px as f32, Some(radius));
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

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use ritk_image::Image;
    use ritk_spatial::{Direction, Point, Spacing};

    type B = SequentialBackend;

    fn image(values: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
        Image::from_flat_on(
            values,
            shape,
            Point::origin(),
            Spacing::uniform(1.0),
            Direction::identity(),
            &B::default(),
        )
        .expect("invariant: valid native test image")
    }

    #[test]
    fn gaussian_blur_preserves_uniform_field() {
        let shape = [8usize, 8, 8];
        let blurred = gaussian_blur_vec(
            &vec![5.0; shape.iter().product()],
            shape,
            1.0,
            [1.0, 1.0, 1.0],
        );
        for (index, &value) in blurred.iter().enumerate() {
            assert!(
                (value - 5.0).abs() < 1e-4,
                "uniform blur at {index}: expected 5, got {value}"
            );
        }
    }

    #[test]
    fn gaussian_blur_reduces_impulse_peak_without_creating_energy() {
        let shape = [9usize, 9, 9];
        let mut values = vec![0.0; shape.iter().product()];
        values[4 * 9 * 9 + 4 * 9 + 4] = 1000.0;
        let blurred = gaussian_blur_vec(&values, shape, 1.5, [1.0, 1.0, 1.0]);
        let peak = blurred.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let source_energy: f32 = values.iter().sum();
        let blurred_energy: f32 = blurred.iter().sum();

        assert!(peak < 1000.0, "blurred impulse peak: {peak}");
        assert!(
            blurred_energy <= source_energy * 1.01,
            "blur must not create energy: source={source_energy}, blurred={blurred_energy}"
        );
    }

    #[test]
    fn hessian_trace_matches_native_laplacian() {
        let shape = [8usize, 8, 8];
        let values: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|index| (index as f32 * 0.01) % 1.0)
            .collect();
        let hessians = compute_hessian_iir(&values, shape, [1.0, 1.0, 1.0], 1.5);
        let trace: Vec<f32> = hessians
            .iter()
            .map(|hessian| hessian[0] + hessian[3] + hessian[5])
            .collect();
        let input = image(values, shape);
        let laplacian =
            crate::recursive_gaussian::laplacian_recursive_gaussian(&input, 1.5, &B::default())
                .expect("native Laplacian succeeds");
        let centre = 4 * 8 * 8 + 4 * 8 + 4;
        let difference =
            (trace[centre] - laplacian.data_slice().expect("contiguous")[centre]).abs();

        assert!(
            difference < 1e-3,
            "Hessian trace differs from Laplacian at center by {difference}"
        );
    }
}
