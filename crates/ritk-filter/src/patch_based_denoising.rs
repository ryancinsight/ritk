//! Patch-based denoising filter (`itk::PatchBasedDenoisingImageFilter`).
//!
//! # Mathematical Specification
//!
//! Faithful port of `sitk.PatchBasedDenoising` with `NoiseModel = GAUSSIAN`
//! (fidelity weight 0 ⇒ no-op) and `KernelBandwidthEstimation = false`. The
//! per-pixel update is the Gaussian-kernel joint-entropy gradient
//! (`itk::PatchBasedDenoisingImageFilter::ComputeGradientJointEntropy`):
//!
//! ```text
//! update[p] = 0.2 · Σ_q (I[q] − I[p]) · exp(−Σ_jj w[jj]²·(I[p+jj] − I[q+jj])² / (2σ²)) / Σ_q g
//! new[p]    = I[p] + update[p]
//! ```
//!
//! over patches `q` selected by the `GaussianRandomSpatialNeighborSubsampler`:
//! per query pixel, `min(number_of_sample_patches, region_size)` patches are
//! drawn **with replacement**, each coordinate `q[d] = ⌊N(p[d], sample_variance)⌋`
//! (rejected to the search region), using ITK's `MersenneTwisterRandomVariateGenerator`
//! (seeded `0` for the single-thread reference). `w` are the cubic-spline
//! smooth-disc patch weights; `σ = kernel_sigma`.
//!
//! Pixels are visited in `itk::ImageBoundaryFacesCalculator` order (interior
//! region first, then each boundary face, each raster-scanned), because the
//! shared RNG state threads through every draw — the visitation order is part
//! of the contract.
//!
//! Validated bit-exact (≤ f32 round-off) against single-threaded
//! `sitk.PatchBasedDenoising` across patch radii 1/2/4 and 1–2 iterations.
//!
//! ## References
//! - Awate, S.P. & Whitaker, R.T. (2006). "Unsupervised, Information-Theoretic,
//!   Adaptive Image Filtering for Image Restoration." *IEEE TPAMI*.
//! - ITK `itkPatchBasedDenoisingImageFilter.hxx`,
//!   `itkGaussianRandomSpatialNeighborSubsampler.hxx`,
//!   `itkMersenneTwisterRandomVariateGenerator.cxx`.

use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};
use std::f64::consts::PI;

// ── ITK MersenneTwister (MT19937) ───────────────────────────────────────────────

/// Bit-exact port of `itk::Statistics::MersenneTwisterRandomVariateGenerator`.
struct ItkMt {
    state: [u32; 624],
    left: usize,
    next: usize,
}

impl ItkMt {
    fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        let mut mt = Self {
            state,
            left: 0,
            next: 0,
        };
        mt.reload();
        mt
    }

    #[inline]
    fn twist(m: u32, s0: u32, s1: u32) -> u32 {
        let mix = (s0 & 0x8000_0000) | (s1 & 0x7fff_ffff);
        m ^ (mix >> 1) ^ ((s1 & 1).wrapping_neg() & 0x9908_b0df)
    }

    fn reload(&mut self) {
        const M: usize = 397;
        let s = &mut self.state;
        for i in 0..(624 - M) {
            s[i] = Self::twist(s[i + M], s[i], s[i + 1]);
        }
        for i in (624 - M)..623 {
            s[i] = Self::twist(s[i + M - 624], s[i], s[i + 1]);
        }
        s[623] = Self::twist(s[M - 1], s[623], s[0]);
        self.left = 624;
        self.next = 0;
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        if self.left == 0 {
            self.reload();
        }
        self.left -= 1;
        let mut s1 = self.state[self.next];
        self.next += 1;
        s1 ^= s1 >> 11;
        s1 ^= (s1 << 7) & 0x9d2c_5680;
        s1 ^= (s1 << 15) & 0xefc6_0000;
        s1 ^ (s1 >> 18)
    }

    /// `GetNormalVariate(mean, variance)` via the Box-Muller transform.
    #[inline]
    fn normal(&mut self, mean: f64, variance: f64) -> f64 {
        // GetVariateWithOpenRange(): (u32 + 0.5) / 2^32.
        let u1 = (self.next_u32() as f64 + 0.5) / 4_294_967_296.0;
        let r = (-2.0 * (1.0 - u1).ln() * variance).sqrt();
        // GetVariateWithOpenUpperRange(): u32 / 2^32.
        let u2 = self.next_u32() as f64 / 4_294_967_296.0;
        mean + r * (2.0 * PI * u2).cos()
    }

    /// `GaussianRandomSpatialNeighborSubsampler::GetIntegerVariate`: draw
    /// `⌊N(mean, variance)⌋`, rejecting until within `[lo, hi]`.
    #[inline]
    fn gauss_int(&mut self, lo: i64, hi: i64, mean: f64, variance: f64) -> i64 {
        loop {
            let ri = self.normal(mean, variance).floor() as i64;
            if ri >= lo && ri <= hi {
                return ri;
            }
        }
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Patch-based denoising (faithful ITK port), bit-exact to single-threaded
/// `sitk.PatchBasedDenoising`.
///
/// # Default parameters (match ITK)
///
/// | Field | Default |
/// |-------|---------|
/// | `number_of_iterations` | 1 |
/// | `number_of_sample_patches` | 200 |
/// | `patch_radius` | 4 |
/// | `sample_variance` | 400.0 |
/// | `kernel_sigma` | 400.0 |
#[derive(Debug, Clone)]
pub struct PatchBasedDenoisingImageFilter {
    /// Number of denoising iterations.
    pub number_of_iterations: usize,
    /// Patches sampled per pixel (`NumberOfResultsRequested`).
    pub number_of_sample_patches: usize,
    /// Half-width of the comparison patch in voxels.
    pub patch_radius: usize,
    /// Variance of the Gaussian patch-sampling domain (`SampleVariance`).
    pub sample_variance: f64,
    /// Gaussian kernel bandwidth σ (`KernelBandwidthSigma`).
    pub kernel_sigma: f64,
}

impl Default for PatchBasedDenoisingImageFilter {
    fn default() -> Self {
        Self {
            number_of_iterations: 1,
            number_of_sample_patches: 200,
            patch_radius: 4,
            sample_variance: 400.0,
            kernel_sigma: 400.0,
        }
    }
}

impl PatchBasedDenoisingImageFilter {
    /// Apply patch-based denoising to a 3-D image (`nz == 1` ⇒ 2-D).
    ///
    /// # Errors
    /// Returns `Err` if the tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (data, dims) = extract_vec(image)?;
        let result = self.run(&data, dims);
        Ok(rebuild(result, dims, image))
    }

    fn run(&self, data: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let n = dims[0] * dims[1] * dims[2];
        if n == 0 {
            return Vec::new();
        }
        let mut current: Vec<f32> = data.to_vec();
        for _ in 0..self.number_of_iterations {
            current = self.pass(&current, dims);
        }
        current
    }
}

// ── Core single iteration ───────────────────────────────────────────────────────

impl PatchBasedDenoisingImageFilter {
    fn pass(&self, data: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let ndim = if nz == 1 { 2 } else { 3 };
        let r = self.patch_radius as i64;
        let s2 = self.kernel_sigma * self.kernel_sigma;
        let nrr = self.number_of_sample_patches;
        let variance = self.sample_variance;

        // Sizes in ITK index order (x, y, z).
        let sizes: [i64; 3] = [nx as i64, ny as i64, nz as i64];

        // Smooth-disc squared patch weights, indexed by patch offset.
        let weights = smooth_disc_weights_sq(self.patch_radius, ndim);

        // Patch offsets in (dz, dy, dx) with their weight index.
        let mut offsets: Vec<(i64, i64, i64, usize)> = Vec::new();
        {
            let mut wi = 0usize;
            let zr = if ndim == 3 { r } else { 0 };
            for dz in -zr..=zr {
                for dy in -r..=r {
                    for dx in -r..=r {
                        offsets.push((dz, dy, dx, wi));
                        wi += 1;
                    }
                }
            }
        }

        let idx = |x: i64, y: i64, z: i64| -> usize {
            (z as usize) * ny * nx + (y as usize) * nx + (x as usize)
        };

        let mut mt = ItkMt::new(0);
        let mut out = data.to_vec();

        // Visit pixels in ImageBoundaryFacesCalculator order (interior, then faces).
        for (x, y, z) in face_calculator_order(sizes, self.patch_radius as i64, ndim) {
            // Region constraint (itkPatchBasedDenoisingImageFilter ComputeGradientJointEntropy).
            // Per dim: rIndex = min(idx, radius); rEnd = max(idx, size-radius-1).
            let mut lo = [0i64; 3];
            let mut hi = [0i64; 3];
            let q0 = [x, y, z];
            for d in 0..ndim {
                lo[d] = q0[d].min(r);
                hi[d] = q0[d].max(sizes[d] - r - 1);
            }
            // Number of patches to draw = min(NRR, region size).
            let mut region: u64 = 1;
            for d in 0..ndim {
                region *= (hi[d] - lo[d] + 1) as u64;
            }
            let nump = (nrr as u64).min(region) as usize;

            let p_center = data[idx(x, y, z)] as f64;
            let mut sum_g = 0.0f64;
            let mut grad = 0.0f64;

            for _ in 0..nump {
                // Draw q per dim in order x, y[, z].
                let qx = mt.gauss_int(lo[0], hi[0], x as f64, variance);
                let qy = mt.gauss_int(lo[1], hi[1], y as f64, variance);
                let qz = if ndim == 3 {
                    mt.gauss_int(lo[2], hi[2], z as f64, variance)
                } else {
                    0
                };

                // Weighted squared patch distance over in-bounds query offsets.
                let mut sq = 0.0f64;
                for &(dz, dy, dx, wi) in &offsets {
                    let (px, py, pz) = (x + dx, y + dy, z + dz);
                    if px < 0
                        || py < 0
                        || pz < 0
                        || px >= sizes[0]
                        || py >= sizes[1]
                        || pz >= sizes[2]
                    {
                        continue; // current-patch pixel out of bounds: skipped
                    }
                    // Selected patch is guaranteed in-bounds by the region constraint.
                    let vp = data[idx(px, py, pz)] as f64;
                    let vq = data[idx(qx + dx, qy + dy, qz + dz)] as f64;
                    let diff = vp - vq;
                    sq += weights[wi] * diff * diff;
                }

                let g = (-(sq / s2) / 2.0).exp();
                sum_g += g;
                grad += (data[idx(qx, qy, qz)] as f64 - p_center) * g;
            }

            let update = 0.2 * grad / (sum_g + f64::MIN_POSITIVE * 100.0);
            out[idx(x, y, z)] = (p_center + update) as f32;
        }

        out
    }
}

// ── Smooth-disc patch weights ────────────────────────────────────────────────────

/// `InitializePatchWeightsSmoothDisc` (isotropic unit spacing), squared, in
/// patch-offset order matching the `(dz, dy, dx)` triple loop.
fn smooth_disc_weights_sq(patch_radius: usize, ndim: usize) -> Vec<f64> {
    let r = patch_radius as f64;
    let disc = (patch_radius / 2) as f64;
    let interval = (patch_radius as f64 + 1.0) - disc;
    let rr = patch_radius as i64;
    let zr = if ndim == 3 { rr } else { 0 };
    let mut w: Vec<f64> = Vec::new();
    for dz in -zr..=zr {
        for dy in -rr..=rr {
            for dx in -rr..=rr {
                let dist = ((dz * dz + dy * dy + dx * dx) as f64).sqrt();
                let v = if dist >= r + 1.0 {
                    0.0
                } else if dist <= disc {
                    1.0
                } else {
                    let t = (r + 1.0) - dist;
                    let weight = (-2.0 / interval.powi(3)) * t.powi(3)
                        + (3.0 / interval.powi(2)) * t.powi(2);
                    weight.clamp(0.0, 1.0)
                };
                w.push(v * v); // squared
            }
        }
    }
    w
}

// ── ImageBoundaryFacesCalculator visitation order ────────────────────────────────

/// Pixel visitation order (ITK `(x, y, z)` index space) reproducing
/// `itk::NeighborhoodAlgorithm::ImageBoundaryFacesCalculator`: the interior
/// (non-boundary) region first, then each boundary face, each raster-scanned
/// (x fastest). `radius` is the patch radius; `ndim` ∈ {2, 3}.
fn face_calculator_order(sizes: [i64; 3], radius: i64, ndim: usize) -> Vec<(i64, i64, i64)> {
    // Each face is a region [start, start+size) in ITK (x, y, z).
    let mut faces: Vec<([i64; 3], [i64; 3])> = Vec::new();
    let mut vr_start = [0i64; 3];
    let mut vr_size = [sizes[0], sizes[1], sizes[2]];
    if ndim == 2 {
        vr_size[2] = 1;
    }
    let mut nb_start = vr_start;
    let mut nb_size = vr_size;

    for i in 0..ndim {
        // Low face (overlapLow = -radius < 0).
        let mut f_start = vr_start;
        let mut f_size = vr_size;
        f_size[i] = radius;
        nb_start[i] += radius;
        nb_size[i] = nb_size[i].saturating_sub(radius);
        vr_start[i] += radius;
        vr_size[i] -= radius;
        faces.push((f_start, f_size));

        // High face (overlapHigh = -radius < 0).
        f_start = vr_start;
        f_size = vr_size;
        f_start[i] = sizes[i] - radius;
        f_size[i] = radius;
        nb_size[i] = nb_size[i].saturating_sub(radius);
        vr_size[i] -= radius;
        faces.push((f_start, f_size));
    }

    // Order: interior (non-boundary) first, then the boundary faces.
    let mut regions: Vec<([i64; 3], [i64; 3])> = Vec::with_capacity(faces.len() + 1);
    regions.push((nb_start, nb_size));
    regions.extend(faces);

    let mut order: Vec<(i64, i64, i64)> = Vec::new();
    for (start, size) in regions {
        if size[0] <= 0 || size[1] <= 0 || size[2] <= 0 {
            continue;
        }
        // Raster scan: x fastest, then y, then z.
        for z in start[2]..start[2] + size[2] {
            for y in start[1]..start[1] + size[1] {
                for x in start[0]..start[0] + size[0] {
                    order.push((x, y, z));
                }
            }
        }
    }
    order
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_patch_based_denoising.rs"]
mod tests;
