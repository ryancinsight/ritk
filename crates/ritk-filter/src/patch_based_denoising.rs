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
//! (rejected to the intersection of the valid-patch region and the sampler's
//! `⌊2.5·√sample_variance⌋` neighborhood), using ITK's
//! `MersenneTwisterRandomVariateGenerator` (seeded `0`). `w` are the cubic-spline
//! smooth-disc patch weights constructed through ITK's `f32` weight-image
//! rounding boundary. Pixel differences likewise execute in input `f32` before
//! widening into the `f64` entropy accumulator; `σ = kernel_sigma`.
//!
//! Pixels are visited in `itk::ImageBoundaryFacesCalculator` order (interior
//! region first, then each boundary face, each raster-scanned), because the
//! shared RNG state threads through every draw — the visitation order is part
//! of the contract. Within each patch, ITK's partial loop unroll interleaves
//! offsets from the lower and upper halves, then accumulates the center last;
//! that floating-point reduction order is also preserved.
//!
//! Validated within one final output rounding step against single-threaded
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
mod neighborhood;
mod sampling;

use neighborhood::{face_calculator_order, smooth_disc_weights_sq};
use sampling::{sampling_interval, ItkMt};
use std::mem::size_of;

/// Bounds the transient sample-index storage used by one parallel batch.
const SAMPLE_BATCH_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
struct PixelWork {
    position: [i64; 3],
    center_index: usize,
    interior: bool,
    first_sample: usize,
    sample_count: usize,
}

#[derive(Clone, Copy)]
struct PatchOffset {
    coordinate: [i64; 3],
    displacement: isize,
    weight: f64,
}

#[inline]
fn pixel_difference(current: f32, selected: f32) -> f64 {
    f64::from(selected - current)
}

#[inline]
fn pixel_differences_are_finite(data: &[f32]) -> bool {
    data.iter()
        .copied()
        .try_fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), value| {
                value
                    .is_finite()
                    .then_some((minimum.min(value), maximum.max(value)))
            },
        )
        .is_some_and(|(minimum, maximum)| (maximum - minimum).is_finite())
}

fn itk_reduction_indices(length: usize) -> impl Iterator<Item = usize> {
    debug_assert_eq!(length % 2, 1, "patch length must be odd");
    let center = length / 2;
    (0..center)
        .flat_map(move |index| [index, center + 1 + index])
        .chain(std::iter::once(center))
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Patch-based denoising (faithful ITK port), bit-exact to single-threaded
/// `sitk.PatchBasedDenoising` within one final output rounding step.
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
    /// Returns `Err` if the configuration is invalid, the image is smaller than
    /// one patch, or the tensor data cannot be read as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let max_samples = SAMPLE_BATCH_BYTES / size_of::<usize>();
        anyhow::ensure!(
            self.number_of_iterations > 0,
            "number_of_iterations must be positive"
        );
        anyhow::ensure!(
            self.number_of_sample_patches > 0,
            "number_of_sample_patches must be positive"
        );
        anyhow::ensure!(
            self.number_of_sample_patches <= max_samples,
            "number_of_sample_patches {} exceeds bounded capacity {max_samples}",
            self.number_of_sample_patches
        );
        anyhow::ensure!(
            self.sample_variance.is_finite() && self.sample_variance >= 0.0,
            "sample_variance must be finite and nonnegative, got {}",
            self.sample_variance
        );
        anyhow::ensure!(
            self.kernel_sigma.is_finite() && self.kernel_sigma > 0.0,
            "kernel_sigma must be finite and positive, got {}",
            self.kernel_sigma
        );
        let patch_diameter = self
            .patch_radius
            .checked_mul(2)
            .and_then(|diameter| diameter.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("patch_radius {} overflows", self.patch_radius))?;
        let dims = image.shape();
        let active_dims = if dims[0] == 1 { &dims[1..] } else { &dims[..] };
        anyhow::ensure!(
            active_dims
                .iter()
                .all(|&dimension| dimension >= patch_diameter),
            "patch diameter {patch_diameter} exceeds active image dimensions {active_dims:?}"
        );
        let (data, dims) = extract_vec(image)?;
        let result = self.run(&data, dims);
        Ok(rebuild(result, dims, image))
    }
    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let max_samples = SAMPLE_BATCH_BYTES / size_of::<usize>();
        anyhow::ensure!(
            self.number_of_iterations > 0,
            "number_of_iterations must be positive"
        );
        anyhow::ensure!(
            self.number_of_sample_patches > 0,
            "number_of_sample_patches must be positive"
        );
        anyhow::ensure!(
            self.number_of_sample_patches <= max_samples,
            "number_of_sample_patches {} exceeds bounded capacity {max_samples}",
            self.number_of_sample_patches
        );
        anyhow::ensure!(
            self.sample_variance.is_finite() && self.sample_variance >= 0.0,
            "sample_variance must be finite and nonnegative, got {}",
            self.sample_variance
        );
        anyhow::ensure!(
            self.kernel_sigma.is_finite() && self.kernel_sigma > 0.0,
            "kernel_sigma must be finite and positive, got {}",
            self.kernel_sigma
        );
        let patch_diameter = self
            .patch_radius
            .checked_mul(2)
            .and_then(|diameter| diameter.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("patch_radius {} overflows", self.patch_radius))?;
        let dims = image.shape();
        let active_dims = if dims[0] == 1 { &dims[1..] } else { &dims[..] };
        anyhow::ensure!(
            active_dims
                .iter()
                .all(|&dimension| dimension >= patch_diameter),
            "patch diameter {patch_diameter} exceeds active image dimensions {active_dims:?}"
        );
        let (data, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let result = self.run(&data, dims);
        crate::native_support::rebuild_image(result, dims, image, backend)
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
        self.pass_with_sample_budget(data, dims, SAMPLE_BATCH_BYTES)
    }

    fn pass_with_sample_budget(
        &self,
        data: &[f32],
        dims: [usize; 3],
        sample_budget: usize,
    ) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let ndim = if nz == 1 { 2 } else { 3 };
        let r = self.patch_radius as i64;
        let s2 = self.kernel_sigma * self.kernel_sigma;
        let nrr = self.number_of_sample_patches;
        let variance = self.sample_variance;
        let sample_radius = (variance.sqrt() * 2.5).floor() as i64;

        // Sizes in ITK index order (x, y, z).
        let sizes: [i64; 3] = [nx as i64, ny as i64, nz as i64];

        // ITK constructs the smooth-disc image in f32 before promoting the
        // sampled weights into its real-valued denoising arithmetic.
        let weights = smooth_disc_weights_sq(self.patch_radius, ndim);

        // Patch offsets in (dz, dy, dx), with the equivalent flat displacement.
        let row_stride = isize::try_from(nx).expect("invariant: image storage fits isize");
        let plane_stride = isize::try_from(ny * nx).expect("invariant: image storage fits isize");
        let mut offsets = {
            let mut natural_offsets = Vec::new();
            let mut wi = 0usize;
            let zr = if ndim == 3 { r } else { 0 };
            for dz in -zr..=zr {
                for dy in -r..=r {
                    for dx in -r..=r {
                        let flat = isize::try_from(dz).expect("invariant: patch offset fits isize")
                            * plane_stride
                            + isize::try_from(dy).expect("invariant: patch offset fits isize")
                                * row_stride
                            + isize::try_from(dx).expect("invariant: patch offset fits isize");
                        natural_offsets.push(PatchOffset {
                            coordinate: [dx, dy, dz],
                            displacement: flat,
                            weight: weights[wi],
                        });
                        wi += 1;
                    }
                }
            }
            itk_reduction_indices(natural_offsets.len())
                .map(|index| natural_offsets[index])
                .collect::<Vec<_>>()
        };
        // A zero smooth-disc weight contributes exactly +0.0 when every pixel
        // difference is finite, so removing it preserves every retained term
        // and their ITK reduction order. Non-finite input keeps the full
        // sequence because 0 * NaN/Inf participates in the filter's
        // propagation contract.
        if pixel_differences_are_finite(data) {
            offsets.retain(|offset| offset.weight != 0.0);
        }

        let idx = |x: i64, y: i64, z: i64| -> usize {
            (z as usize) * ny * nx + (y as usize) * nx + (x as usize)
        };

        let mut mt = ItkMt::new(0);
        let mut out = data.to_vec();
        let order = face_calculator_order(sizes, self.patch_radius as i64, ndim);
        let sample_index_bytes = size_of::<usize>();
        let sample_bytes_per_pixel = nrr
            .saturating_mul(sample_index_bytes)
            .max(sample_index_bytes);
        let pixels_per_batch = (sample_budget / sample_bytes_per_pixel).max(1);
        let batch_capacity = pixels_per_batch.min(order.len());
        let sample_capacity = batch_capacity
            .saturating_mul(nrr)
            .min(sample_budget / sample_index_bytes);
        let mut work = Vec::with_capacity(batch_capacity);
        let mut samples = Vec::with_capacity(sample_capacity);
        let mut values = Vec::with_capacity(batch_capacity);

        // Sampling remains serial and follows ImageBoundaryFacesCalculator order,
        // preserving ITK's shared RNG stream. Only independent pixel evaluation
        // is parallel; every pixel retains its original sample and reduction order.
        for pixel_batch in order.chunks(pixels_per_batch) {
            work.clear();
            samples.clear();
            for &(x, y, z) in pixel_batch {
                // Region constraint (itkPatchBasedDenoisingImageFilter
                // ComputeGradientJointEntropy). Per dimension:
                // rIndex = min(idx, radius); rEnd = max(idx, size-radius-1).
                let mut lo = [0i64; 3];
                let mut hi = [0i64; 3];
                let q0 = [x, y, z];
                for d in 0..ndim {
                    (lo[d], hi[d]) = sampling_interval(q0[d], sizes[d], r, sample_radius);
                }
                let region = (0..ndim)
                    .map(|d| (hi[d] - lo[d] + 1) as u64)
                    .product::<u64>();
                let sample_count = (nrr as u64).min(region) as usize;
                let first_sample = samples.len();
                for _ in 0..sample_count {
                    let qx = mt.gauss_int(lo[0], hi[0], x as f64, variance);
                    let qy = mt.gauss_int(lo[1], hi[1], y as f64, variance);
                    let qz = if ndim == 3 {
                        mt.gauss_int(lo[2], hi[2], z as f64, variance)
                    } else {
                        0
                    };
                    samples.push(idx(qx, qy, qz));
                }
                let center_index = idx(x, y, z);
                work.push(PixelWork {
                    position: q0,
                    center_index,
                    interior: x >= r
                        && y >= r
                        && z >= if ndim == 3 { r } else { 0 }
                        && x < sizes[0] - r
                        && y < sizes[1] - r
                        && z < sizes[2] - if ndim == 3 { r } else { 0 },
                    first_sample,
                    sample_count,
                });
            }

            values.resize(work.len(), 0.0);
            moirai::enumerate_mut_with::<moirai::Parallel, _, _>(
                &mut values,
                |work_index, value| {
                    let pixel = work[work_index];
                    let [x, y, z] = pixel.position;
                    let p_center_value = data[pixel.center_index];
                    let p_center = f64::from(p_center_value);
                    let mut sum_g = 0.0f64;
                    let mut grad = 0.0f64;

                    for &q_center_index in
                        &samples[pixel.first_sample..pixel.first_sample + pixel.sample_count]
                    {
                        let mut sq = 0.0f64;
                        if pixel.interior {
                            for offset in &offsets {
                                let vp = data
                                    [pixel.center_index.wrapping_add_signed(offset.displacement)];
                                let vq =
                                    data[q_center_index.wrapping_add_signed(offset.displacement)];
                                let diff = pixel_difference(vp, vq);
                                sq += offset.weight * diff * diff;
                            }
                        } else {
                            #[cfg(debug_assertions)]
                            let qz = q_center_index / (ny * nx);
                            #[cfg(debug_assertions)]
                            let q_plane_index = q_center_index % (ny * nx);
                            #[cfg(debug_assertions)]
                            let qy = q_plane_index / nx;
                            #[cfg(debug_assertions)]
                            let qx = q_plane_index % nx;
                            #[cfg(debug_assertions)]
                            let q_position = [
                                i64::try_from(qx).expect("invariant: x index fits i64"),
                                i64::try_from(qy).expect("invariant: y index fits i64"),
                                i64::try_from(qz).expect("invariant: z index fits i64"),
                            ];
                            for offset in &offsets {
                                let [dx, dy, dz] = offset.coordinate;
                                let (px, py, pz) = (x + dx, y + dy, z + dz);
                                if px < 0
                                    || py < 0
                                    || pz < 0
                                    || px >= sizes[0]
                                    || py >= sizes[1]
                                    || pz >= sizes[2]
                                {
                                    continue;
                                }
                                let vp = data[idx(px, py, pz)];
                                #[cfg(debug_assertions)]
                                {
                                    let [qx, qy, qz] = q_position;
                                    let q_offset_position = [qx + dx, qy + dy, qz + dz];
                                    debug_assert!(
                                        q_offset_position
                                            .iter()
                                            .zip(sizes)
                                            .all(|(&coordinate, size)| coordinate >= 0
                                                && coordinate < size),
                                        "sampled patch must be at least as in-bounds as the current patch"
                                    );
                                    debug_assert_eq!(
                                        idx(
                                            q_offset_position[0],
                                            q_offset_position[1],
                                            q_offset_position[2],
                                        ),
                                        q_center_index.wrapping_add_signed(offset.displacement)
                                    );
                                }
                                let vq =
                                    data[q_center_index.wrapping_add_signed(offset.displacement)];
                                let diff = pixel_difference(vp, vq);
                                sq += offset.weight * diff * diff;
                            }
                        }

                        let g = (-(sq / s2) / 2.0).exp();
                        sum_g += g;
                        grad += pixel_difference(p_center_value, data[q_center_index]) * g;
                    }

                    // ITK normalizes the entropy gradient before applying the
                    // smoothing step; this operation order fixes its rounding contract.
                    let normalized_gradient = grad / (sum_g + f64::MIN_POSITIVE * 100.0);
                    let update = normalized_gradient * 0.2;
                    *value = (p_center + update) as f32;
                },
            );

            for (pixel, &value) in work.iter().zip(&values) {
                let [x, y, z] = pixel.position;
                out[idx(x, y, z)] = value;
            }
        }

        out
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_patch_based_denoising.rs"]
mod tests;
