//! Canny edge detector for 3-D images.
//!
//! # Mathematical Specification
//!
//! The Canny edge detection algorithm (Canny 1986) produces a binary edge map
//! through four stages:
//!
//! 1. **Gaussian smoothing**: Convolve with G_σ to suppress noise.
//! 2. **Gradient computation**: Estimate ∇I via central finite differences,
//!    yielding gradient magnitude |∇I| and direction θ = atan2(g_y, g_x) at
//!    each voxel.
//! 3. **Non-maximum suppression (NMS)**: For each voxel, step ±1 pixel along
//!    the continuous gradient direction and evaluate the magnitude via
//!    trilinear interpolation; suppress the voxel if either interpolated
//!    neighbour exceeds its magnitude (sub-pixel NMS, no direction quantisation).
//! 4. **Double hysteresis thresholding**: Classify surviving voxels as
//!    *strong* (|∇I| ≥ T_high) or *weak* (T_low ≤ |∇I| < T_high). Retain
//!    weak edges only if they are connected to a strong edge via BFS on the
//!    26-connected neighbourhood.
//!
//! # Output
//!
//! Binary image: 1.0 at edge voxels, 0.0 elsewhere.
//!
//! # Complexity
//!
//! O(N) where N is the total voxel count (each stage is linear in N).
//!
//! # References
//!
//! - Canny, J. (1986). A computational approach to edge detection. *IEEE
//!   Transactions on Pattern Analysis and Machine Intelligence*, 8(6),
//!   pp. 679–698.

use super::GaussianSigma;
use ritk_image::native::Image;
use ritk_spatial::Spacing;
use std::collections::VecDeque;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Canny edge detector for 3-D images.
///
/// Produces a binary edge map by applying Gaussian smoothing, gradient
/// estimation via central differences, non-maximum suppression along the
/// gradient direction, and double hysteresis thresholding with BFS
/// connectivity.
///
/// The `sigma` parameter is a [`GaussianSigma`] newtype enforcing `sigma > 0`.
#[derive(Debug, Clone)]
pub struct CannyEdgeDetector {
    /// Standard deviation of the pre-smoothing Gaussian (physical units, mm).
    sigma: GaussianSigma,
    /// Lower hysteresis threshold applied to gradient magnitude.
    low_threshold: f64,
    /// Upper hysteresis threshold applied to gradient magnitude.
    high_threshold: f64,
}

impl CannyEdgeDetector {
    /// Create a new Canny edge detector.
    ///
    /// # Arguments
    ///
    /// * `sigma` — Standard deviation of the Gaussian smoothing kernel,
    ///   wrapped in [`GaussianSigma`] (physical units).
    /// * `low_threshold` — Lower hysteresis threshold on gradient magnitude.
    /// * `high_threshold` — Upper hysteresis threshold on gradient magnitude.
    ///
    /// # Panics
    ///
    /// Panics if `low_threshold > high_threshold`.
    pub fn new(sigma: GaussianSigma, low_threshold: f64, high_threshold: f64) -> Self {
        assert!(
            low_threshold <= high_threshold,
            "CannyEdgeDetector: low_threshold ({low_threshold}) must be <= high_threshold ({high_threshold})"
        );
        Self {
            sigma,
            low_threshold,
            high_threshold,
        }
    }

    /// Set the Gaussian sigma.
    pub fn with_sigma(mut self, sigma: GaussianSigma) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set the low hysteresis threshold.
    pub fn with_low_threshold(mut self, t: f64) -> Self {
        assert!(
            t <= self.high_threshold,
            "low_threshold ({t}) must be <= high_threshold ({})",
            self.high_threshold
        );
        self.low_threshold = t;
        self
    }

    /// Set the high hysteresis threshold.
    pub fn with_high_threshold(mut self, t: f64) -> Self {
        assert!(
            self.low_threshold <= t,
            "high_threshold ({t}) must be >= low_threshold ({})",
            self.low_threshold
        );
        self.high_threshold = t;
        self
    }

    /// Apply the Canny edge detector to a 3-D Coeus-native image.
    ///
    /// Alias for [`Self::apply_native`]; kept for API compatibility.
    pub fn apply<B>(
        &self,
        image: &Image<f32, B, 3>,
        backend: &B) -> anyhow::Result<Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        self.apply_native(image, backend)
    }

    /// Apply the Canny edge detector to a 3-D Coeus-native image.
    ///
    /// Smooths natively via the burn-free
    /// `gaussian_smooth_native_flat`
    /// core, then runs the identical gradient / non-maximum-suppression /
    /// hysteresis pipeline via the shared `canny_edges_flat` host core. No Burn
    /// tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Parity
    /// Stages 2–4 are bitwise-identical to the legacy Burn path (shared core).
    /// The native Gaussian (`convolve_zero_pad_3d`) and the legacy Burn
    /// Gaussian (`conv1d`) evaluate the same kernels but sum taps in different
    /// orders, so the pre-threshold magnitude field differs by accumulation
    /// rounding only (`O(width·ε·‖I‖∞)`); the binary edge map is unaffected
    /// except for voxels whose magnitude sits within that bound of a threshold.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &Image<f32, B, 3>,
        backend: &B) -> anyhow::Result<Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let sp = *image.spacing();
        let sigma = self.sigma.get();
        let spacing = [sp[0], sp[1], sp[2]];
        let low = self.low_threshold as f32;
        let high = self.high_threshold as f32;
        crate::native_support::map_flat_image(image, backend, move |vals, dims| {
            let smoothed = crate::gaussian::gaussian_smooth_native_flat(
                vals,
                dims,
                [sigma, sigma, sigma],
                spacing,
                // Matches `GaussianFilter::new`'s default max kernel width.
                32,
            );
            canny_edges_flat(&smoothed, dims, sp, low, high)
        })
    }
}

/// Substrate-agnostic host core for [`CannyEdgeDetector`] stages 2–4: gradient
/// magnitude/direction, non-maximum suppression along the continuous gradient,
/// and double-hysteresis thresholding with BFS connectivity, on an
/// already-smoothed flat z-major buffer. Returns the binary edge map (`1.0`
/// edge, `0.0` else). Single source of truth for the legacy Burn and
/// Coeus-native paths.
fn canny_edges_flat(
    smoothed: &[f32],
    dims: [usize; 3],
    spacing: Spacing<3>,
    low_threshold: f32,
    high_threshold: f32,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let (mag, dir_z, dir_y, dir_x) = compute_gradient(smoothed, dims, spacing);
    let nms = non_maximum_suppression(&mag, &dir_x, &dir_y, &dir_z, &dims);
    let edges = hysteresis_threshold(&nms, dims, low_threshold, high_threshold);
    let n = nz * ny * nx;
    (0..n).map(|i| if edges[i] { 1.0 } else { 0.0 }).collect()
}

/// Gradient magnitude threshold below which a pixel is treated as flat.
const NEAR_ZERO_MAG: f32 = 1e-10;

// ── Gradient computation ──────────────────────────────────────────────────────

/// Compute gradient magnitude and per-component direction using central
/// differences with one-sided boundary handling, parallelised over z-slices.
///
/// Returns `(magnitude, dir_z, dir_y, dir_x)` where each direction component
/// is the normalised gradient component (unit vector) at each voxel.
fn compute_gradient(
    data: &[f32],
    dims: [usize; 3],
    spacing: Spacing<3>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let slab = ny * nx;

    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;

    // Pack [mag, dz, dy, dx] per voxel interleaved; scatter to separate arrays
    // after the parallel pass.  Each z-slice chunk owns `slab * 4` elements so
    // chunks are disjoint and can be filled by independent threads.
    let mut combined = vec![0.0_f32; n * 4];

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut combined,
        slab * 4,
        |iz, iz_out| {
            for iy in 0..ny {
                for ix in 0..nx {
                    let local_flat = iy * nx + ix;
                    let flat = iz * slab + local_flat;

                    let gz = if nz == 1 {
                        0.0
                    } else if iz == 0 {
                        (data[flat + slab] - data[flat]) / sz
                    } else if iz == nz - 1 {
                        (data[flat] - data[flat - slab]) / sz
                    } else {
                        (data[flat + slab] - data[flat - slab]) / (2.0 * sz)
                    };

                    let gy = if ny == 1 {
                        0.0
                    } else if iy == 0 {
                        (data[flat + nx] - data[flat]) / sy
                    } else if iy == ny - 1 {
                        (data[flat] - data[flat - nx]) / sy
                    } else {
                        (data[flat + nx] - data[flat - nx]) / (2.0 * sy)
                    };

                    let gx = if nx == 1 {
                        0.0
                    } else if ix == 0 {
                        (data[flat + 1] - data[flat]) / sx
                    } else if ix == nx - 1 {
                        (data[flat] - data[flat - 1]) / sx
                    } else {
                        (data[flat + 1] - data[flat - 1]) / (2.0 * sx)
                    };

                    let m = (gz * gz + gy * gy + gx * gx).sqrt();
                    let out_base = local_flat * 4;
                    iz_out[out_base] = m;
                    if m > NEAR_ZERO_MAG {
                        iz_out[out_base + 1] = gz / m;
                        iz_out[out_base + 2] = gy / m;
                        iz_out[out_base + 3] = gx / m;
                    }
                }
            }
        },
    );

    // Scatter the interleaved combined output to separate arrays.
    let mut mag = vec![0.0_f32; n];
    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];
    for i in 0..n {
        let base = i * 4;
        mag[i] = combined[base];
        dz[i] = combined[base + 1];
        dy[i] = combined[base + 2];
        dx[i] = combined[base + 3];
    }
    (mag, dz, dy, dx)
}

// ── Non-maximum suppression ───────────────────────────────────────────────────

/// Trilinearly interpolate `vals` at continuous position `(fz, fy, fx)` within
/// `dims`, clamped to the valid voxel range.
///
/// Used by [`non_maximum_suppression`] to evaluate gradient magnitude at
/// sub-pixel positions along the continuous gradient direction.
fn trilinear_interp(vals: &[f32], dims: &[usize; 3], fz: f64, fy: f64, fx: f64) -> f32 {
    let [nz, ny, nx] = *dims;
    // Clamp to valid coordinate range; `.max(0.0)` guards the nz/ny/nx == 1 case
    // where the upper bound would otherwise be negative.
    let fz = fz.clamp(0.0, (nz as f64 - 1.0).max(0.0));
    let fy = fy.clamp(0.0, (ny as f64 - 1.0).max(0.0));
    let fx = fx.clamp(0.0, (nx as f64 - 1.0).max(0.0));
    let iz0 = fz as usize;
    let wz1 = fz - iz0 as f64;
    let wz0 = 1.0 - wz1;
    let iy0 = fy as usize;
    let wy1 = fy - iy0 as f64;
    let wy0 = 1.0 - wy1;
    let ix0 = fx as usize;
    let wx1 = fx - ix0 as f64;
    let wx0 = 1.0 - wx1;
    let iz1 = (iz0 + 1).min(nz - 1);
    let iy1 = (iy0 + 1).min(ny - 1);
    let ix1 = (ix0 + 1).min(nx - 1);
    let v = |az: usize, ay: usize, ax: usize| vals[az * ny * nx + ay * nx + ax] as f64;
    let interp = wz0
        * (wy0 * (wx0 * v(iz0, iy0, ix0) + wx1 * v(iz0, iy0, ix1))
            + wy1 * (wx0 * v(iz0, iy1, ix0) + wx1 * v(iz0, iy1, ix1)))
        + wz1
            * (wy0 * (wx0 * v(iz1, iy0, ix0) + wx1 * v(iz1, iy0, ix1))
                + wy1 * (wx0 * v(iz1, iy1, ix0) + wx1 * v(iz1, iy1, ix1)));
    interp as f32
}

/// Suppress voxels whose gradient magnitude is not a local maximum along the
/// continuous gradient direction, evaluated via trilinear interpolation.
///
/// For each voxel the magnitude at positions ±1 step along the unit gradient
/// vector `(gx, gy, gz)` is obtained by trilinear interpolation from the
/// magnitude field; the voxel is suppressed if either interpolated value
/// exceeds its own magnitude.  Parallelised over z-slices.
///
/// # Parameters
///
/// * `gx` — x-component of the normalised gradient direction.
/// * `gy` — y-component of the normalised gradient direction.
/// * `gz` — z-component of the normalised gradient direction.
fn non_maximum_suppression(
    mag: &[f32],
    gx: &[f32],
    gy: &[f32],
    gz: &[f32],
    dims: &[usize; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = *dims;
    let slab = ny * nx;
    let mut out = vec![0.0_f32; nz * slab];

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut out,
        slab,
        |iz, iz_out| {
            for iy in 0..ny {
                for ix in 0..nx {
                    let local = iy * nx + ix;
                    let flat = iz * slab + local;
                    let m = mag[flat];
                    if m < NEAR_ZERO_MAG {
                        continue;
                    }
                    let dgx = gx[flat] as f64;
                    let dgy = gy[flat] as f64;
                    let dgz = gz[flat] as f64;
                    let len = (dgx * dgx + dgy * dgy + dgz * dgz).sqrt();
                    if len < 1e-10 {
                        iz_out[local] = m;
                        continue;
                    }
                    let nx_ = dgx / len;
                    let ny_ = dgy / len;
                    let nz_ = dgz / len;
                    let fwd = trilinear_interp(
                        mag,
                        dims,
                        iz as f64 + nz_,
                        iy as f64 + ny_,
                        ix as f64 + nx_,
                    );
                    let bwd = trilinear_interp(
                        mag,
                        dims,
                        iz as f64 - nz_,
                        iy as f64 - ny_,
                        ix as f64 - nx_,
                    );
                    if m >= fwd && m >= bwd {
                        iz_out[local] = m;
                    }
                }
            }
        },
    );

    out
}

// ── Double hysteresis thresholding ────────────────────────────────────────────

/// Apply double hysteresis thresholding with BFS connectivity.
///
/// A voxel is marked as an edge if:
/// - Its NMS-surviving magnitude ≥ `high_threshold` (*strong* edge), OR
/// - Its magnitude ≥ `low_threshold` (*weak* edge) AND it is 26-connected to
///   at least one strong edge.
///
/// The BFS starts from all strong-edge voxels and propagates to adjacent weak
/// edges, promoting them to edges.
fn hysteresis_threshold(nms: &[f32], dims: [usize; 3], low: f32, high: f32) -> Vec<bool> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut edges = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();

    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };

    // Seed BFS with strong edges
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);
                if nms[flat] >= high {
                    edges[flat] = true;
                    queue.push_back(flat);
                }
            }
        }
    }

    // BFS: propagate to weak edges connected to strong edges
    while let Some(flat) = queue.pop_front() {
        let iz = flat / (ny * nx);
        let iy = (flat / nx) % ny;
        let ix = flat % nx;

        // 26-connected neighbourhood
        for dz in -1isize..=1 {
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dz == 0 && dy == 0 && dx == 0 {
                        continue;
                    }
                    let niz = iz as isize + dz;
                    let niy = iy as isize + dy;
                    let nix = ix as isize + dx;
                    if niz < 0
                        || niz >= nz as isize
                        || niy < 0
                        || niy >= ny as isize
                        || nix < 0
                        || nix >= nx as isize
                    {
                        continue;
                    }
                    let nflat = idx(niz as usize, niy as usize, nix as usize);
                    if !edges[nflat] && nms[nflat] >= low {
                        edges[nflat] = true;
                        queue.push_back(nflat);
                    }
                }
            }
        }
    }

    edges
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_canny.rs"]
mod tests_canny;
