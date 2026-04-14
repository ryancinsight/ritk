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
//! 3. **Non-maximum suppression (NMS)**: For each voxel, quantise θ into one
//!    of four directions {0°, 45°, 90°, 135°} and suppress the voxel if its
//!    gradient magnitude is not a local maximum along that direction.
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

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use std::collections::VecDeque;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Canny edge detector for 3-D images.
///
/// Produces a binary edge map by applying Gaussian smoothing, gradient
/// estimation via central differences, non-maximum suppression along the
/// gradient direction, and double hysteresis thresholding with BFS
/// connectivity.
#[derive(Debug, Clone)]
pub struct CannyEdgeDetector {
    /// Standard deviation of the pre-smoothing Gaussian (physical units, mm).
    sigma: f64,
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
    /// * `sigma` — Standard deviation of the Gaussian smoothing kernel
    ///   (physical units).
    /// * `low_threshold` — Lower hysteresis threshold on gradient magnitude.
    /// * `high_threshold` — Upper hysteresis threshold on gradient magnitude.
    ///
    /// # Panics
    ///
    /// Panics if `low_threshold > high_threshold`.
    pub fn new(sigma: f64, low_threshold: f64, high_threshold: f64) -> Self {
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
    pub fn with_sigma(mut self, sigma: f64) -> Self {
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

    /// Apply the Canny edge detector to a 3-D image.
    ///
    /// Returns a binary `Image<B, 3>` with 1.0 at detected edge voxels and
    /// 0.0 elsewhere.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as
    /// `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let [nz, ny, nx] = dims;
        let spacing = image.spacing();
        let sp = [spacing[0], spacing[1], spacing[2]];

        // ── Stage 1: Gaussian smoothing ───────────────────────────────────
        let smoothed = {
            let gauss =
                crate::filter::GaussianFilter::<B>::new(vec![self.sigma, self.sigma, self.sigma]);
            gauss.apply(image)
        };

        let td = smoothed.data().clone().into_data();
        let vals: Vec<f32> = td
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("CannyEdgeDetector requires f32 data: {:?}", e))?
            .to_vec();

        // ── Stage 2: Gradient magnitude and direction ─────────────────────
        let (mag, dir_z, dir_y, dir_x) = gradient_3d(&vals, dims, sp);

        // ── Stage 3: Non-maximum suppression ──────────────────────────────
        let nms = non_maximum_suppression(&mag, &dir_z, &dir_y, &dir_x, dims);

        // ── Stage 4: Double hysteresis thresholding with BFS ──────────────
        let edges = hysteresis_threshold(
            &nms,
            dims,
            self.low_threshold as f32,
            self.high_threshold as f32,
        );

        // Build output image
        let n = nz * ny * nx;
        let out_vals: Vec<f32> = (0..n).map(|i| if edges[i] { 1.0 } else { 0.0 }).collect();

        let device = image.data().device();
        let out_td = TensorData::new(out_vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            image.origin().clone(),
            image.spacing().clone(),
            image.direction().clone(),
        ))
    }
}

// ── Gradient computation ──────────────────────────────────────────────────────

/// Compute gradient magnitude and per-component direction using central
/// differences with one-sided boundary handling.
///
/// Returns `(magnitude, dir_z, dir_y, dir_x)` where each direction component
/// is the normalised gradient component (unit vector) at each voxel.
fn gradient_3d(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let mut mag = vec![0.0_f32; n];
    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];

    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;

    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);

                let gz = if nz == 1 {
                    0.0
                } else if iz == 0 {
                    (data[idx(1, iy, ix)] - data[flat]) / sz
                } else if iz == nz - 1 {
                    (data[flat] - data[idx(nz - 2, iy, ix)]) / sz
                } else {
                    (data[idx(iz + 1, iy, ix)] - data[idx(iz - 1, iy, ix)]) / (2.0 * sz)
                };

                let gy = if ny == 1 {
                    0.0
                } else if iy == 0 {
                    (data[idx(iz, 1, ix)] - data[flat]) / sy
                } else if iy == ny - 1 {
                    (data[flat] - data[idx(iz, ny - 2, ix)]) / sy
                } else {
                    (data[idx(iz, iy + 1, ix)] - data[idx(iz, iy - 1, ix)]) / (2.0 * sy)
                };

                let gx = if nx == 1 {
                    0.0
                } else if ix == 0 {
                    (data[idx(iz, iy, 1)] - data[flat]) / sx
                } else if ix == nx - 1 {
                    (data[flat] - data[idx(iz, iy, nx - 2)]) / sx
                } else {
                    (data[idx(iz, iy, ix + 1)] - data[idx(iz, iy, ix - 1)]) / (2.0 * sx)
                };

                let m = (gz * gz + gy * gy + gx * gx).sqrt();
                mag[flat] = m;

                if m > 1e-10 {
                    dz[flat] = gz / m;
                    dy[flat] = gy / m;
                    dx[flat] = gx / m;
                }
            }
        }
    }

    (mag, dz, dy, dx)
}

// ── Non-maximum suppression ───────────────────────────────────────────────────

/// Suppress voxels whose gradient magnitude is not a local maximum along the
/// gradient direction.
///
/// For 3-D images, the gradient direction is projected onto the 13 unique axis
/// directions of the 26-connected neighbourhood. The two neighbours along the
/// dominant direction are examined; if either has a greater magnitude, the
/// centre voxel is suppressed to zero.
fn non_maximum_suppression(
    mag: &[f32],
    dir_z: &[f32],
    dir_y: &[f32],
    dir_x: &[f32],
    dims: [usize; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut out = vec![0.0_f32; n];

    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);
                let m = mag[flat];
                if m < 1e-10 {
                    continue;
                }

                // Quantise gradient direction to the nearest 3-D neighbour offset
                let (oz, oy, ox) = quantise_direction(dir_z[flat], dir_y[flat], dir_x[flat]);

                // Neighbour in the positive gradient direction
                let pz = iz as isize + oz;
                let py = iy as isize + oy;
                let px = ix as isize + ox;
                let m_pos = if pz >= 0
                    && pz < nz as isize
                    && py >= 0
                    && py < ny as isize
                    && px >= 0
                    && px < nx as isize
                {
                    mag[idx(pz as usize, py as usize, px as usize)]
                } else {
                    0.0
                };

                // Neighbour in the negative gradient direction
                let nz_ = iz as isize - oz;
                let ny_ = iy as isize - oy;
                let nx_ = ix as isize - ox;
                let m_neg = if nz_ >= 0
                    && nz_ < nz as isize
                    && ny_ >= 0
                    && ny_ < ny as isize
                    && nx_ >= 0
                    && nx_ < nx as isize
                {
                    mag[idx(nz_ as usize, ny_ as usize, nx_ as usize)]
                } else {
                    0.0
                };

                // Suppress if not a local maximum
                if m >= m_pos && m >= m_neg {
                    out[flat] = m;
                }
            }
        }
    }

    out
}

/// Quantise a 3-D unit gradient direction vector to the nearest integer offset
/// in {-1, 0, 1}³ \ {(0,0,0)}.
///
/// This effectively selects one of the 26 directions. We round each component
/// to the nearest of {-1, 0, 1} using a threshold of 0.4 to bias towards axis-
/// aligned directions and avoid ambiguity at diagonal boundaries.
fn quantise_direction(dz: f32, dy: f32, dx: f32) -> (isize, isize, isize) {
    let snap = |v: f32| -> isize {
        if v > 0.4 {
            1
        } else if v < -0.4 {
            -1
        } else {
            0
        }
    };

    let oz = snap(dz);
    let oy = snap(dy);
    let ox = snap(dx);

    // Fallback: if all components round to 0, pick the dominant axis
    if oz == 0 && oy == 0 && ox == 0 {
        let az = dz.abs();
        let ay = dy.abs();
        let ax = dx.abs();
        if az >= ay && az >= ax {
            return (if dz >= 0.0 { 1 } else { -1 }, 0, 0);
        } else if ay >= ax {
            return (0, if dy >= 0.0 { 1 } else { -1 }, 0);
        } else {
            return (0, 0, if dx >= 0.0 { 1 } else { -1 });
        }
    }

    (oz, oy, ox)
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
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }

    /// A uniform image has zero gradient everywhere at interior voxels, so the
    /// Canny detector must produce no edges in the interior.
    ///
    /// **Proof**: ∇(constant) = 0 ⇒ |∇I| = 0 < T_low ⇒ no edges.
    ///
    /// Note: The `GaussianFilter` uses burn's `conv1d` with zero-padding,
    /// which introduces boundary artefacts on a nonzero constant image
    /// (the convolution sees zeros outside the domain). Interior voxels
    /// beyond the kernel support are unaffected.
    #[test]
    fn test_uniform_image_no_edges() {
        let dims = [24, 24, 24];
        let vals = vec![100.0_f32; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims, [1.0, 1.0, 1.0]);

        let detector = CannyEdgeDetector::new(1.0, 0.1, 0.2);
        let result = detector.apply(&img).unwrap();
        let out = extract_vals(&result);

        // Check only interior voxels: skip a margin of 6 voxels per side to
        // avoid the zero-padding boundary artefacts from the Gaussian
        // convolution (kernel support ≈ 3σ = 3, plus gradient stencil = 1,
        // plus safety = 2).
        let margin = 6;
        let [nz, ny, nx] = dims;
        let mut interior_edge_count = 0usize;
        for iz in margin..nz - margin {
            for iy in margin..ny - margin {
                for ix in margin..nx - margin {
                    let flat = iz * ny * nx + iy * nx + ix;
                    if out[flat] > 0.5 {
                        interior_edge_count += 1;
                    }
                }
            }
        }
        assert!(
            interior_edge_count == 0,
            "uniform image should have no interior edges, but found {interior_edge_count} edge voxels"
        );
    }

    /// A step-edge image (half at 0.0, half at 100.0 along the x-axis) must
    /// produce edges near the boundary plane.
    ///
    /// **Derivation**: The gradient magnitude at the step is
    /// |∇I| ≈ 100 / (2·sx) at the boundary voxels, which should exceed any
    /// reasonable high threshold. After NMS, the boundary plane survives.
    #[test]
    fn test_step_edge_produces_edges() {
        let [nz, ny, nx] = [16usize, 16, 32];
        let n = nz * ny * nx;
        let half = nx / 2;
        let vals: Vec<f32> = (0..n)
            .map(|flat| {
                let ix = flat % nx;
                if ix < half {
                    0.0
                } else {
                    100.0
                }
            })
            .collect();
        let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

        // Use a small sigma and thresholds that comfortably detect the 100-unit
        // step while ignoring noise.
        let detector = CannyEdgeDetector::new(1.0, 2.0, 10.0);
        let result = detector.apply(&img).unwrap();
        let out = extract_vals(&result);

        // There should be at least some nonzero voxels near the boundary
        let edge_count: usize = out.iter().filter(|&&v| v > 0.5).count();
        assert!(
            edge_count > 0,
            "step-edge image should produce edges, but edge_count = 0"
        );

        // Edges should be concentrated near the step (ix ≈ half).
        // The Gaussian with sigma=1.0 spreads the edge; use a wide margin
        // that accounts for the smoothing kernel support plus NMS/hysteresis
        // propagation.
        let margin = 8;
        let mut edges_near_step = 0usize;
        let mut edges_far_from_step = 0usize;
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let flat = iz * ny * nx + iy * nx + ix;
                    if out[flat] > 0.5 {
                        if ix >= half.saturating_sub(margin) && ix <= half + margin {
                            edges_near_step += 1;
                        } else {
                            edges_far_from_step += 1;
                        }
                    }
                }
            }
        }
        assert!(
            edges_near_step > edges_far_from_step,
            "edges should be concentrated near the step: near = {edges_near_step}, \
             far = {edges_far_from_step}"
        );
    }

    /// Low threshold must be ≤ high threshold.
    #[test]
    #[should_panic(expected = "low_threshold")]
    fn test_invalid_thresholds() {
        let _ = CannyEdgeDetector::new(1.0, 5.0, 2.0);
    }
}
