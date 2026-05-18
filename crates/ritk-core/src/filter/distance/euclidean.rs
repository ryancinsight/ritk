//! Exact Euclidean distance transform using the Meijster–Roerdink–Hesselink (2000) algorithm.
//!
//! # Mathematical Specification
//!
//! Given a binary image `B : ℤ³ → {0,1}` (1 = foreground), the **unsigned** Euclidean
//! distance transform is:
//!
//! `EDT(x) = min_{y ∈ S} ||x − y||₂`
//!
//! where `S = { y : B(y) = 1 }` and distances are in physical units (mm).
//!
//! The **signed** transform uses the convention:
//!
//! - `x ∉ S` (background): `SEDT(x) = +EDT(x)` (positive = outside)
//! - `x ∈ S` (foreground): `SEDT(x) = −EDT_bg(x)` (negative = inside, where `EDT_bg` is
//!   distance to nearest background voxel)
//!
//! # Algorithm
//!
//! Meijster, A., Roerdink, J.B.T.M., Hesselink, W.H. (2000). "A General Algorithm for
//! Computing Distance Transforms in Linear Time." *Mathematical Morphology and its
//! Applications to Image and Signal Processing*, Springer, pp. 331–340.
//!
//! The algorithm decomposes the 3-D EDT into three sequential 1-D passes, each of which
//! applies a parabolic lower-envelope sweep in O(N). Total complexity: O(N) time, O(N) space.
//!
//! Pass 1 (X-axis): for each ZY row, compute squared 1-D distance to nearest foreground
//! voxel along X using a two-pass linear scan.
//!
//! Passes 2–3 (Y- and Z-axes): apply the Meijster parabolic lower-envelope algorithm.
//! At each voxel the squared distance accumulates the contribution of the previous pass,
//! exploiting the separability of the squared Euclidean norm:
//! `||p−q||² = (p_x−q_x)² + (p_y−q_y)² + (p_z−q_z)²`.
//!
//! # ITK / SimpleITK Parity
//!
//! | Filter                                 | ITK class                                |
//! |----------------------------------------|------------------------------------------|
//! | `DistanceTransformImageFilter`         | `DanielssonDistanceMapImageFilter`       |
//! | `SignedDistanceTransformImageFilter`   | `SignedMaurerDistanceMapImageFilter`     |

use crate::filter::ops::extract_vec;
use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

// ── Meijster core ─────────────────────────────────────────────────────────────

const INF: f64 = 1e30_f64;

/// 1-D first pass: squared 1-D distance to nearest foreground along one axis.
/// `fg_row[i] = true` iff voxel i is foreground.  `s` is the voxel spacing in mm.
/// Returns a `Vec<f64>` of length n where each entry is `(dist_mm)²`.
fn phase1_1d(fg_row: &[bool], n: usize, s: f64) -> Vec<f64> {
    let mut h = vec![INF; n];
    // Forward scan: accumulate distance from left-most foreground
    if fg_row[0] {
        h[0] = 0.0;
    }
    for i in 1..n {
        if fg_row[i] {
            h[i] = 0.0;
        } else if h[i - 1] < INF {
            h[i] = h[i - 1] + s;
        }
    }
    // Backward scan: correct for foreground to the right
    for i in (0..n - 1).rev() {
        let d = h[i + 1] + s;
        if d < h[i] {
            h[i] = d;
        }
    }
    // Square to get squared distances
    h.iter().map(|&v| v * v).collect()
}

/// Meijster separability function: voxel index `x` at which parabola centered at `u`
/// overtakes the one centered at `i`, given accumulated squared distances `gi` and `gu`
/// and spacing `s` in mm.
///
/// Derivation: solve `(x−i)²s² + gi = (x−u)²s² + gu` for x:
/// `x = (s²(u²−i²) + gu − gi) / (2s²(u−i))`
#[inline]
fn sep(i: isize, u: isize, gi: f64, gu: f64, s: f64) -> isize {
    let s2 = s * s;
    let num = s2 * (u * u - i * i) as f64 + gu - gi;
    let den = 2.0 * s2 * (u - i) as f64;
    (num / den).floor() as isize
}

/// Distance contribution of parabola centered at index `i` evaluated at `x`.
/// `gi` = accumulated squared distance from previous axis.
#[inline]
fn f_dt(x: isize, i: isize, gi: f64, s: f64) -> f64 {
    let d = (x - i) as f64 * s;
    d * d + gi
}

/// Meijster parabolic lower-envelope pass for one row.
/// Input: `g[i]` = accumulated squared distance from all previous axes at position i.
/// Output: `Vec<f64>` of same length with squared distances updated for this axis.
fn meijster_1d(g: &[f64], n: usize, s: f64) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return g.to_vec();
    }

    let mut s_stack = vec![0isize; n]; // parabola centers
    let mut t_stack = vec![0isize; n]; // left boundary of each parabola's dominance
    let mut q: usize = 0;

    // Seed the stack with the first finite entry.
    // INF parabolas (no foreground in upstream pass) are skipped — they never
    // contribute a finite minimum so they can never become dominant.
    let mut initialized = false;
    for u0 in 0..n {
        if g[u0] < INF {
            s_stack[0] = u0 as isize;
            t_stack[0] = 0;
            q = 0;
            initialized = true;
            // Process remaining voxels
            for u in (u0 + 1)..n {
                let gu = g[u];
                if gu >= INF {
                    continue; // skip all-background parabolas
                }
                loop {
                    if q == 0 {
                        if f_dt(t_stack[0], s_stack[0], g[s_stack[0] as usize], s)
                            >= f_dt(t_stack[0], u as isize, gu, s)
                        {
                            s_stack[0] = u as isize;
                            t_stack[0] = 0;
                        } else {
                            let w = sep(s_stack[0], u as isize, g[s_stack[0] as usize], gu, s)
                                .saturating_add(1);
                            if w < n as isize {
                                q += 1;
                                s_stack[q] = u as isize;
                                t_stack[q] = w;
                            }
                        }
                        break;
                    }
                    if f_dt(t_stack[q], s_stack[q], g[s_stack[q] as usize], s)
                        >= f_dt(t_stack[q], u as isize, gu, s)
                    {
                        q -= 1;
                    } else {
                        let w = sep(s_stack[q], u as isize, g[s_stack[q] as usize], gu, s)
                            .saturating_add(1);
                        if w < n as isize {
                            q += 1;
                            s_stack[q] = u as isize;
                            t_stack[q] = w;
                        }
                        break;
                    }
                }
            }
            break;
        }
    }
    // If no foreground found in this row, all distances remain INF.
    if !initialized {
        return vec![INF; n];
    }

    // Backward pass: assign distances
    let mut dt = vec![0.0f64; n];
    for u in (0..n).rev() {
        dt[u] = f_dt(u as isize, s_stack[q], g[s_stack[q] as usize], s);
        if q > 0 && u as isize == t_stack[q] {
            q -= 1;
        }
    }
    dt
}

/// Compute the unsigned squared Euclidean distance transform for a 3-D binary volume.
/// `fg[iz*ny*nx + iy*nx + ix] = true` for foreground voxels.
/// `spacing = [sz, sy, sx]` in mm.
/// Returns `Vec<f32>` of Euclidean distances (not squared) in mm.
pub(crate) fn edt_3d(fg: &[bool], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let [sz, sy, sx] = spacing;
    let n_total = nz * ny * nx;

    // Phase 1: 1-D DT along X for each (iz, iy) row
    let mut g1 = vec![INF; n_total];
    for iz in 0..nz {
        for iy in 0..ny {
            let base = iz * ny * nx + iy * nx;
            let row: Vec<bool> = (0..nx).map(|ix| fg[base + ix]).collect();
            let d = phase1_1d(&row, nx, sx);
            g1[base..base + nx].copy_from_slice(&d[..nx]);
        }
    }

    // Phase 2: parabolic envelope along Y for each (iz, ix) column
    let mut g2 = vec![INF; n_total];
    for iz in 0..nz {
        for ix in 0..nx {
            let col: Vec<f64> = (0..ny).map(|iy| g1[iz * ny * nx + iy * nx + ix]).collect();
            let d = meijster_1d(&col, ny, sy);
            for iy in 0..ny {
                g2[iz * ny * nx + iy * nx + ix] = d[iy];
            }
        }
    }

    // Phase 3: parabolic envelope along Z for each (iy, ix) column
    let mut edt2 = vec![INF; n_total];
    for iy in 0..ny {
        for ix in 0..nx {
            let col: Vec<f64> = (0..nz).map(|iz| g2[iz * ny * nx + iy * nx + ix]).collect();
            let d = meijster_1d(&col, nz, sz);
            for iz in 0..nz {
                edt2[iz * ny * nx + iy * nx + ix] = d[iz];
            }
        }
    }

    edt2.iter().map(|&v| v.sqrt() as f32).collect()
}

// ── DistanceTransformImageFilter ─────────────────────────────────────────────

/// Unsigned Euclidean distance transform.
///
/// For each voxel, computes the physical distance (mm) to the nearest voxel
/// with intensity strictly greater than `threshold` (default 0.5, appropriate
/// for binary images).
///
/// Foreground voxels receive distance 0.
///
/// # Mathematical Specification
///
/// `out(x) = min_{y ∈ S} ||x − y||₂`  where `S = { y : in(y) > threshold }`
///
/// # ITK Parity
///
/// `DanielssonDistanceMapImageFilter` with `UseImageSpacing = true`.
///
/// # Complexity
///
/// O(N) time via Meijster 2000; O(N) additional space.
#[derive(Debug, Clone)]
pub struct DistanceTransformImageFilter {
    /// Intensity threshold separating background (≤ threshold) from foreground (> threshold).
    pub threshold: f32,
}

impl Default for DistanceTransformImageFilter {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl DistanceTransformImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;

        let fg: Vec<bool> = vals.iter().map(|&v| v > self.threshold).collect();
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];
        let result = edt_3d(&fg, dims, spacing);

        let device = image.data().device();
        let td_out = TensorData::new(result, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<B, 3>::from_data(td_out, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── SignedDistanceTransformImageFilter ───────────────────────────────────────

/// Signed Euclidean distance transform.
///
/// Convention (matches ITK `SignedMaurerDistanceMapImageFilter`):
/// - Background voxels: `+dist` (positive = outside the object)
/// - Foreground voxels: `−dist` (negative = inside the object, distance to nearest background)
///
/// # Mathematical Specification
///
/// `SEDT(x) = EDT_bg(x)` if `in(x) ≤ threshold` (outside object)
/// `SEDT(x) = −EDT_fg(x)` if `in(x) > threshold` (inside object)
///
/// where `EDT_bg` = distance to nearest background, `EDT_fg` = distance to nearest foreground.
///
/// # ITK Parity
///
/// `SignedMaurerDistanceMapImageFilter` with `UseImageSpacing = true`,
/// `InsideIsPositive = false`.
#[derive(Debug, Clone)]
pub struct SignedDistanceTransformImageFilter {
    /// Intensity threshold separating background from foreground.
    pub threshold: f32,
}

impl Default for SignedDistanceTransformImageFilter {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl SignedDistanceTransformImageFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;

        let fg: Vec<bool> = vals.iter().map(|&v| v > self.threshold).collect();
        let bg: Vec<bool> = fg.iter().map(|&b| !b).collect();
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];

        // EDT from each voxel to nearest foreground (background voxels get positive value)
        let edt_to_fg = edt_3d(&fg, dims, spacing);
        // EDT from each voxel to nearest background (foreground voxels get positive value)
        let edt_to_bg = edt_3d(&bg, dims, spacing);

        // Signed: outside (+) = edt_to_fg, inside (−) = −edt_to_bg
        let result: Vec<f32> = fg
            .iter()
            .zip(edt_to_fg.iter())
            .zip(edt_to_bg.iter())
            .map(|((&is_fg, &d_fg), &d_bg)| if is_fg { -d_bg } else { d_fg })
            .collect();

        let device = image.data().device();
        let td_out = TensorData::new(result, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<B, 3>::from_data(td_out, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_euclidean.rs"]
mod tests_euclidean;
