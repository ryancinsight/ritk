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
            for ix in 0..nx {
                g1[base + ix] = d[ix];
            }
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
        let dims = image.shape();
        let [nz, ny, nx] = dims;
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("DistanceTransformImageFilter requires f32 data: {:?}", e))?;

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
        let dims = image.shape();
        let [nz, ny, nx] = dims;
        let td = image.data().clone().into_data();
        let vals: Vec<f32> = td
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("SignedDistanceTransformImageFilter requires f32 data: {:?}", e))?;

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
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        img.data().clone().into_data().into_vec::<f32>().unwrap()
    }

    // --- edt_3d unit tests ---------------------------------------------------

    #[test]
    fn edt_3d_single_foreground_voxel_at_origin() {
        // 5x5x5 volume, single foreground at (0,0,0)
        let dims = [5usize, 5, 5];
        let mut fg = vec![false; 5 * 5 * 5];
        fg[0] = true; // iz=0, iy=0, ix=0
        let dt = edt_3d(&fg, dims, [1.0, 1.0, 1.0]);
        // Voxel (0,0,0): distance 0
        assert!((dt[0] - 0.0).abs() < 1e-5);
        // Voxel (0,0,1): distance 1
        let idx = 0 * 25 + 0 * 5 + 1;
        assert!((dt[idx] - 1.0).abs() < 1e-4, "expected 1.0, got {}", dt[idx]);
        // Voxel (0,1,0): distance 1
        let idx = 0 * 25 + 1 * 5 + 0;
        assert!((dt[idx] - 1.0).abs() < 1e-4, "expected 1.0, got {}", dt[idx]);
        // Voxel (1,1,1): distance sqrt(3) ≈ 1.732
        let idx = 1 * 25 + 1 * 5 + 1;
        assert!((dt[idx] - 3.0_f64.sqrt() as f32).abs() < 1e-4,
            "expected sqrt(3), got {}", dt[idx]);
    }

    #[test]
    fn edt_3d_all_foreground_gives_zero_everywhere() {
        let dims = [4usize, 4, 4];
        let fg = vec![true; 64];
        let dt = edt_3d(&fg, dims, [1.0, 1.0, 1.0]);
        for (i, &v) in dt.iter().enumerate() {
            assert!((v - 0.0).abs() < 1e-5, "voxel {} expected 0, got {}", i, v);
        }
    }

    #[test]
    fn edt_3d_two_foreground_voxels_midpoint() {
        // 1×1×5 volume, foreground at ix=0 and ix=4
        let dims = [1usize, 1, 5];
        let fg = vec![true, false, false, false, true];
        let dt = edt_3d(&fg, dims, [1.0, 1.0, 1.0]);
        // Distances: 0, 1, 2, 1, 0
        let expected = [0.0f32, 1.0, 2.0, 1.0, 0.0];
        for (i, (&d, &e)) in dt.iter().zip(expected.iter()).enumerate() {
            assert!((d - e).abs() < 1e-4, "ix={}: expected {}, got {}", i, e, d);
        }
    }

    #[test]
    fn edt_3d_anisotropic_spacing_scales_distance() {
        // 1×1×3 with spacing sx=2.0; foreground at ix=0 only
        let dims = [1usize, 1, 3];
        let fg = vec![true, false, false];
        let dt = edt_3d(&fg, dims, [1.0, 1.0, 2.0]);
        // Distances: 0, 2, 4 (in mm with sx=2)
        assert!((dt[0] - 0.0).abs() < 1e-4);
        assert!((dt[1] - 2.0).abs() < 1e-4, "expected 2.0, got {}", dt[1]);
        assert!((dt[2] - 4.0).abs() < 1e-4, "expected 4.0, got {}", dt[2]);
    }

    // --- DistanceTransformImageFilter tests ----------------------------------

    #[test]
    fn unsigned_edt_filter_preserves_spatial_metadata() {
        let img = make_image(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 2, 2]);
        let out = DistanceTransformImageFilter::new().apply(&img).unwrap();
        assert_eq!(out.shape(), img.shape());
        assert_eq!(out.spacing(), img.spacing());
        assert_eq!(out.origin(), img.origin());
    }

    #[test]
    fn unsigned_edt_filter_foreground_voxel_receives_zero() {
        let img = make_image(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2, 2, 2],
        );
        let out = DistanceTransformImageFilter::new().apply(&img).unwrap();
        let v = voxels(&out);
        // iz=0,iy=0,ix=0 is foreground → distance 0
        assert!((v[0] - 0.0).abs() < 1e-4, "foreground voxel expected 0, got {}", v[0]);
    }

    #[test]
    fn unsigned_edt_filter_background_voxels_have_positive_distance() {
        // Single foreground at (0,0,0) in a 3×3×3 volume
        let mut vals = vec![0.0f32; 27];
        vals[0] = 1.0;
        let img = make_image(vals, [3, 3, 3]);
        let out = DistanceTransformImageFilter::new().apply(&img).unwrap();
        let v = voxels(&out);
        // All non-foreground voxels must have distance > 0
        for (i, &d) in v.iter().enumerate() {
            if i == 0 {
                assert!((d - 0.0).abs() < 1e-4);
            } else {
                assert!(d > 0.0, "voxel {} expected positive distance, got {}", i, d);
            }
        }
    }

    // --- SignedDistanceTransformImageFilter tests ----------------------------

    #[test]
    fn signed_edt_filter_inside_negative_outside_positive() {
        // 1×1×5: foreground is ix=[1,2,3], background is ix=[0,4]
        let vals = vec![0.0f32, 1.0, 1.0, 1.0, 0.0];
        let img = make_image(vals, [1, 1, 5]);
        let out = SignedDistanceTransformImageFilter::new().apply(&img).unwrap();
        let v = voxels(&out);
        // ix=0 (background): positive distance to nearest fg (ix=1) = 1
        assert!(v[0] > 0.0, "background expected positive, got {}", v[0]);
        assert!((v[0] - 1.0).abs() < 1e-4, "expected +1, got {}", v[0]);
        // ix=1 (foreground): negative distance to nearest bg (ix=0) = −1
        assert!(v[1] < 0.0, "foreground edge expected negative, got {}", v[1]);
        assert!((v[1] - (-1.0)).abs() < 1e-4, "expected -1, got {}", v[1]);
        // ix=2 (foreground center): distance to nearest bg is 2
        assert!(v[2] < 0.0, "foreground center expected negative, got {}", v[2]);
        assert!((v[2] - (-2.0)).abs() < 1e-4, "expected -2, got {}", v[2]);
        // ix=4 (background): positive 1
        assert!(v[4] > 0.0, "background expected positive, got {}", v[4]);
    }

    #[test]
    fn signed_edt_filter_preserves_spatial_metadata() {
        let img = make_image(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 2, 2]);
        let out = SignedDistanceTransformImageFilter::new().apply(&img).unwrap();
        assert_eq!(out.shape(), img.shape());
        assert_eq!(out.spacing(), img.spacing());
    }
}
