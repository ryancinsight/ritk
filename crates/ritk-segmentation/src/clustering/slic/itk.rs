//! ITK-convention SLIC super-pixel core (`itk::SLICImageFilter` parity).
//!
//! This is a distinct algorithm from the Achanta variant in the parent module:
//! it is parameterised by a per-axis **super-grid size** (the k-means grid step)
//! rather than a target super-pixel count, uses **raw** intensity differences
//! (no `m_c` normalisation), and initialises cluster centres from a shrink of
//! the input. It reproduces ITK's deterministic core — the configuration
//! `enforceConnectivity = false`, `initializationPerturbation = false` — which is
//! a fixed-count Lloyd iteration with no order-sensitive post-processing, hence
//! bit-reproducible against `sitk.SLIC` with those flags.
//!
//! # Mathematical specification
//!
//! For a `D`-dimensional image with per-axis grid step `g_d` and spatial
//! proximity weight `m`, cluster centres are placed on the shrink grid: centre
//! `(r_0,…,r_{D-1})` has continuous-index position `c_d = (g_d−1)/2 + r_d·g_d`
//! and intensity `I(s)` sampled at integer index `s_d = r_d·g_d + g_d/2`. The
//! number of centres is `∏_d ⌊shape_d / g_d⌋` in row-major (axis-0-outer) scan
//! order, which fixes the output label numbering.
//!
//! The squared distance from voxel at integer index **p** (intensity `I`) to a
//! centre with intensity `I_c` and position **c** is
//!
//! D² = (I − I_c)² + Σ_d ((p_d − c_d) · m/g_d)²    (ITK `Distance`, raw colour).
//!
//! Each of `max_iterations` iterations assigns every voxel within a per-centre
//! search window `[round(c_d) − g_d, round(c_d) + g_d]` to the nearest centre
//! (strict `<`, so the lowest-index centre wins ties — matching ITK's scan-order
//! overwrite), then recomputes each centre as the mean of its members. The loop
//! is fixed-count (no convergence break), exactly as `itkSLICImageFilter`.
//!
//! `RoundHalfIntegerUp` (round halves up) is used for the window/sample index,
//! matching ITK's `Math::RoundHalfIntegerUp`.
//!
//! # Validation scope
//!
//! Validated **label-for-label exact** against `sitk.SLIC` (deterministic core)
//! for super-grid sizes that **evenly divide** each axis — confirmed in 2-D and
//! 3-D over multiple images (`tests_slic_itk.rs`). When `g_d` does not divide
//! `shape_d`, ITK's `ShrinkImageFilter` places the trailing cluster centre with
//! a remainder-handling convention this core does not yet reproduce, so the
//! partition diverges; that case is the documented open follow-up. Default-on
//! `enforceConnectivity` / `initializationPerturbation` (order-sensitive
//! post-processing) are likewise out of scope here.

/// Round half-integer values up, matching ITK's `Math::RoundHalfIntegerUp`.
#[inline]
fn round_half_up(v: f64) -> i64 {
    (v + 0.5).floor() as i64
}

/// A cluster centre: one intensity component plus `D` continuous-index
/// positions.
struct Center {
    intensity: f64,
    pos: Vec<f64>,
}

/// ITK-convention SLIC core over a flat row-major `f32` buffer.
///
/// `super_grid` holds the per-axis grid step `g_d` (length `shape.len()`);
/// `proximity_weight` is ITK's `m_SpatialProximityWeight`. Returns a flat label
/// buffer (`0..K−1` as `f32`) in centre scan order. Reproduces
/// `sitk.SLIC(enforceConnectivity=False, initializationPerturbation=False)`.
pub fn slic_itk_impl(
    data: &[f32],
    shape: &[usize],
    super_grid: &[usize],
    proximity_weight: f64,
    max_iterations: usize,
) -> Vec<f32> {
    let ndim = shape.len();
    let n: usize = shape.iter().product();
    if n == 0 || ndim == 0 {
        return vec![0.0_f32; n];
    }
    let g: Vec<usize> = (0..ndim).map(|d| super_grid[d].max(1)).collect();

    // Row-major strides (axis 0 outermost).
    let mut stride = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        stride[d] = stride[d + 1] * shape[d + 1];
    }
    let flat = |idx: &[usize]| -> usize { (0..ndim).map(|d| idx[d] * stride[d]).sum() };

    // ── Initialise centres on the shrink grid (scan order fixes labels) ──────
    let grid_pts: Vec<usize> = (0..ndim).map(|d| (shape[d] / g[d]).max(1)).collect();
    let k: usize = grid_pts.iter().product();
    if k <= 1 {
        return vec![0.0_f32; n];
    }
    let scale: Vec<f64> = (0..ndim).map(|d| proximity_weight / g[d] as f64).collect();

    // Row-major strides over the centre grid (axis 0 outermost), so linear
    // centre index → multi-index preserves ITK's scan-order label numbering.
    let mut grid_stride = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        grid_stride[d] = grid_stride[d + 1] * grid_pts[d + 1];
    }
    let mut centers: Vec<Center> = Vec::with_capacity(k);
    for ci in 0..k {
        let mut pos = vec![0.0_f64; ndim];
        let mut sample = vec![0usize; ndim];
        let mut rem = ci;
        for d in 0..ndim {
            let gd = rem / grid_stride[d];
            rem %= grid_stride[d];
            pos[d] = (g[d] as f64 - 1.0) / 2.0 + gd as f64 * g[d] as f64;
            sample[d] = (gd * g[d] + g[d] / 2).min(shape[d] - 1);
        }
        centers.push(Center {
            intensity: data[flat(&sample)] as f64,
            pos,
        });
    }

    // ── Fixed-count Lloyd iteration ──────────────────────────────────────────
    let mut labels = vec![0u32; n];
    let mut dist = vec![f64::MAX; n];
    let mut idx = vec![0usize; ndim];
    let mut lo = vec![0usize; ndim];
    let mut hi = vec![0usize; ndim];
    for _ in 0..max_iterations {
        dist.iter_mut().for_each(|x| *x = f64::MAX);
        labels.iter_mut().for_each(|x| *x = 0);
        for (ci, c) in centers.iter().enumerate() {
            // Search window [round(c_d) − g_d, round(c_d) + g_d] ∩ image.
            for d in 0..ndim {
                let center = round_half_up(c.pos[d]);
                lo[d] = (center - g[d] as i64).max(0) as usize;
                hi[d] = ((center + g[d] as i64).min(shape[d] as i64 - 1)).max(0) as usize;
            }
            idx.copy_from_slice(&lo);
            loop {
                let fi = flat(&idx);
                let mut dd = {
                    let di = data[fi] as f64 - c.intensity;
                    di * di
                };
                for d in 0..ndim {
                    let dp = (idx[d] as f64 - c.pos[d]) * scale[d];
                    dd += dp * dp;
                }
                if dd < dist[fi] {
                    dist[fi] = dd;
                    labels[fi] = ci as u32;
                }
                // Odometer over the window (axis 0 outermost; innermost fastest).
                let mut d = ndim;
                let carry = loop {
                    if d == 0 {
                        break true;
                    }
                    d -= 1;
                    idx[d] += 1;
                    if idx[d] <= hi[d] {
                        break false;
                    }
                    idx[d] = lo[d];
                };
                if carry {
                    break;
                }
            }
        }

        // ── Update centres to the mean of their members ──────────────────────
        let mut sum_i = vec![0.0_f64; k];
        let mut sum_p = vec![0.0_f64; k * ndim];
        let mut count = vec![0usize; k];
        let mut p = vec![0usize; ndim];
        p.iter_mut().for_each(|x| *x = 0);
        for fi in 0..n {
            // Recover multi-index from flat (row-major).
            let mut rem = fi;
            for d in 0..ndim {
                p[d] = rem / stride[d];
                rem %= stride[d];
            }
            let ci = labels[fi] as usize;
            sum_i[ci] += data[fi] as f64;
            for d in 0..ndim {
                sum_p[ci * ndim + d] += p[d] as f64;
            }
            count[ci] += 1;
        }
        for ci in 0..k {
            if count[ci] == 0 {
                continue;
            }
            let inv = 1.0 / count[ci] as f64;
            centers[ci].intensity = sum_i[ci] * inv;
            for d in 0..ndim {
                centers[ci].pos[d] = sum_p[ci * ndim + d] * inv;
            }
        }
    }

    labels.iter().map(|&l| l as f32).collect()
}

#[cfg(test)]
#[path = "tests_slic_itk.rs"]
mod tests;
