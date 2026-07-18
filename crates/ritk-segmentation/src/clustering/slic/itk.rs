//! ITK-convention SLIC super-pixel core (`itk::SLICImageFilter` parity).
//!
//! This is a distinct algorithm from the Achanta variant in the parent module:
//! it is parameterised by a per-axis **super-grid size** (the k-means grid step)
//! rather than a target super-pixel count, uses **raw** intensity differences
//! (no `m_c` normalisation), and initialises cluster centres from a shrink of
//! the input. It reproduces ITK's deterministic core â€” the configuration
//! `enforceConnectivity = false`, `initializationPerturbation = false` â€” which is
//! a fixed-count Lloyd iteration with no order-sensitive post-processing, hence
//! bit-reproducible against `sitk.SLIC` with those flags.
//!
//! # Mathematical specification
//!
//! For a `D`-dimensional image with per-axis grid step `g_d` and spatial
//! proximity weight `m`, cluster centres are placed on the shrink grid: centre
//! `(r_0,â€¦,r_{D-1})` has continuous-index position `c_d = (g_dâˆ’1)/2 + r_dÂ·g_d`
//! and intensity `I(s)` sampled at integer index `s_d = r_dÂ·g_d + g_d/2`. The
//! number of centres is `âˆ_d âŒŠshape_d / g_dâŒ‹` in row-major (axis-0-outer) scan
//! order, which fixes the output label numbering.
//!
//! The squared distance from voxel at integer index **p** (intensity `I`) to a
//! centre with intensity `I_c` and position **c** is
//!
//! DÂ² = (I âˆ’ I_c)Â² + Î£_d ((p_d âˆ’ c_d) Â· m/g_d)Â²    (ITK `Distance`, raw colour).
//!
//! Each of `max_iterations` iterations assigns every voxel within a per-centre
//! search window `[round(c_d) âˆ’ g_d, round(c_d) + g_d]` to the nearest centre
//! (strict `<`, so the lowest-index centre wins ties â€” matching ITK's scan-order
//! overwrite), then recomputes each centre as the mean of its members. The loop
//! is fixed-count (no convergence break), exactly as `itkSLICImageFilter`.
//!
//! `RoundHalfIntegerUp` (round halves up) is used for the window/sample index,
//! matching ITK's `Math::RoundHalfIntegerUp`.
//!
//! # Validation scope
//!
//! Validated **label-for-label exact** against `sitk.SLIC` in 2-D and 3-D over
//! multiple images, for both evenly- and non-evenly-dividing super-grids and for
//! **both** the deterministic core and the sitk-default configuration
//! (`initializationPerturbation` + `enforceConnectivity` on) â€” see
//! `tests_slic_itk.rs`. The centered shrink origin handles the remainder case;
//! perturbation and the two-phase connectivity relabelling reproduce ITK's
//! order-sensitive post-processing exactly.

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
/// `proximity_weight` is ITK's `m_SpatialProximityWeight`. `perturbation` moves
/// each initial centre to the lowest-gradient voxel in its 3^D neighbourhood
/// (ITK `initializationPerturbation`, applied only when every `g_d â‰¥ 3`);
/// `enforce_connectivity` relabels sub-threshold connected fragments into an
/// adjacent super-pixel (ITK `enforceConnectivity`). Returns a flat label buffer
/// (`0..Kâˆ’1` as `f32`) in centre scan order, matching `sitk.SLIC` with the
/// corresponding flags.
#[allow(clippy::too_many_arguments)]
pub(crate) fn slic_itk_impl(
    data: &[f32],
    shape: &[usize],
    super_grid: &[usize],
    proximity_weight: f64,
    max_iterations: usize,
    perturbation: bool,
    enforce_connectivity: bool,
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

    // â”€â”€ Initialise centres on the shrink grid (scan order fixes labels) â”€â”€â”€â”€â”€â”€
    let grid_pts: Vec<usize> = (0..ndim).map(|d| (shape[d] / g[d]).max(1)).collect();
    let k: usize = grid_pts.iter().product();
    if k <= 1 {
        return vec![0.0_f32; n];
    }
    let scale: Vec<f64> = (0..ndim).map(|d| proximity_weight / g[d] as f64).collect();

    // Row-major strides over the centre grid (axis 0 outermost), so linear
    // centre index â†’ multi-index preserves ITK's scan-order label numbering.
    let mut grid_stride = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        grid_stride[d] = grid_stride[d + 1] * grid_pts[d + 1];
    }
    // ITK ShrinkImageFilter places the shrunk grid with a *centered* origin (the
    // output centre maps to the input centre), so the continuous-index position
    // of grid point j on axis d is `out_origin_d + jÂ·g_d` with
    // `out_origin_d = (shape_dâˆ’1)/2 âˆ’ g_dÂ·(grid_pts_dâˆ’1)/2`. This reduces to
    // `(g_dâˆ’1)/2` only when g_d divides shape_d; the general form is required
    // for non-evenly-dividing super-grids.
    let out_origin: Vec<f64> = (0..ndim)
        .map(|d| (shape[d] as f64 - 1.0) / 2.0 - g[d] as f64 * (grid_pts[d] as f64 - 1.0) / 2.0)
        .collect();
    let mut centers: Vec<Center> = Vec::with_capacity(k);
    for ci in 0..k {
        let mut pos = vec![0.0_f64; ndim];
        let mut sample = vec![0usize; ndim];
        let mut rem = ci;
        for d in 0..ndim {
            let gd = rem / grid_stride[d];
            rem %= grid_stride[d];
            pos[d] = out_origin[d] + gd as f64 * g[d] as f64;
            // Intensity is the shrunk pixel value: input sampled at the nearest
            // index to the centre's continuous position (ITK RoundHalfIntegerUp).
            sample[d] = round_half_up(pos[d]).clamp(0, shape[d] as i64 - 1) as usize;
        }
        centers.push(Center {
            intensity: data[flat(&sample)] as f64,
            pos,
        });
    }

    // â”€â”€ Optional perturbation: move each centre to the lowest-gradient voxel â”€â”€
    // in its 3^D neighbourhood (central differences, ZeroFluxNeumann edge clamp).
    // ITK gates this on every super-grid factor being â‰¥ 3.
    if perturbation && g.iter().all(|&gd| gd >= 3) {
        let clamp_get = |p: &[i64]| -> f64 {
            let mut q = vec![0usize; ndim];
            for d in 0..ndim {
                q[d] = p[d].clamp(0, shape[d] as i64 - 1) as usize;
            }
            data[flat(&q)] as f64
        };
        let mut nb = vec![0i64; ndim];
        let mut p = vec![0i64; ndim];
        for c in centers.iter_mut() {
            let center: Vec<i64> = (0..ndim).map(|d| round_half_up(c.pos[d])).collect();
            let mut min_g = f64::MAX;
            let mut min_idx: Vec<i64> = center.clone();
            // Odometer over the 3^D offset cube {-1,0,1}^D.
            let mut off = vec![-1i64; ndim];
            loop {
                let mut in_bounds = true;
                for d in 0..ndim {
                    nb[d] = center[d] + off[d];
                    if nb[d] < 0 || nb[d] >= shape[d] as i64 {
                        in_bounds = false;
                    }
                }
                if in_bounds {
                    let mut gnorm = 0.0;
                    for d in 0..ndim {
                        p.copy_from_slice(&nb);
                        p[d] = nb[d] + 1;
                        let fwd = clamp_get(&p);
                        p[d] = nb[d] - 1;
                        let bwd = clamp_get(&p);
                        let gr = (fwd - bwd) / 2.0;
                        gnorm += gr * gr;
                    }
                    if gnorm < min_g {
                        min_g = gnorm;
                        min_idx.copy_from_slice(&nb);
                    }
                }
                let mut d = ndim;
                let carry = loop {
                    if d == 0 {
                        break true;
                    }
                    d -= 1;
                    off[d] += 1;
                    if off[d] <= 1 {
                        break false;
                    }
                    off[d] = -1;
                };
                if carry {
                    break;
                }
            }
            let mi: Vec<usize> = min_idx.iter().map(|&v| v as usize).collect();
            c.intensity = data[flat(&mi)] as f64;
            c.pos
                .iter_mut()
                .zip(&min_idx)
                .for_each(|(p, &m)| *p = m as f64);
        }
    }

    // â”€â”€ Fixed-count Lloyd iteration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let mut labels = vec![0u32; n];
    let mut dist = vec![f64::MAX; n];
    let mut idx = vec![0usize; ndim];
    let mut lo = vec![0usize; ndim];
    let mut hi = vec![0usize; ndim];
    // Per-centre accumulators reused across iterations (cleared, not re-allocated).
    let mut sum_i = vec![0.0_f64; k];
    let mut sum_p = vec![0.0_f64; k * ndim];
    let mut count = vec![0usize; k];
    for _ in 0..max_iterations {
        dist.iter_mut().for_each(|x| *x = f64::MAX);
        labels.iter_mut().for_each(|x| *x = 0);
        for (ci, c) in centers.iter().enumerate() {
            // Search window [round(c_d) âˆ’ g_d, round(c_d) + g_d] âˆ© image.
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

        // â”€â”€ Update centres to the mean of their members â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        sum_i.iter_mut().for_each(|x| *x = 0.0);
        sum_p.iter_mut().for_each(|x| *x = 0.0);
        count.iter_mut().for_each(|x| *x = 0);
        // Walk voxels in flat order, tracking the multi-index with an odometer
        // (innermost axis fastest) instead of decoding it with `ndim` divisions
        // per voxel â€” same accumulation, no per-voxel division.
        let mut p = vec![0usize; ndim];
        for fi in 0..n {
            let ci = labels[fi] as usize;
            sum_i[ci] += data[fi] as f64;
            for d in 0..ndim {
                sum_p[ci * ndim + d] += p[d] as f64;
            }
            count[ci] += 1;
            // Increment the multi-index (row-major: last axis fastest).
            let mut d = ndim;
            while d > 0 {
                d -= 1;
                p[d] += 1;
                if p[d] < shape[d] {
                    break;
                }
                p[d] = 0;
            }
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

    if enforce_connectivity {
        enforce_slic_connectivity(&mut labels, shape, &stride, &g, &centers, k);
    }

    labels.iter().map(|&l| l as f32).collect()
}

/// Decode a flat row-major index into its multi-index.
#[inline]
fn decode(fi: usize, stride: &[usize], out: &mut [usize]) {
    let mut rem = fi;
    for d in 0..out.len() {
        out[d] = rem / stride[d];
        rem %= stride[d];
    }
}

/// ITK `enforceConnectivity`: relabel sub-`minSuperSize` fragments so every
/// label is a single face-connected region. Phase 1 marks each cluster's main
/// component (the one reachable from its centroid, kept only if â‰¥ minSuperSize);
/// phase 2 raster-scans the unmarked fragments, giving each a fresh label if
/// large enough else merging it into the previous raster label
/// (`minSuperSize = âˆ g_d / 4`). Face-connectivity (Â±1 per axis); flood order
/// does not affect the labelling (only the member set and raster `prev_label`).
fn enforce_slic_connectivity(
    labels: &mut [u32],
    shape: &[usize],
    stride: &[usize],
    g: &[usize],
    centers: &[Center],
    k: usize,
) {
    let ndim = shape.len();
    let n = labels.len();
    let flat = |idx: &[usize]| -> usize { (0..ndim).map(|d| idx[d] * stride[d]).sum() };
    let min_super = (g.iter().product::<usize>() / 4).max(1);
    let mut marker = vec![0u8; n];
    let mut comp: Vec<usize> = Vec::new();
    let mut p = vec![0usize; ndim];
    let mut q = vec![0usize; ndim];

    // Flood the face-connected component of voxels with `labels == req` and
    // `marker == 0` from `seed`, marking them; if `relabel_to` is Some, also set
    // their label. Returns the component as flat indices in `comp`.
    macro_rules! flood {
        ($seed:expr, $req:expr, $relabel_to:expr) => {{
            comp.clear();
            marker[$seed] = 1;
            if let Some(out) = $relabel_to {
                labels[$seed] = out;
            }
            comp.push($seed);
            let mut head = 0;
            while head < comp.len() {
                let cur = comp[head];
                head += 1;
                decode(cur, stride, &mut p);
                for d in 0..ndim {
                    for s in [1i64, -1] {
                        let v = p[d] as i64 + s;
                        if v < 0 || v >= shape[d] as i64 {
                            continue;
                        }
                        q.copy_from_slice(&p);
                        q[d] = v as usize;
                        let nf = flat(&q);
                        if labels[nf] == $req && marker[nf] == 0 {
                            marker[nf] = 1;
                            if let Some(out) = $relabel_to {
                                labels[nf] = out;
                            }
                            comp.push(nf);
                        }
                    }
                }
            }
        }};
    }

    // â”€â”€ Phase 1: mark each cluster's centroid-connected main component â”€â”€â”€â”€â”€â”€â”€â”€
    let half: Vec<i64> = g.iter().map(|&gd| (gd / 2) as i64).collect();
    let mut off = vec![0i64; ndim];
    for (ci, center) in centers.iter().enumerate().take(k) {
        let cidx: Vec<usize> = (0..ndim)
            .map(|d| round_half_up(center.pos[d]).clamp(0, shape[d] as i64 - 1) as usize)
            .collect();
        let cf = flat(&cidx);
        let mut seed: Option<usize> = None;
        if labels[cf] == ci as u32 {
            seed = Some(cf);
        } else {
            // Search the Â±g/2 box (raster order) for the nearest voxel of label ci.
            off.iter_mut().enumerate().for_each(|(d, o)| *o = -half[d]);
            'search: loop {
                let mut in_bounds = true;
                for d in 0..ndim {
                    let v = cidx[d] as i64 + off[d];
                    if v < 0 || v >= shape[d] as i64 {
                        in_bounds = false;
                        break;
                    }
                    q[d] = v as usize;
                }
                if in_bounds && labels[flat(&q)] == ci as u32 {
                    seed = Some(flat(&q));
                    break 'search;
                }
                let mut d = ndim;
                let carry = loop {
                    if d == 0 {
                        break true;
                    }
                    d -= 1;
                    off[d] += 1;
                    if off[d] <= half[d] {
                        break false;
                    }
                    off[d] = -half[d];
                };
                if carry {
                    break;
                }
            }
        }
        if let Some(s) = seed {
            flood!(s, ci as u32, None::<u32>);
            if comp.len() < min_super {
                for &pf in &comp {
                    marker[pf] = 0;
                }
            }
        }
    }

    // â”€â”€ Phase 2: raster-scan unmarked fragments, relabel by size â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let mut next_label = k as u32;
    let mut prev_label = k as u32;
    for fi in 0..n {
        if marker[fi] == 0 {
            let req = labels[fi];
            flood!(fi, req, Some(next_label));
            if comp.len() >= min_super {
                prev_label = next_label;
                next_label += 1;
            } else {
                for &pf in &comp {
                    labels[pf] = prev_label;
                }
            }
        } else {
            prev_label = labels[fi];
        }
    }
}

#[cfg(test)]
#[path = "tests_slic_itk.rs"]
mod tests;
