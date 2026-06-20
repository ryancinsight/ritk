/// Row-major (C-order) strides for `shape` — last axis fastest.
pub(crate) fn row_major_strides<const D: usize>(shape: &[usize; D]) -> [usize; D] {
    let mut stride = [1usize; D];
    for a in (0..D.saturating_sub(1)).rev() {
        stride[a] = stride[a + 1] * shape[a + 1];
    }
    stride
}

/// Per-voxel central-difference gradient field of `data` (one `[f32; D]` per
/// voxel). Computed once per image and, for the fixed image, reused across every
/// transform evaluation.
pub(crate) fn compute_gradient_field<const D: usize>(
    data: &[f32],
    shape: &[usize; D],
    stride: &[usize; D],
) -> Vec<[f32; D]> {
    let mut g = vec![[0.0_f32; D]; data.len()];
    for (flat, gv) in g.iter_mut().enumerate() {
        grad_at(data, flat, shape, stride, gv);
    }
    g
}

/// Per-voxel weight: 0 outside the mask, else `weights[flat]` (or 1 when no
/// weight field is supplied — recovering the uniform Haber–Modersitzki mean).
pub(crate) fn voxel_weight(flat: usize, mask: Option<&[bool]>, weights: Option<&[f32]>) -> f64 {
    if mask.is_some_and(|mk| !mk[flat]) {
        return 0.0;
    }
    weights.map_or(1.0, |w| f64::from(w[flat]).max(0.0))
}

/// `η² = (weighted-mean masked gradient magnitude)²` — the edge-noise scale
/// (Haber & Modersitzki), clamped away from zero.
pub(crate) fn weighted_eta2<const D: usize>(
    grads: &[[f32; D]],
    mask: Option<&[bool]>,
    weights: Option<&[f32]>,
) -> f32 {
    let (mut s, mut ws) = (0.0_f64, 0.0_f64);
    for (flat, g) in grads.iter().enumerate() {
        let w = voxel_weight(flat, mask, weights);
        if w == 0.0 {
            continue;
        }
        s += w * f64::from(magnitude(g));
        ws += w;
    }
    if ws <= 0.0 {
        1e-12
    } else {
        ((s / ws).powi(2)).max(1e-12) as f32
    }
}

/// `η²` over an explicit gradient/weight subset (the stochastic-sampling path).
pub(crate) fn weighted_eta2_slice<const D: usize>(grads: &[[f32; D]], weight: &[f64]) -> f32 {
    let (mut s, mut ws) = (0.0_f64, 0.0_f64);
    for (g, &w) in grads.iter().zip(weight) {
        s += w * f64::from(magnitude(g));
        ws += w;
    }
    if ws <= 0.0 {
        1e-12
    } else {
        ((s / ws).powi(2)).max(1e-12) as f32
    }
}

/// Weighted mean squared normalized gradient dot product, given the PRECOMPUTED
/// fixed gradient field `gf`/`eta_f2` and the moving host volume `m`. The moving
/// gradient and `η_M` are computed here (they vary with the transform); the fixed
/// side is supplied so a registration reuses it across every evaluation.
pub(crate) fn weighted_ngf_from_fixed<const D: usize>(
    gf: &[[f32; D]],
    eta_f2: f32,
    m: &[f32],
    shape: &[usize; D],
    stride: &[usize; D],
    mask: Option<&[bool]>,
    weights: Option<&[f32]>,
) -> f32 {
    let n = gf.len();
    if m.len() != n {
        return 0.0;
    }
    let mut gm = [0.0_f32; D];

    // Pass 1: η_M² from the (weighted) moving gradient magnitude.
    let (mut s, mut ws) = (0.0_f64, 0.0_f64);
    for flat in 0..n {
        let w = voxel_weight(flat, mask, weights);
        if w == 0.0 {
            continue;
        }
        grad_at(m, flat, shape, stride, &mut gm);
        s += w * f64::from(magnitude(&gm));
        ws += w;
    }
    if ws <= 0.0 {
        return 0.0;
    }
    let eta_m2 = ((s / ws).powi(2)).max(1e-12) as f32;

    // Pass 2: weighted squared normalized gradient dot product over the mask.
    let mut acc = 0.0_f64;
    for (flat, g) in gf.iter().enumerate() {
        let w = voxel_weight(flat, mask, weights);
        if w == 0.0 {
            continue;
        }
        grad_at(m, flat, shape, stride, &mut gm);
        let dot: f32 = (0..D).map(|a| g[a] * gm[a]).sum();
        let na2: f32 = (0..D).map(|a| g[a] * g[a]).sum::<f32>() + eta_f2;
        let nb2: f32 = (0..D).map(|a| gm[a] * gm[a]).sum::<f32>() + eta_m2;
        acc += w * f64::from((dot * dot) / (na2 * nb2));
    }
    (acc / ws) as f32
}

/// Build a brain-centroid Gaussian weight field over `mask` for the weighted NGF
/// metric. `w(x) = exp(−‖x − c‖² / (2σ²))` for masked voxels (0 elsewhere), with
/// `c` the mask centroid and `σ = sigma_frac · r_rms`, where `r_rms` is the
/// mask's root-mean-square radius — all distances in PHYSICAL units via `spacing`
/// so anisotropic voxels (e.g. 0.4×0.4×3 mm) are weighted correctly.
///
/// `sigma_frac` controls how sharply the periphery is suppressed: smaller →
/// tighter central focus. With the RMS radius as the scale, `sigma_frac ≈ 0.7`
/// places σ near ⅓ of the outer brain radius (the value the multimodal-edge
/// literature uses to emphasise deep structure over the skull rim).
#[must_use]
pub fn center_gaussian_weight_field<const D: usize>(
    shape: &[usize; D],
    mask: Option<&[bool]>,
    spacing: &[f64; D],
    sigma_frac: f64,
) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut stride = [1usize; D];
    for a in (0..D.saturating_sub(1)).rev() {
        stride[a] = stride[a + 1] * shape[a + 1];
    }
    let included = |flat: usize| mask.is_none_or(|mk| mk[flat]);
    let phys = |flat: usize, a: usize| ((flat / stride[a]) % shape[a]) as f64 * spacing[a];

    // Centroid (physical) and RMS radius over the mask.
    let mut c = [0.0_f64; D];
    let mut cnt = 0.0_f64;
    for flat in 0..n {
        if !included(flat) {
            continue;
        }
        for (a, ca) in c.iter_mut().enumerate() {
            *ca += phys(flat, a);
        }
        cnt += 1.0;
    }
    if cnt < 1.0 {
        return vec![1.0; n];
    }
    for ca in &mut c {
        *ca /= cnt;
    }
    let mut r2sum = 0.0_f64;
    for flat in 0..n {
        if !included(flat) {
            continue;
        }
        r2sum += (0..D).map(|a| (phys(flat, a) - c[a]).powi(2)).sum::<f64>();
    }
    let r_rms = (r2sum / cnt).sqrt().max(f64::EPSILON);
    let two_sigma2 = 2.0 * (sigma_frac * r_rms).powi(2);

    let mut w = vec![0.0_f32; n];
    for (flat, wf) in w.iter_mut().enumerate() {
        if !included(flat) {
            continue;
        }
        let r2 = (0..D).map(|a| (phys(flat, a) - c[a]).powi(2)).sum::<f64>();
        *wf = (-r2 / two_sigma2).exp() as f32;
    }
    w
}

/// Central-difference spatial gradient of `data` at flat index `flat`, written
/// into `out` (one component per axis). Borders use a one-sided difference.
pub(crate) fn grad_at<const D: usize>(
    data: &[f32],
    flat: usize,
    shape: &[usize; D],
    stride: &[usize; D],
    out: &mut [f32; D],
) {
    for a in 0..D {
        let idx_a = (flat / stride[a]) % shape[a];
        let has_lo = idx_a > 0;
        let has_hi = idx_a + 1 < shape[a];
        let lo = if has_lo {
            data[flat - stride[a]]
        } else {
            data[flat]
        };
        let hi = if has_hi {
            data[flat + stride[a]]
        } else {
            data[flat]
        };
        let denom = (usize::from(has_lo) + usize::from(has_hi)).max(1) as f32;
        out[a] = (hi - lo) / denom;
    }
}

pub(crate) fn magnitude<const D: usize>(g: &[f32; D]) -> f32 {
    (0..D).map(|a| g[a] * g[a]).sum::<f32>().sqrt()
}

/// `NGF ∈ [0, 1]` of two co-gridded volumes `f`, `m` of (row-major) `shape`,
/// averaged over the `true` voxels of `mask` (or all voxels if `mask` is `None`).
///
/// Gradients always use real neighbours (the image is *not* zeroed outside the
/// mask), so the mask only restricts which voxels are *counted* — no artificial
/// mask-boundary edge is introduced. Masking to the brain+skull region is what
/// makes cross-modal NGF lock onto the shared rigid anatomy instead of the
/// scalp/scanner-bed/FOV edges. See the [module docs](self).
#[cfg(test)]
pub(crate) fn ngf_scalar<const D: usize>(
    f: &[f32],
    m: &[f32],
    shape: &[usize; D],
    mask: Option<&[bool]>,
    weights: Option<&[f32]>,
) -> f32 {
    let n = f.len();
    if n == 0 || m.len() != n {
        return 0.0;
    }
    let stride = row_major_strides(shape);
    let gf = compute_gradient_field(f, shape, &stride);
    let eta_f2 = weighted_eta2(&gf, mask, weights);
    weighted_ngf_from_fixed(&gf, eta_f2, m, shape, &stride, mask, weights)
}
