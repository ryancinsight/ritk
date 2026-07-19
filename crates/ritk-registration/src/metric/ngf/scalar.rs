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

    // Pass 2: weighted squared normalized augmented-gradient dot product over
    // the mask. The edge terms form an additional gradient component, so the
    // numerator contains η_F η_M as well as the spatial-gradient dot product.
    // This makes the normalized inner product exactly one for identical fields.
    let eta_product = (eta_f2 * eta_m2).sqrt();
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
        let augmented_dot = dot + eta_product;
        acc += w * f64::from((augmented_dot * augmented_dot) / (na2 * nb2));
    }
    (acc / ws) as f32
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
