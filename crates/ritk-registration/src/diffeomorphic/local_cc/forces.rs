// ── Force computation ────────────────────────────────────────────────────────

use crate::deformable_field_ops::{VectorField, VectorFieldMut};

/// Compute local CC gradient forces (Avants 2008, eq. 10).
///
/// For each voxel p the local window W = {q : |q−p|_∞ ≤ r} yields:
/// fz[p] = force_scale · gIz[p]
/// where force_scale = (J_w(p)−μ_J)/denom − CC·(I_w(p)−μ_I)/(σ_I²+ε)
/// and denom = √(σ_I²·σ_J²) + ε
///
/// Parallelized over voxels via Rayon; each voxel's window reads are
/// independent, producing no data race.
#[cfg(test)]
pub(crate) fn cc_forces(
    i_w: &[f32],
    j_w: &[f32],
    gi_z: &[f32],
    gi_y: &[f32],
    gi_x: &[f32],
    dims: [usize; 3],
    radius: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let sats = super::CcSats::build(i_w, j_w, dims, radius);
    let forces: Vec<(f32, f32, f32)> =
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |fi| {
            let ix = fi % nx;
            let iy = (fi / nx) % ny;
            let iz = fi / (ny * nx);
            let (mu_i, mu_j, num, vi, vj, _) = sats.query_at(iz, iy, ix);
            if vi < 1e-10 {
                return (0.0_f32, 0.0_f32, 0.0_f32);
            }
            let iw_c = i_w[fi] as f64 - mu_i;
            let jw_c = j_w[fi] as f64 - mu_j;
            // Avants 2008, eq. 10 — gradient ascent on local CC.
            // ∂CC/∂v₁_k(x) = [(J_w−μ_J)/√(σ_I²·σ_J²) − CC·(I_w−μ_I)/σ_I²] · ∇_k I_w
            let denom = (vi * vj).sqrt() + 1e-10;
            let cc = num / denom;
            let force_scale = jw_c / denom - cc * iw_c / (vi + 1e-10);
            (
                (force_scale * gi_z[fi] as f64) as f32,
                (force_scale * gi_y[fi] as f64) as f32,
                (force_scale * gi_x[fi] as f64) as f32,
            )
        });
    let mut fz = Vec::with_capacity(n);
    let mut fy = Vec::with_capacity(n);
    let mut fx = Vec::with_capacity(n);
    for (z, y, x) in forces {
        fz.push(z);
        fy.push(y);
        fx.push(x);
    }
    (fz, fy, fx)
}

/// Compute local CC gradient forces into caller-provided buffers (Avants 2008, eq. 10).
///
/// Equivalent to the allocating CC-force helper used by tests, but writes
/// directly into `fz`, `fy`, `fx` without intermediate allocation. All three buffers must have length
/// `dims[0] * dims[1] * dims[2]`.
///
/// Parallelized over z-slices via Rayon; each slice writes to a disjoint
/// contiguous range in the output buffers, producing no data race.
#[cfg(test)]
pub(crate) fn cc_forces_into(
    i_w: &[f32],
    j_w: &[f32],
    gi_z: &[f32],
    gi_y: &[f32],
    gi_x: &[f32],
    dims: [usize; 3],
    radius: usize,
    fz: &mut [f32],
    fy: &mut [f32],
    fx: &mut [f32],
) {
    let sats = super::CcSats::build(i_w, j_w, dims, radius);
    cc_forces_from_sats_into::<false>(i_w, j_w, gi_z, gi_y, gi_x, dims, &sats, fz, fy, fx);
}

/// Compute local CC forces from a caller-owned summed-area table set.
///
/// `REVERSED` exchanges the fixed and moving statistics while retaining the
/// single canonical table representation for a warped image pair.
#[expect(
    clippy::too_many_arguments,
    reason = "the three component slices preserve structure-of-arrays field storage"
)]
#[cfg(test)]
pub(crate) fn cc_forces_from_sats_into<const REVERSED: bool>(
    i_w: &[f32],
    j_w: &[f32],
    gi_z: &[f32],
    gi_y: &[f32],
    gi_x: &[f32],
    dims: [usize; 3],
    sats: &super::CcSats,
    fz: &mut [f32],
    fy: &mut [f32],
    fx: &mut [f32],
) {
    let [_nz, _ny, nx] = dims;
    let slice_len = dims[1] * nx;
    // Zero-initialize outputs.
    fz.iter_mut().for_each(|v| *v = 0.0);
    fy.iter_mut().for_each(|v| *v = 0.0);
    fx.iter_mut().for_each(|v| *v = 0.0);

    moirai::for_each_chunk_triple_mut_enumerated_with::<moirai::Adaptive, _, _, _, _>(
        fz,
        fy,
        fx,
        slice_len,
        |iz, fz_s, fy_s, fx_s| {
            let base = iz * slice_len;
            let ny = slice_len / nx;
            for iy in 0..ny {
                for ix in 0..nx {
                    let local = iy * nx + ix;
                    let fi = base + local;
                    let (mu_f, mu_m, num, var_f, var_m, _) = sats.query_at(iz, iy, ix);
                    let (mu_i, mu_j, vi, vj) = if REVERSED {
                        (mu_m, mu_f, var_m, var_f)
                    } else {
                        (mu_f, mu_m, var_f, var_m)
                    };
                    if vi < 1e-10 {
                        continue; // already zeroed
                    }
                    let iw_c = i_w[fi] as f64 - mu_i;
                    let jw_c = j_w[fi] as f64 - mu_j;
                    let denom = (vi * vj).sqrt() + 1e-10;
                    let cc = num / denom;
                    let force_scale = jw_c / denom - cc * iw_c / (vi + 1e-10);
                    fz_s[local] = (force_scale * gi_z[fi] as f64) as f32;
                    fy_s[local] = (force_scale * gi_y[fi] as f64) as f32;
                    fx_s[local] = (force_scale * gi_x[fi] as f64) as f32;
                }
            }
        },
    );
}

/// Compute both symmetric CC force fields and mean local CC in one traversal.
#[expect(
    clippy::too_many_arguments,
    reason = "two images, gradients, outputs, and slice reductions are distinct kernel roles"
)]
pub(crate) fn bidirectional_cc_from_sats_into(
    i_w: &[f32],
    j_w: &[f32],
    gradient_i: VectorField<'_>,
    gradient_j: VectorField<'_>,
    dims: [usize; 3],
    sats: &super::CcSats,
    force_i: VectorFieldMut<'_>,
    force_j: VectorFieldMut<'_>,
    slice_cc: &mut [(f64, usize)],
) -> f64 {
    let [nz, ny, nx] = dims;
    let slice_len = ny * nx;
    debug_assert_eq!(slice_cc.len(), nz);
    let mut chunks: Vec<_> = force_i
        .z
        .chunks_exact_mut(slice_len)
        .zip(force_i.y.chunks_exact_mut(slice_len))
        .zip(force_i.x.chunks_exact_mut(slice_len))
        .zip(force_j.z.chunks_exact_mut(slice_len))
        .zip(force_j.y.chunks_exact_mut(slice_len))
        .zip(force_j.x.chunks_exact_mut(slice_len))
        .zip(slice_cc.iter_mut())
        .map(|((((((iz, iy), ix), jz), jy), jx), cc)| (iz, iy, ix, jz, jy, jx, cc))
        .collect();

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Parallel, _, _>(
        &mut chunks,
        1,
        |z, chunk| {
            let (uiz, uiy, uix, ujz, ujy, ujx, cc_out) = &mut chunk[0];
            let base = z * slice_len;
            let mut cc_sum = 0.0_f64;
            let mut cc_count = 0usize;
            for y in 0..ny {
                for x in 0..nx {
                    let local = y * nx + x;
                    let index = base + local;
                    let (mu_i, mu_j, numerator, var_i, var_j, _) = sats.query_at(z, y, x);
                    let centered_i = i_w[index] as f64 - mu_i;
                    let centered_j = j_w[index] as f64 - mu_j;
                    let raw_denom = (var_i * var_j).sqrt();
                    if raw_denom > 1e-10 {
                        cc_sum += numerator / raw_denom;
                        cc_count += 1;
                    }
                    let denom = raw_denom + 1e-10;
                    let cc = numerator / denom;
                    let scale_i = if var_i < 1e-10 {
                        0.0
                    } else {
                        centered_j / denom - cc * centered_i / (var_i + 1e-10)
                    };
                    let scale_j = if var_j < 1e-10 {
                        0.0
                    } else {
                        centered_i / denom - cc * centered_j / (var_j + 1e-10)
                    };
                    uiz[local] = (scale_i * gradient_i.z[index] as f64) as f32;
                    uiy[local] = (scale_i * gradient_i.y[index] as f64) as f32;
                    uix[local] = (scale_i * gradient_i.x[index] as f64) as f32;
                    ujz[local] = (scale_j * gradient_j.z[index] as f64) as f32;
                    ujy[local] = (scale_j * gradient_j.y[index] as f64) as f32;
                    ujx[local] = (scale_j * gradient_j.x[index] as f64) as f32;
                }
            }
            **cc_out = (cc_sum, cc_count);
        },
    );

    let (sum, count) = slice_cc
        .iter()
        .fold((0.0_f64, 0usize), |(sum, count), &(part, n)| {
            (sum + part, count + n)
        });
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

// ── Utility ──────────────────────────────────────────────────────────────────

/// RMS magnitude of a displacement field component.
///
/// Used by registration engine test modules for field-quality assertions.
#[cfg(test)]
pub(crate) fn field_rms(v: &[f32]) -> f64 {
    let ss: f64 = v.iter().map(|&x| (x as f64).powi(2)).sum();
    (ss / v.len() as f64).sqrt()
}
