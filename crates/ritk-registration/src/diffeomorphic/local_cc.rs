//! Canonical local cross-correlation (CC) primitives for SyN registration.
//!
//! Single authoritative implementation of the Avants 2008 eq. 10 local CC
//! metric, shared by `syn_core`, `bspline_syn`, and `multires_syn`.
//!
//! # Algorithm
//! Local CC metric (Avants 2008, eq. 10): for each voxel p with window W of
//! radius `r`:
//!
//! cc(p) = Σ_{q∈W}(I(q)−μ_I)(J(q)−μ_J)
//!       / √(Σ_{q∈W}(I(q)−μ_I)² · Σ_{q∈W}(J(q)−μ_J)²)
//!
//! Force for velocity field v₁ (fixed→midpoint):
//!
//! f_k(p) = [(J_w(p)−μ_J)/√(σ_I²σ_J²) − CC·(I_w(p)−μ_I)/σ_I²] · ∇_k I_w(p)
//!
//! All outer voxel loops are parallelized via Rayon, since each voxel's
//! window reads are independent (read-only access, no data races).

mod forces;
mod sat;
#[cfg(test)]
pub(crate) use forces::cc_forces;
pub(crate) use forces::cc_forces_into;
#[cfg(test)]
pub(crate) use forces::field_rms;
pub(crate) use sat::CcSats;

// ── Window statistics ─────────────────────────────────────────────────────────

/// Reference implementation of local CC window statistics, used for
/// differential testing of [`CcSats`].
///
/// Returns `(mu_i, mu_j, cc_numerator, var_i, var_j, count)`.
///
/// Two-pass computation: first pass computes window means, second pass
/// computes covariance and variances using the means. Production callers
/// use [`CcSats`] instead.
///
/// SAT-based O(N) implementation: 5 summed-area tables are built once per
/// `(i_w, j_w)` pair via `CcSats::build`, then each voxel's statistics are
/// retrieved in O(1) via `CcSats::query_at`.
///
/// # Numerical contract
/// Inputs must be [0, 1]-normalized. For f64 SATs, the König–Huygens
/// cancellation error is bounded by 2·cnt·ε_f64 ≈ 1.5×10⁻¹³, well below
/// the 1×10⁻¹⁰ force guard. See `sat::CcSats` for details.
#[cfg(test)]
#[inline]
pub(crate) fn window_cc_stats(
    i_w: &[f32],
    j_w: &[f32],
    dims: [usize; 3],
    iz: usize,
    iy: usize,
    ix: usize,
    r: isize,
) -> (f64, f64, f64, f64, f64, u32) {
    let [nz, ny, nx] = dims;
    let (mut si, mut sj, mut cnt) = (0.0_f64, 0.0_f64, 0u32);
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let qi = crate::deformable_field_ops::flat(
                    (iz as isize + dz).max(0).min(nz as isize - 1) as usize,
                    (iy as isize + dy).max(0).min(ny as isize - 1) as usize,
                    (ix as isize + dx).max(0).min(nx as isize - 1) as usize,
                    ny,
                    nx,
                );
                si += i_w[qi] as f64;
                sj += j_w[qi] as f64;
                cnt += 1;
            }
        }
    }
    let (mu_i, mu_j) = (si / cnt as f64, sj / cnt as f64);
    let (mut num, mut vi, mut vj) = (0.0_f64, 0.0_f64, 0.0_f64);
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let qi = crate::deformable_field_ops::flat(
                    (iz as isize + dz).max(0).min(nz as isize - 1) as usize,
                    (iy as isize + dy).max(0).min(ny as isize - 1) as usize,
                    (ix as isize + dx).max(0).min(nx as isize - 1) as usize,
                    ny,
                    nx,
                );
                let di = i_w[qi] as f64 - mu_i;
                let dj = j_w[qi] as f64 - mu_j;
                num += di * dj;
                vi += di * di;
                vj += dj * dj;
            }
        }
    }
    (mu_i, mu_j, num, vi, vj, cnt)
}

// ── Convergence metric ───────────────────────────────────────────────────────

/// Compute mean local CC over all voxels for convergence monitoring.
///
/// Parallelized over voxels via moirai; each voxel's reads are independent.
pub(crate) fn mean_local_cc(i_w: &[f32], j_w: &[f32], dims: [usize; 3], radius: usize) -> f64 {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let sats = CcSats::build(i_w, j_w, dims, radius);
    let (total_cc, count) = moirai::reduce_index_with::<moirai::Adaptive, _, _, _>(
        n,
        (0.0_f64, 0usize),
        |fi| {
            let ix = fi % nx;
            let iy = (fi / nx) % ny;
            let iz = fi / (ny * nx);
            let (_, _, num, di2, dj2, _) = sats.query_at(iz, iy, ix);
            let d = (di2 * dj2).sqrt();
            if d > 1e-10 {
                (num / d, 1usize)
            } else {
                (0.0_f64, 0usize)
            }
        },
        |(a, b), (c, d)| (a + c, b + d),
    );
    if count == 0 {
        0.0
    } else {
        total_cc / count as f64
    }
}

#[cfg(test)]
mod tests;
