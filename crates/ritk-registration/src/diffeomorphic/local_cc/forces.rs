//! CC force computation primitives for diffeomorphic registration.

use super::*;

// ── Parallel-slice helper ────────────────────────────────────────────────────

/// Send+Sync wrapper for a mutable slice pointer, enabling safe parallel
/// access to disjoint sub-slices via offset arithmetic.
///
/// The pointer and length are stored; callers must construct disjoint
/// `&mut [f32]` slices from non-overlapping offset ranges, which
/// z-slice parallelism guarantees by construction.
pub(crate) struct CellSlice {
    pub(crate) ptr: *mut f32,
    pub(crate) len: usize,
}

impl CellSlice {
    pub(crate) fn from_mut(s: &mut [f32]) -> Self {
        Self {
            ptr: s.as_mut_ptr(),
            len: s.len(),
        }
    }

    /// Reconstruct a mutable slice at `offset` with length `chunk_len`.
    ///
    /// # Safety
    /// Caller must ensure `[offset, offset + chunk_len)` is within `[0, len)`
    /// and no other reference to the same memory range exists.
    ///
    /// The `&self → &mut [f32]` pattern is sound here because the caller
    /// partitions `[0, len)` into disjoint offset ranges before passing them
    /// to parallel threads, so no two `&mut` references alias the same memory.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slice_mut(&self, offset: usize, chunk_len: usize) -> &mut [f32] {
        debug_assert!(offset + chunk_len <= self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(offset), chunk_len)
    }
}

// SAFETY: CellSlice is only used to create disjoint mutable slices within
// Rayon's parallel z-slice iteration, which guarantees non-overlapping
// regions per closure invocation. No two threads ever write to the same
// memory via this wrapper.
unsafe impl Send for CellSlice {}
unsafe impl Sync for CellSlice {}

// ── Force computation ────────────────────────────────────────────────────────

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
    let r = radius as isize;
    let forces: Vec<(f32, f32, f32)> = (0..n)
        .into_par_iter()
        .map(|fi| {
            let ix = fi % nx;
            let iy = (fi / nx) % ny;
            let iz = fi / (ny * nx);
            let (mu_i, mu_j, num, vi, vj, _) = window_cc_stats(i_w, j_w, dims, iz, iy, ix, r);
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
        })
        .collect();
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
    let [nz, _ny, nx] = dims;
    let r = radius as isize;
    let slice_len = dims[1] * nx;
    // Zero-initialize outputs.
    fz.iter_mut().for_each(|v| *v = 0.0);
    fy.iter_mut().for_each(|v| *v = 0.0);
    fx.iter_mut().for_each(|v| *v = 0.0);
    // Convert &mut [f32] to (ptr, len) pairs and wrap the pointers in
    // CellSlice — a Send+Sync wrapper. The Fn closure cannot capture
    // &mut references; CellSlice allows Rayon to share the pointer across
    // threads. Each thread reconstructs only its own disjoint slice via
    // offset arithmetic.
    let fz = CellSlice::from_mut(fz);
    let fy = CellSlice::from_mut(fy);
    let fx = CellSlice::from_mut(fx);
    // Process in parallel by z-slice. Each z-slice writes to a disjoint
    // contiguous range in the output buffers.
    (0..nz).into_par_iter().for_each(|iz| {
        let base = iz * slice_len;
        // SAFETY: fz, fy, fx have identical length and are split at the
        // same chunk boundaries. Each thread writes to a disjoint region.
        let fz_s = unsafe { fz.slice_mut(base, slice_len) };
        let fy_s = unsafe { fy.slice_mut(base, slice_len) };
        let fx_s = unsafe { fx.slice_mut(base, slice_len) };
        let ny = slice_len / nx;
        for iy in 0..ny {
            for ix in 0..nx {
                let local = iy * nx + ix;
                let fi = base + local;
                let (mu_i, mu_j, num, vi, vj, _) = window_cc_stats(i_w, j_w, dims, iz, iy, ix, r);
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
    });
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
