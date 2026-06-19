//! Optical-flow force computation and field clamping utilities.

use crate::deformable_field_ops::{VectorField, VectorFieldMut};
use crate::parallel::CellSlice;

/// Compute optical-flow Thirion forces into caller-provided buffers.
///
/// Parallelized over z-slices via Rayon; each slice writes to a disjoint
/// contiguous region in `forces`, producing no data race. Voxel forces are
/// independent (no cross-voxel reads), so z-slice granularity matches the
/// `cc_forces_into` pattern used in the SyN diffeomorphic path.
///
/// # Arguments
/// * `dims` — `[nz, ny, nx]` spatial extent; required for z-slice dispatch.
pub(crate) fn thirion_forces_into(
    fixed: &[f32],
    m_warped: &[f32],
    grad: VectorField<'_>,
    max_step_length: f32,
    forces: VectorFieldMut<'_>,
    dims: [usize; 3],
) {
    let [nz, ny, nx] = dims;
    let slice_len = ny * nx;
    let n = nz * slice_len;
    let VectorField {
        z: grad_z,
        y: grad_y,
        x: grad_x,
    } = grad;
    let VectorFieldMut {
        z: fz,
        y: fy,
        x: fx,
    } = forces;
    let sigma_x2 = max_step_length * max_step_length;

    // Wrap output buffers in CellSlice so the Fn closure can capture them
    // across parallel z-slices. Each thread reconstructs only its own
    // disjoint slice region via offset arithmetic.
    let fz_cell = CellSlice::from_mut(fz);
    let fy_cell = CellSlice::from_mut(fy);
    let fx_cell = CellSlice::from_mut(fx);

    moirai::for_each_index_with::<moirai::Adaptive, _>(nz, |iz| {
        let base = iz * slice_len;
        // SAFETY: each thread writes to a disjoint z-slice [base, base+slice_len).
        // moirai::for_each_index_with guarantees iz is unique per closure invocation.
        let fz_s = unsafe { fz_cell.slice_mut(base, slice_len) };
        let fy_s = unsafe { fy_cell.slice_mut(base, slice_len) };
        let fx_s = unsafe { fx_cell.slice_mut(base, slice_len) };

        for local in 0..slice_len {
            let i = base + local;
            let diff = fixed[i] - m_warped[i];
            let gz = grad_z[i];
            let gy = grad_y[i];
            let gx = grad_x[i];
            let grad_sq = gz * gz + gy * gy + gx * gx;
            let denom = grad_sq + diff * diff / sigma_x2 + 1e-5;
            let scale = diff / denom;
            fz_s[local] = scale * gz;
            fy_s[local] = scale * gy;
            fx_s[local] = scale * gx;
        }
    });

    // After the parallel section (moirai::for_each_index_with is synchronous),
    // reconstruct the full mutable slices from the CellSlice pointers for the
    // sequential magnitude clamp pass.
    // SAFETY: all z-slice writes are complete; CellSlice is not used after this.
    let fz_full = unsafe { fz_cell.slice_mut(0, n) };
    let fy_full = unsafe { fy_cell.slice_mut(0, n) };
    let fx_full = unsafe { fx_cell.slice_mut(0, n) };
    clamp_field_magnitude(fz_full, fy_full, fx_full, max_step_length);
}

fn clamp_field_magnitude(fz: &mut [f32], fy: &mut [f32], fx: &mut [f32], max_length: f32) {
    let max2 = max_length * max_length;
    for i in 0..fz.len() {
        let mag2 = fz[i] * fz[i] + fy[i] * fy[i] + fx[i] * fx[i];
        if mag2 > max2 {
            let scale = max_length / mag2.sqrt();
            fz[i] *= scale;
            fy[i] *= scale;
            fx[i] *= scale;
        }
    }
}
