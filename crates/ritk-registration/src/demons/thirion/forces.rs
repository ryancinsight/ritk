//! Optical-flow force computation and field clamping utilities.

use crate::deformable_field_ops::{VectorField, VectorFieldMut};

/// Compute optical-flow Thirion forces into caller-provided buffers.
///
/// Parallelized over z-slices; each slice writes to a disjoint
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
    let [_nz, ny, nx] = dims;
    let slice_len = ny * nx;
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

    let mut zipped: Vec<(&mut [f32], &mut [f32], &mut [f32])> = fz
        .chunks_exact_mut(slice_len)
        .zip(fy.chunks_exact_mut(slice_len))
        .zip(fx.chunks_exact_mut(slice_len))
        .map(|((z, y), x)| (z, y, x))
        .collect();

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut zipped,
        1,
        |iz, chunk| {
            let (fz_s, fy_s, fx_s) = &mut chunk[0];
            let base = iz * slice_len;
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
        },
    );

    clamp_field_magnitude(fz, fy, fx, max_step_length);
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
