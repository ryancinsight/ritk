//! Fixed-point iterative inversion of a general displacement field.
//!
//! # Mathematical Specification
//!
//! For a general displacement field `u`, the inverse `u^{-1}` satisfies:
//!
//!   `Ï†(x + u^{-1}(x)) = x  âŸ¹  u^{-1}(x) = âˆ’u(x + u^{-1}(x))`
//!
//! **Fixed-point iteration** (Christensen & Johnson 2001):
//!
//!   `u^{-1}_0(x)      = âˆ’u(x)`                         (initialisation)
//!   `u^{-1}_{k+1}(x)  = âˆ’u(x + u^{-1}_k(x))`           (update rule)
//!
//! **Convergence guarantee:** When the Lipschitz constant `L = maxâ€–âˆ‡uâ€– < 1`,
//! the update map is a contraction and the iterate error satisfies:
//!
//!   `â€–u^{-1}_{k+1} âˆ’ u^{-1}_*â€–_âˆž  â‰¤  L^k Â· â€–u^{-1}_1 âˆ’ u^{-1}_0â€–_âˆž`
//!
//! # References
//! - Christensen, G. E. & Johnson, H. J. (2001). Consistent image registration.
//!   *IEEE Trans. Med. Imaging* 20(7):568â€“582.

use crate::deformable_field_ops::{
    trilinear_interpolate_field, VectorField, VectorFieldMut, VelocityField };

/// Default convergence tolerance for iterative inverse displacement field computation.
/// Maximum per-voxel Euclidean-norm change threshold between successive iterates.
pub const DEFAULT_INVERSE_FIELD_TOLERANCE: f64 = 1e-4;

/// Configuration for iterative inverse computation (used for non-SVF fields).
///
/// # Defaults
/// - `max_iterations`: 20
/// - `tolerance`: 1e-4 (max-norm convergence threshold, in voxels)
#[derive(Debug, Clone)]
pub struct InverseFieldConfig {
    /// Maximum number of fixed-point iterations.
    pub max_iterations: usize,
    /// Convergence threshold (voxels).
    ///
    /// Terminates early when the maximum per-voxel Euclidean norm of the change
    /// between successive iterates drops below this value:
    ///
    ///   `max_i â€–u^{-1}_{k+1}(i) âˆ’ u^{-1}_k(i)â€–_2 < tolerance`
    pub tolerance: f64 }

impl Default for InverseFieldConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            tolerance: DEFAULT_INVERSE_FIELD_TOLERANCE }
    }
}

/// Compute an approximate inverse of a general displacement field using
/// fixed-point iteration (Christensen & Johnson 2001).
///
/// Returns `(VelocityField, num_iterations_performed)`.
pub fn invert_displacement_field(
    disp_z: &[f32],
    disp_y: &[f32],
    disp_x: &[f32],
    dims: [usize; 3],
    config: &InverseFieldConfig,
) -> (VelocityField, usize) {
    let n = dims[0] * dims[1] * dims[2];

    let mut inv_z: Vec<f32> = disp_z.iter().map(|&v| -v).collect();
    let mut inv_y: Vec<f32> = disp_y.iter().map(|&v| -v).collect();
    let mut inv_x: Vec<f32> = disp_x.iter().map(|&v| -v).collect();

    let mut next_z = vec![0.0_f32; n];
    let mut next_y = vec![0.0_f32; n];
    let mut next_x = vec![0.0_f32; n];

    let mut iters = 0usize;

    for _ in 0..config.max_iterations {
        iters += 1;

        warp_displacement_into(
            VectorField {
                z: disp_z,
                y: disp_y,
                x: disp_x },
            VectorField {
                z: &inv_z,
                y: &inv_y,
                x: &inv_x },
            dims,
            VectorFieldMut {
                z: &mut next_z,
                y: &mut next_y,
                x: &mut next_x },
        );

        let max_change = (0..n)
            .map(|i| {
                let dz = (next_z[i] - inv_z[i]) as f64;
                let dy = (next_y[i] - inv_y[i]) as f64;
                let dx = (next_x[i] - inv_x[i]) as f64;
                (dz * dz + dy * dy + dx * dx).sqrt()
            })
            .fold(0.0_f64, f64::max);

        std::mem::swap(&mut inv_z, &mut next_z);
        std::mem::swap(&mut inv_y, &mut next_y);
        std::mem::swap(&mut inv_x, &mut next_x);

        if max_change < config.tolerance {
            break;
        }
    }

    (
        VelocityField {
            z: inv_z,
            y: inv_y,
            x: inv_x },
        iters,
    )
}

/// Warp a displacement field by a query displacement field via trilinear
/// interpolation, computing `-disp(x + query(x))` for the fixed-point iteration.
///
/// All three displacement components are sampled in one pass to avoid
/// repeating the same coordinate computation three times.
pub(super) fn warp_displacement_into(
    disp: VectorField<'_>,
    query: VectorField<'_>,
    dims: [usize; 3],
    out: VectorFieldMut<'_>,
) {
    let [nz, ny, nx] = dims;
    let VectorField {
        z: query_z,
        y: query_y,
        x: query_x } = query;
    let VectorFieldMut {
        z: out_z,
        y: out_y,
        x: out_x } = out;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let fi = iz * ny * nx + iy * nx + ix;
                let wz = iz as f32 + query_z[fi];
                let wy = iy as f32 + query_y[fi];
                let wx = ix as f32 + query_x[fi];

                let [sample_z, sample_y, sample_x] =
                    trilinear_interpolate_field(disp, dims.into(), wz, wy, wx);
                out_z[fi] = -sample_z;
                out_y[fi] = -sample_y;
                out_x[fi] = -sample_x;
            }
        }
    }
}
