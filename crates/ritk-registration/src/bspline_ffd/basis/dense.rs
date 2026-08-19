//! Bounded dense support tables for B-spline displacement evaluation.

use super::super::super::volume_dims::VolumeDims;
use super::super::scalar::cubic_bspline_basis;
use crate::deformable_field_ops::flat;

/// Evaluate a dense displacement field through a newly built support table.
///
/// The support table clamps every control-point index in advance and stores a
/// zero mask for out-of-bounds support cells. The inner 4³ accumulation can
/// therefore access only valid control-point entries without a branch per
/// cell. Call [`evaluate_bspline_displacement_dense_with`] when the lattice is
/// reused across registration iterations.
///
/// # Panics
/// Panics if an output buffer is shorter than `dims.product()`.
pub fn evaluate_bspline_displacement_dense_into(
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    ctrl_spacing: &[f64; 3],
    dims: VolumeDims,
    dz: &mut [f32],
    dy: &mut [f32],
    dx: &mut [f32],
) {
    let support = DenseSupport::build(dims, *ctrl_dims, ctrl_spacing);
    support.evaluate(cp_z, cp_y, cp_x, ctrl_dims, dz, dy, dx);
}

/// Evaluate a dense displacement field through a pre-built support table.
///
/// Building the table at the registration level keeps the hot iteration path
/// allocation-free while retaining the branch-free 4³ support accumulation.
///
/// # Panics
/// Panics if an output buffer is shorter than the table's voxel count.
#[inline]
pub fn evaluate_bspline_displacement_dense_with(
    support: &DenseSupport,
    cp_z: &[f32],
    cp_y: &[f32],
    cp_x: &[f32],
    ctrl_dims: &[usize; 3],
    dz: &mut [f32],
    dy: &mut [f32],
    dx: &mut [f32],
) {
    support.evaluate(cp_z, cp_y, cp_x, ctrl_dims, dz, dy, dx);
}

/// Dense support tables reused across B-spline displacement evaluations.
#[derive(Clone, Debug)]
pub struct DenseSupport {
    z_idx: Vec<[u32; 4]>,
    z_w: Vec<[f64; 4]>,
    z_mask: Vec<[f64; 4]>,
    y_idx: Vec<[u32; 4]>,
    y_w: Vec<[f64; 4]>,
    y_mask: Vec<[f64; 4]>,
    x_idx: Vec<[u32; 4]>,
    x_w: Vec<[f64; 4]>,
    x_mask: Vec<[f64; 4]>,
}

impl DenseSupport {
    /// Build support tables for an image lattice and control-point lattice.
    ///
    /// The dense dispatch contract bounds the control-lattice product by
    /// [`super::grid::DENSE_LATTICE_CUTOFF`], so the stored linear indices fit in
    /// `u32`.
    pub fn build(dims: VolumeDims, ctrl_dims: [usize; 3], ctrl_spacing: &[f64; 3]) -> Self {
        let [nz, ny, nx] = dims.as_array();
        let [cnz, cny, cnx] = ctrl_dims;

        Self {
            z_idx: build_axis_idx_table(nz, ctrl_spacing[0], cnz),
            z_w: build_axis_w_table(nz, ctrl_spacing[0]),
            z_mask: build_axis_mask_table(nz, ctrl_spacing[0], cnz),
            y_idx: build_axis_idx_table(ny, ctrl_spacing[1], cny),
            y_w: build_axis_w_table(ny, ctrl_spacing[1]),
            y_mask: build_axis_mask_table(ny, ctrl_spacing[1], cny),
            x_idx: build_axis_idx_table(nx, ctrl_spacing[2], cnx),
            x_w: build_axis_w_table(nx, ctrl_spacing[2]),
            x_mask: build_axis_mask_table(nx, ctrl_spacing[2], cnx),
        }
    }

    /// Accumulate all voxels through the cached branch-free support tables.
    ///
    /// Output buffers are caller-owned and are overwritten from index zero.
    ///
    /// # Panics
    /// Panics if an output buffer is shorter than the table's voxel count.
    #[inline]
    pub fn evaluate(
        &self,
        cp_z: &[f32],
        cp_y: &[f32],
        cp_x: &[f32],
        ctrl_dims: &[usize; 3],
        dz: &mut [f32],
        dy: &mut [f32],
        dx: &mut [f32],
    ) {
        let [cnz, cny, cnx] = *ctrl_dims;
        let ctrl_n = cnz * cny * cnx;
        let cny_cnx = cny * cnx;
        let nz = self.z_idx.len();
        let ny = self.y_idx.len();
        let nx = self.x_idx.len();
        let n = nz * ny * nx;

        dz[..n].fill(0.0);
        dy[..n].fill(0.0);
        dx[..n].fill(0.0);

        for iz in 0..nz {
            let z_idx_row = &self.z_idx[iz];
            let z_w_row = &self.z_w[iz];
            let z_mask_row = &self.z_mask[iz];
            for iy in 0..ny {
                let y_idx_row = &self.y_idx[iy];
                let y_w_row = &self.y_w[iy];
                let y_mask_row = &self.y_mask[iy];
                for ix in 0..nx {
                    let x_idx_row = &self.x_idx[ix];
                    let x_w_row = &self.x_w[ix];
                    let x_mask_row = &self.x_mask[ix];
                    let fi = flat(iz, iy, ix, ny, nx);

                    let mut sum_z = 0.0_f64;
                    let mut sum_y = 0.0_f64;
                    let mut sum_x = 0.0_f64;
                    for az in 0..4usize {
                        let row_base = (z_idx_row[az] as usize) * cny_cnx;
                        let wz = z_w_row[az];
                        let mz = z_mask_row[az];
                        for ay in 0..4usize {
                            let slice_base = row_base + (y_idx_row[ay] as usize) * cnx;
                            let wzy = wz * y_w_row[ay];
                            let mzy = mz * y_mask_row[ay];
                            for ax in 0..4usize {
                                let ci = slice_base + x_idx_row[ax] as usize;
                                let w = wzy * x_w_row[ax];
                                debug_assert!(ci < ctrl_n, "dense support OOB");
                                let weight = mzy * x_mask_row[ax] * w;
                                sum_z += weight * cp_z[ci] as f64;
                                sum_y += weight * cp_y[ci] as f64;
                                sum_x += weight * cp_x[ci] as f64;
                            }
                        }
                    }
                    dz[fi] = sum_z as f32;
                    dy[fi] = sum_y as f32;
                    dx[fi] = sum_x as f32;
                }
            }
        }
    }
}

#[inline]
fn build_axis_idx_table(dim: usize, ctrl_spacing: f64, ctrl_axis: usize) -> Vec<[u32; 4]> {
    let mut table = vec![[0_u32; 4]; dim];
    for (i, row) in table.iter_mut().enumerate() {
        let u = i as f64 / ctrl_spacing + 1.0;
        let ki = u.floor() as isize - 1;
        for (k, slot) in row.iter_mut().enumerate() {
            let cidx = ki + k as isize;
            *slot = if cidx < 0 || cidx >= ctrl_axis as isize {
                0
            } else {
                cidx as u32
            };
        }
    }
    table
}

#[inline]
fn build_axis_w_table(dim: usize, ctrl_spacing: f64) -> Vec<[f64; 4]> {
    let mut table = vec![[0_f64; 4]; dim];
    for (i, row) in table.iter_mut().enumerate() {
        let u = i as f64 / ctrl_spacing + 1.0;
        let ki = u.floor() as isize - 1;
        let t = u - (ki + 1) as f64;
        *row = cubic_bspline_basis(t);
    }
    table
}

#[inline]
fn build_axis_mask_table(dim: usize, ctrl_spacing: f64, ctrl_axis: usize) -> Vec<[f64; 4]> {
    let mut table = vec![[0_f64; 4]; dim];
    for (i, row) in table.iter_mut().enumerate() {
        let u = i as f64 / ctrl_spacing + 1.0;
        let ki = u.floor() as isize - 1;
        for (k, slot) in row.iter_mut().enumerate() {
            let cidx = ki + k as isize;
            *slot = if cidx < 0 || cidx >= ctrl_axis as isize {
                0.0
            } else {
                1.0
            };
        }
    }
    table
}
