//! Pre-allocated scratch buffers for zero-allocation BSplineSyN iteration.
//!
//! All volume-sized buffers are sized at construction time and rebuilt or
//! overwritten in place. The fused CC dispatcher retains one bounded `O(nz)`
//! slice-descriptor allocation per iteration.

use crate::diffeomorphic::local_cc::CcSats;

/// Dense-field and control-point scratch buffers for one BSplineSyN registration.
///
/// Sized to `n = nz * ny * nx` dense voxels and `cp_n = cp_d[0]*cp_d[1]*cp_d[2]`
/// control points. All buffers are 0-initialized.
pub(super) struct BSplineSyNBuffers {
    // â”€â”€ Control-point lattice (one field per SyN branch, per spatial axis) â”€â”€
    pub cp1z: Vec<f32>,
    pub cp1y: Vec<f32>,
    pub cp1x: Vec<f32>,
    pub cp2z: Vec<f32>,
    pub cp2y: Vec<f32>,
    pub cp2x: Vec<f32>,

    // â”€â”€ Dense velocity fields vâ‚, vâ‚‚ â”€â”€
    pub v1z: Vec<f32>,
    pub v1y: Vec<f32>,
    pub v1x: Vec<f32>,
    pub v2z: Vec<f32>,
    pub v2y: Vec<f32>,
    pub v2x: Vec<f32>,

    // â”€â”€ Exponential maps Ï†â‚ = exp(vâ‚), Ï†â‚‚ = exp(vâ‚‚) â”€â”€
    pub phi1_z: Vec<f32>,
    pub phi1_y: Vec<f32>,
    pub phi1_x: Vec<f32>,
    pub phi2_z: Vec<f32>,
    pub phi2_y: Vec<f32>,
    pub phi2_x: Vec<f32>,

    // â”€â”€ Scaling-and-squaring scratch (ping-pong buffers) â”€â”€
    pub scratch_ss_z: Vec<f32>,
    pub scratch_ss_y: Vec<f32>,
    pub scratch_ss_x: Vec<f32>,

    // â”€â”€ Warped images â”€â”€
    pub i_w: Vec<f32>,
    pub j_w: Vec<f32>,

    // â”€â”€ Image gradients â”€â”€
    pub gi_z: Vec<f32>,
    pub gi_y: Vec<f32>,
    pub gi_x: Vec<f32>,
    pub gj_z: Vec<f32>,
    pub gj_y: Vec<f32>,
    pub gj_x: Vec<f32>,

    // â”€â”€ CC force fields â”€â”€
    pub u1z: Vec<f32>,
    pub u1y: Vec<f32>,
    pub u1x: Vec<f32>,
    pub u2z: Vec<f32>,
    pub u2y: Vec<f32>,
    pub u2x: Vec<f32>,

    // â”€â”€ Local-CC statistics and per-slice reductions â”€â”€
    pub cc_sats: CcSats,
    pub cc_slices: Vec<(f64, usize)>,

    // â”€â”€ CP-space gradient and Laplacian buffers â”€â”€
    pub cp_accum: Vec<f64>,
    pub cp_weight: Vec<f64>,
    pub d1z: Vec<f32>,
    pub d1y: Vec<f32>,
    pub d1x: Vec<f32>,
    pub d2z: Vec<f32>,
    pub d2y: Vec<f32>,
    pub d2x: Vec<f32>,
    pub l1z: Vec<f32>,
    pub l1y: Vec<f32>,
    pub l1x: Vec<f32>,
    pub l2z: Vec<f32>,
    pub l2y: Vec<f32>,
    pub l2x: Vec<f32> }

impl BSplineSyNBuffers {
    /// Allocate all buffers for a volume with `n = nz*ny*nx` voxels and
    /// `cp_n = cp_d[0]*cp_d[1]*cp_d[2]` control points.
    pub(super) fn new(n: usize, cp_n: usize, dims: [usize; 3], cc_radius: usize) -> Self {
        Self {
            cp1z: vec![0.0_f32; cp_n],
            cp1y: vec![0.0_f32; cp_n],
            cp1x: vec![0.0_f32; cp_n],
            cp2z: vec![0.0_f32; cp_n],
            cp2y: vec![0.0_f32; cp_n],
            cp2x: vec![0.0_f32; cp_n],
            v1z: vec![0.0_f32; n],
            v1y: vec![0.0_f32; n],
            v1x: vec![0.0_f32; n],
            v2z: vec![0.0_f32; n],
            v2y: vec![0.0_f32; n],
            v2x: vec![0.0_f32; n],
            phi1_z: vec![0.0_f32; n],
            phi1_y: vec![0.0_f32; n],
            phi1_x: vec![0.0_f32; n],
            phi2_z: vec![0.0_f32; n],
            phi2_y: vec![0.0_f32; n],
            phi2_x: vec![0.0_f32; n],
            scratch_ss_z: vec![0.0_f32; n],
            scratch_ss_y: vec![0.0_f32; n],
            scratch_ss_x: vec![0.0_f32; n],
            i_w: vec![0.0_f32; n],
            j_w: vec![0.0_f32; n],
            gi_z: vec![0.0_f32; n],
            gi_y: vec![0.0_f32; n],
            gi_x: vec![0.0_f32; n],
            gj_z: vec![0.0_f32; n],
            gj_y: vec![0.0_f32; n],
            gj_x: vec![0.0_f32; n],
            u1z: vec![0.0_f32; n],
            u1y: vec![0.0_f32; n],
            u1x: vec![0.0_f32; n],
            u2z: vec![0.0_f32; n],
            u2y: vec![0.0_f32; n],
            u2x: vec![0.0_f32; n],
            cc_sats: CcSats::new(dims, cc_radius),
            cc_slices: vec![(0.0_f64, 0usize); dims[0]],
            cp_accum: vec![0.0_f64; cp_n],
            cp_weight: vec![0.0_f64; cp_n],
            d1z: vec![0.0_f32; cp_n],
            d1y: vec![0.0_f32; cp_n],
            d1x: vec![0.0_f32; cp_n],
            d2z: vec![0.0_f32; cp_n],
            d2y: vec![0.0_f32; cp_n],
            d2x: vec![0.0_f32; cp_n],
            l1z: vec![0.0_f32; cp_n],
            l1y: vec![0.0_f32; cp_n],
            l1x: vec![0.0_f32; cp_n],
            l2z: vec![0.0_f32; cp_n],
            l2y: vec![0.0_f32; cp_n],
            l2x: vec![0.0_f32; cp_n] }
    }
}
