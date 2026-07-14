//! Pre-allocated scratch buffers for zero-allocation SyN iteration.
//!
//! All buffers are sized at construction time; the iteration loop performs
//! zero heap allocations by writing exclusively into these pre-owned buffers.

/// Dense scratch buffers for one SyN registration.
///
/// Sized to `n = nz * ny * nx` voxels. All buffers are 0-initialized.
///
/// The smoothing scratch buffer is handled by [`crate::deformable_field_ops::FieldSmoother`] implementations;
/// it is no longer stored here.
pub(super) struct SyNBuffers {
    // ── Velocity fields v₁, v₂ (output) ──
    pub v1z: Vec<f32>,
    pub v1y: Vec<f32>,
    pub v1x: Vec<f32>,
    pub v2z: Vec<f32>,
    pub v2y: Vec<f32>,
    pub v2x: Vec<f32>,

    // ── Exponential maps φ₁ = exp(v₁), φ₂ = exp(v₂) ──
    pub phi1_z: Vec<f32>,
    pub phi1_y: Vec<f32>,
    pub phi1_x: Vec<f32>,
    pub phi2_z: Vec<f32>,
    pub phi2_y: Vec<f32>,
    pub phi2_x: Vec<f32>,

    // ── Scaling-and-squaring scratch (ping-pong buffers) ──
    pub scratch_ss_z: Vec<f32>,
    pub scratch_ss_y: Vec<f32>,
    pub scratch_ss_x: Vec<f32>,

    // ── Warped images ──
    pub i_w: Vec<f32>,
    pub j_w: Vec<f32>,

    // ── Image gradients ──
    pub gi_z: Vec<f32>,
    pub gi_y: Vec<f32>,
    pub gi_x: Vec<f32>,
    pub gj_z: Vec<f32>,
    pub gj_y: Vec<f32>,
    pub gj_x: Vec<f32>,

    // ── CC force fields ──
    pub u1z: Vec<f32>,
    pub u1y: Vec<f32>,
    pub u1x: Vec<f32>,
    pub u2z: Vec<f32>,
    pub u2y: Vec<f32>,
    pub u2x: Vec<f32>,

    // ── Per-z-slice CC reductions ──
    pub cc_slices: Vec<(f64, usize)>,
}

impl SyNBuffers {
    /// Allocate all 29 buffers for a volume with `n = nz*ny*nx` voxels.
    pub(super) fn new(n: usize, nz: usize) -> Self {
        Self {
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
            cc_slices: vec![(0.0_f64, 0usize); nz],
        }
    }
}
