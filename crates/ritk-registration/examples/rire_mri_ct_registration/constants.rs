//! Ground-truth constants and type aliases for the RIRE CT/MRI T1
//! registration validation example.

use coeus_core::SequentialBackend;
use ritk_image::burn::backend::Autodiff;

/// CPU backend — no autodiff needed for the NCC/resampling validation workflow.
pub type B = SequentialBackend;

/// Autodiff backend required by the CMA-ES optimizer (tracks computation graph
/// for gradient-capable metrics; no `.backward()` is called by CMA-ES itself).
pub type RegB = Autodiff<SequentialBackend>;

// ── Ground-truth constants ────────────────────────────────────────────────────

/// Ground-truth rotation matrix R (row-major 3×3) for the CT → MRI T1 transform.
///
/// Derived from the ITK Euler3DTransform in
/// `training_001_ct_to_mr_T1_ground_truth.tfm` using the ITK convention
/// `R = Rz(aZ) · Rx(aX) · Ry(aY)`.
pub const GT_ROT: [f64; 9] = [
    0.997000003,
    0.077380155,
    -0.001818059,
    -0.077397855,
    0.996449628,
    -0.033131713,
    -0.000752132,
    0.033173032,
    0.999449341,
];

/// Ground-truth translation vector t (mm) for the CT → MRI T1 transform.
pub const GT_TRANS: [f64; 3] = [5.03685847, -17.49694636, -27.16499259];

/// Ground-truth Euler ZYX angles [α, β, γ] in RITK convention (radians).
/// Derived from the ITK Euler3DTransform: α ≈ 0.077 rad ≈ 4.4°.
pub const GT_EULER_ZYX: [f64; 3] = [0.077_40, 0.001_818, -0.033_14];

/// Ground-truth translation [tz, ty, tx] in RITK [z, y, x] order (mm).
/// This is GT_TRANS reversed: (x, y, z) → (z, y, x).
pub const GT_TRANS_ZYX: [f64; 3] = [-27.165, -17.497, 5.037];

/// RIRE 8-corner fiducial point pairs for TRE (Target Registration Error).
///
/// Each row: `[src_x, src_y, src_z, dst_x, dst_y, dst_z]` in mm (RIRE [x,y,z]).
/// `src` = CT volume corner, `dst` = corresponding MRI T1 physical position.
pub const RIRE_CORNERS: [[f64; 6]; 8] = [
    [0.0000, 0.0000, 0.0000, 5.0369, -17.4970, -27.1650],
    [333.9870, 0.0000, 0.0000, 338.0219, -43.3470, -27.4162],
    [0.0000, 333.9870, 0.0000, 30.8808, 315.3043, -16.0856],
    [333.9870, 333.9870, 0.0000, 363.8658, 289.4544, -16.3368],
    [0.0000, 0.0000, 112.0000, 4.8333, -21.2077, 84.7733],
    [333.9870, 0.0000, 112.0000, 337.8183, -47.0576, 84.5221],
    [0.0000, 333.9870, 112.0000, 30.6772, 311.5937, 95.8527],
    [333.9870, 333.9870, 112.0000, 363.6622, 285.7437, 95.6015],
];
