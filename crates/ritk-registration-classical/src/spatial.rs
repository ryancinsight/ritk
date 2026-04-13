use crate::error::{RegistrationError, Result};
use nalgebra::{Matrix3, Vector3};
use ndarray::{Array2, Array3};

/// Spatial transformation types for image registration
#[derive(Debug, Clone)]
pub enum SpatialTransform {
    /// Rigid body transformation (rotation + translation)
    RigidBody {
        rotation: [f64; 9],    // 3x3 rotation matrix
        translation: [f64; 3], // Translation vector
    },
    /// Affine transformation (linear + translation)
    Affine {
        matrix: [f64; 12], // 3x4 affine matrix [R|t]
    },
    /// Non-rigid deformation field
    NonRigid {
        deformation_field: Array3<[f64; 3]>, // Displacement vectors at each voxel
    },
}

pub(crate) fn compute_centroid(points: &Array2<f64>) -> [f64; 3] {
    let n = points.nrows() as f64;
    let sum_x: f64 = points.column(0).sum();
    let sum_y: f64 = points.column(1).sum();
    let sum_z: f64 = points.column(2).sum();

    [sum_x / n, sum_y / n, sum_z / n]
}

pub(crate) fn center_points(points: &Array2<f64>, centroid: &[f64; 3]) -> Array2<f64> {
    let mut centered = points.clone();
    for mut row in centered.outer_iter_mut() {
        row[0] -= centroid[0];
        row[1] -= centroid[1];
        row[2] -= centroid[2];
    }
    centered
}

/// Compute the optimal rotation matrix that aligns `moving` onto `fixed` via the
/// **Kabsch algorithm** (Kabsch 1976).
///
/// ## Algorithm
///
/// Given two zero-mean point clouds P (fixed, N×3) and Q (moving, N×3), the
/// algorithm finds R = argmin_R Σ ||P_i − R Q_i||²:
///
/// ```text
/// 1. Compute cross-covariance: H = Qᵀ P  (3×3, moving^T × fixed)
/// 2. SVD decompose:           H = U Σ Vᵀ
/// 3. Correct for reflection:  d = sign(det(V Uᵀ))
/// 4. Optimal rotation:        R = V · diag(1, 1, d) · Uᵀ
/// ```
///
/// The correction term `d` handles the case where the naive V Uᵀ product is an
/// improper rotation (reflection, det = −1) — Umeyama (1991) §III.
///
/// **Convention note:** H = Qᵀ P (not Pᵀ Q).  With this sign convention the
/// returned R satisfies R · q_i ≈ p_i (rotates moving to fixed), consistent with
/// how `compute_fre` applies the transform.
///
/// ## References
/// - Kabsch, W. (1976). "A solution for the best rotation to relate two sets of
///   vectors." *Acta Crystallographica A* **32**(5), 922–923.
/// - Umeyama, S. (1991). "Least-squares estimation of transformation parameters
///   between two point patterns." *IEEE PAMI* **13**(4), 376–380.
pub(crate) fn kabsch_algorithm(fixed: &Array2<f64>, moving: &Array2<f64>) -> Result<[f64; 9]> {
    let n = fixed.nrows();
    if n == 0 {
        return Err(RegistrationError::InvalidInput(
            "Kabsch algorithm requires at least one point pair".to_string(),
        ));
    }

    // Build cross-covariance H = Qᵀ P  (moving^T × fixed)
    //
    // With this convention the optimal rotation minimising sum ||P_i − R Q_i||²
    // is R = V Uᵀ (Kabsch 1976, eq. 4).  The roles are:
    //   P = fixed  (N×3), Q = moving  (N×3)
    //   H[r,c] = Σ_i  Q[i,r] · P[i,c]
    let mut h = Matrix3::zeros();
    for i in 0..n {
        for r in 0..3 {
            for c in 0..3 {
                h[(r, c)] += moving[[i, r]] * fixed[[i, c]];
            }
        }
    }

    // SVD: H = U Σ Vᵀ
    let svd = nalgebra::linalg::SVD::new(h, true, true);
    let u = svd.u.ok_or_else(|| {
        RegistrationError::NumericalFailure(
            "SVD failed to converge for Kabsch algorithm".to_string(),
        )
    })?;
    let v_t = svd.v_t.ok_or_else(|| {
        RegistrationError::NumericalFailure(
            "SVD failed to converge for Kabsch algorithm".to_string(),
        )
    })?;

    let v: Matrix3<f64> = v_t.transpose();

    // Detect reflection: d = sign(det(V Uᵀ))
    let r_candidate: Matrix3<f64> = v * u.transpose();
    let d = if r_candidate.determinant() < 0.0 {
        -1.0_f64
    } else {
        1.0_f64
    };

    // Corrected rotation: R = V · diag(1, 1, d) · Uᵀ
    let diag = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, d));
    let rotation = v * diag * u.transpose();

    // Flatten row-major: rotation[row*3 + col]
    Ok([
        rotation[(0, 0)],
        rotation[(0, 1)],
        rotation[(0, 2)],
        rotation[(1, 0)],
        rotation[(1, 1)],
        rotation[(1, 2)],
        rotation[(2, 0)],
        rotation[(2, 1)],
        rotation[(2, 2)],
    ])
}

pub(crate) fn build_homogeneous_matrix(rotation: &[f64; 9], translation: &[f64; 3]) -> [f64; 16] {
    [
        rotation[0],
        rotation[1],
        rotation[2],
        translation[0],
        rotation[3],
        rotation[4],
        rotation[5],
        translation[1],
        rotation[6],
        rotation[7],
        rotation[8],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

pub(crate) fn extract_spatial_transform(homogeneous: &[f64; 16]) -> Result<SpatialTransform> {
    let rotation = [
        homogeneous[0],
        homogeneous[1],
        homogeneous[2],
        homogeneous[4],
        homogeneous[5],
        homogeneous[6],
        homogeneous[8],
        homogeneous[9],
        homogeneous[10],
    ];
    let translation = [homogeneous[3], homogeneous[7], homogeneous[11]];

    Ok(SpatialTransform::RigidBody {
        rotation,
        translation,
    })
}

pub(crate) fn transform_point(transform: &[f64; 16], point: [f64; 3]) -> [f64; 3] {
    [
        transform[0] * point[0] + transform[1] * point[1] + transform[2] * point[2] + transform[3],
        transform[4] * point[0] + transform[5] * point[1] + transform[6] * point[2] + transform[7],
        transform[8] * point[0]
            + transform[9] * point[1]
            + transform[10] * point[2]
            + transform[11],
    ]
}

/// Generate 9-DOF affine perturbation candidates for coordinate-descent MI optimisation.
///
/// Each candidate is `[rx, ry, rz, tx, ty, tz, sx, sy, sz]`:
/// - `rx/ry/rz`: Euler angle deltas (radians)
/// - `tx/ty/tz`: translation deltas (voxels)
/// - `sx/sy/sz`: multiplicative scale deltas (pure scale, ×(1 + δ))
///
/// Step sizes are kept small so that the MI landscape remains smooth:
/// Δθ = 0.01 rad, Δt = 1 voxel, Δs = 0.02 (2 % scale change per step).
pub(crate) fn generate_affine_perturbations() -> Vec<[f64; 9]> {
    vec![
        // Rotation perturbations (rx, ry, rz, tx, ty, tz, sx, sy, sz)
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        // Translation perturbations
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        // Scale perturbations (±2 % per axis)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02],
    ]
}

/// Apply a 9-DOF affine perturbation `[rx, ry, rz, tx, ty, tz, sx, sy, sz]` to a
/// homogeneous 4×4 transform.
///
/// ## Algorithm
///
/// The current 3×3 linear block A is factored as `A = R · S` where R is a pure
/// rotation and `S = diag(sx, sy, sz)` holds the current scale.  The new scale is
/// updated multiplicatively: `s'_i = s_i · (1 + δs_i)`.  The rotation is updated
/// as in `apply_transform_perturbation` (ZYX Euler deltas):
///
/// ```text
/// R_new = R_delta · R
/// S_new = diag(s · (1 + δs))
/// A_new = R_new · S_new
/// ```
///
/// Translation is updated additively.
pub(crate) fn apply_affine_perturbation(
    base_transform: &[f64; 16],
    perturbation: &[f64; 9],
) -> [f64; 16] {
    let (rx, ry, rz) = (perturbation[0], perturbation[1], perturbation[2]);
    let (dtx, dty, dtz) = (perturbation[3], perturbation[4], perturbation[5]);
    let (dsx, dsy, dsz) = (perturbation[6], perturbation[7], perturbation[8]);

    // Rotation delta (ZYX Euler)
    let (sx_r, cx) = rx.sin_cos();
    let (sy_r, cy) = ry.sin_cos();
    let (sz_r, cz) = rz.sin_cos();
    let dr = [
        cy * cz,
        cx * sz_r + sx_r * sy_r * cz,
        sx_r * sz_r - cx * sy_r * cz,
        -cy * sz_r,
        cx * cz - sx_r * sy_r * sz_r,
        sx_r * cz + cx * sy_r * sz_r,
        sy_r,
        -sx_r * cy,
        cx * cy,
    ];

    // Extract current 3×3 linear block A from the homogeneous matrix
    let a = [
        base_transform[0],
        base_transform[1],
        base_transform[2],
        base_transform[4],
        base_transform[5],
        base_transform[6],
        base_transform[8],
        base_transform[9],
        base_transform[10],
    ];

    // Decompose A = R · S by extracting column norms (scale) and normalising
    let col_norms = [
        (a[0].powi(2) + a[3].powi(2) + a[6].powi(2))
            .sqrt()
            .max(f64::EPSILON),
        (a[1].powi(2) + a[4].powi(2) + a[7].powi(2))
            .sqrt()
            .max(f64::EPSILON),
        (a[2].powi(2) + a[5].powi(2) + a[8].powi(2))
            .sqrt()
            .max(f64::EPSILON),
    ];
    // Current rotation R = A · S^{-1} (normalise each column)
    let r_cur = [
        a[0] / col_norms[0],
        a[1] / col_norms[1],
        a[2] / col_norms[2],
        a[3] / col_norms[0],
        a[4] / col_norms[1],
        a[5] / col_norms[2],
        a[6] / col_norms[0],
        a[7] / col_norms[1],
        a[8] / col_norms[2],
    ];
    // New rotation: R_new = R_delta · R_cur
    let r_new = mat3_mul(&dr, &r_cur);

    // New scale: multiplicative update
    let s_new = [
        col_norms[0] * (1.0 + dsx),
        col_norms[1] * (1.0 + dsy),
        col_norms[2] * (1.0 + dsz),
    ];

    // Rebuild A_new = R_new · diag(s_new)  (scale each column by s_new[col])
    let a_new = [
        r_new[0] * s_new[0],
        r_new[1] * s_new[1],
        r_new[2] * s_new[2],
        r_new[3] * s_new[0],
        r_new[4] * s_new[1],
        r_new[5] * s_new[2],
        r_new[6] * s_new[0],
        r_new[7] * s_new[1],
        r_new[8] * s_new[2],
    ];

    [
        a_new[0],
        a_new[1],
        a_new[2],
        base_transform[3] + dtx,
        a_new[3],
        a_new[4],
        a_new[5],
        base_transform[7] + dty,
        a_new[6],
        a_new[7],
        a_new[8],
        base_transform[11] + dtz,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

pub(crate) fn generate_transform_perturbations() -> Vec<[f64; 6]> {
    // Generate small perturbations for rigid body transform (3 rotation + 3 translation)
    vec![
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],  // Small rotation around x
        [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], // Negative rotation
        [0.0, 0.01, 0.0, 0.0, 0.0, 0.0],  // Small rotation around y
        [0.0, 0.0, 0.01, 0.0, 0.0, 0.0],  // Small rotation around z
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   // Translation in x
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   // Translation in y
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   // Translation in z
    ]
}

/// Apply a 6-DOF perturbation `[rx, ry, rz, tx, ty, tz]` to a homogeneous 4×4 transform.
///
/// The rotation perturbation is composed as successive small-angle rotations about
/// the X, Y, and Z axes (ZYX Tait-Bryan convention, intrinsic):
///
/// ```text
/// Rx(α) = [[1,  0,     0   ],
///          [0,  cos α, -sin α],
///          [0,  sin α,  cos α]]
///
/// R_delta = Rz(γ) · Ry(β) · Rx(α)   (intrinsic ZYX)
///
/// T_new = T_delta · T_base            (prepend rotation delta, append translation delta)
/// ```
///
/// Translation perturbation `[tx, ty, tz]` is added directly to the current translation
/// column (columns 3, 7, 11 of the homogeneous matrix).
pub(crate) fn apply_transform_perturbation(
    base_transform: &[f64; 16],
    perturbation: &[f64; 6],
) -> [f64; 16] {
    let (rx, ry, rz) = (perturbation[0], perturbation[1], perturbation[2]);

    // Build incremental rotation matrix via small-angle Rodrigues (exact for any angle)
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();

    // R_delta = Rz · Ry · Rx  (row-major, row = output axis)
    let dr = [
        cy * cz,
        cx * sz + sx * sy * cz,
        sx * sz - cx * sy * cz,
        -cy * sz,
        cx * cz - sx * sy * sz,
        sx * cz + cx * sy * sz,
        sy,
        -sx * cy,
        cx * cy,
    ];

    // Extract base rotation (3×3) and translation from homogeneous matrix
    let br = [
        base_transform[0],
        base_transform[1],
        base_transform[2],
        base_transform[4],
        base_transform[5],
        base_transform[6],
        base_transform[8],
        base_transform[9],
        base_transform[10],
    ];

    // New rotation: R_new = R_delta · R_base  (3×3 matrix multiply)
    let r_new = mat3_mul(&dr, &br);

    // Updated translation: t_new = t_base + delta_t
    let tx_new = base_transform[3] + perturbation[3];
    let ty_new = base_transform[7] + perturbation[4];
    let tz_new = base_transform[11] + perturbation[5];

    [
        r_new[0], r_new[1], r_new[2], tx_new, r_new[3], r_new[4], r_new[5], ty_new, r_new[6],
        r_new[7], r_new[8], tz_new, 0.0, 0.0, 0.0, 1.0,
    ]
}

/// 3×3 matrix multiply: C = A · B (both stored row-major as `[f64; 9]`)
fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0_f64; 9];
    for r in 0..3 {
        for c_col in 0..3 {
            for k in 0..3 {
                c[r * 3 + c_col] += a[r * 3 + k] * b[k * 3 + c_col];
            }
        }
    }
    c
}

pub(crate) fn compute_fre(
    fixed: &Array2<f64>,
    moving: &Array2<f64>,
    rotation: &[f64; 9],
    translation: &[f64; 3],
) -> f64 {
    let mut sum_squared_error = 0.0;
    let n_points = fixed.nrows();

    for i in 0..n_points {
        let fixed_point = [fixed[[i, 0]], fixed[[i, 1]], fixed[[i, 2]]];
        let moving_point = [moving[[i, 0]], moving[[i, 1]], moving[[i, 2]]];

        // Apply transformation to moving point
        let transformed = [
            rotation[0] * moving_point[0]
                + rotation[1] * moving_point[1]
                + rotation[2] * moving_point[2]
                + translation[0],
            rotation[3] * moving_point[0]
                + rotation[4] * moving_point[1]
                + rotation[5] * moving_point[2]
                + translation[1],
            rotation[6] * moving_point[0]
                + rotation[7] * moving_point[1]
                + rotation[8] * moving_point[2]
                + translation[2],
        ];

        // Compute Euclidean distance
        let error = (fixed_point[0] - transformed[0]).powi(2)
            + (fixed_point[1] - transformed[1]).powi(2)
            + (fixed_point[2] - transformed[2]).powi(2);

        sum_squared_error += error.sqrt();
    }

    sum_squared_error / n_points as f64
}

/// Apply spatial transformation to image
///
/// # Arguments
/// * `image` - Input image to transform
/// * `transform` - Homogeneous transformation matrix
///
/// # Returns
/// Transformed image
pub fn apply_transform(image: &Array3<f64>, transform: &[f64; 16]) -> Array3<f64> {
    let shape = image.shape();
    let mut result = Array3::zeros((shape[0], shape[1], shape[2]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                // Transform coordinates
                let x = i as f64;
                let y = j as f64;
                let z = k as f64;

                let transformed = transform_point(transform, [x, y, z]);

                // Nearest neighbor sampling
                let ti = transformed[0].round() as isize;
                let tj = transformed[1].round() as isize;
                let tk = transformed[2].round() as isize;

                if ti >= 0
                    && ti < shape[0] as isize
                    && tj >= 0
                    && tj < shape[1] as isize
                    && tk >= 0
                    && tk < shape[2] as isize
                {
                    result[[i, j, k]] = image[[ti as usize, tj as usize, tk as usize]];
                }
            }
        }
    }

    result
}

/// Apply a 4×4 homogeneous transform to a 3-D volume using trilinear interpolation.
///
/// For each voxel `(i, j, k)` in the *output* volume the inverse transform is applied
/// to locate the corresponding source coordinates, which are then sampled from
/// `volume` via trilinear interpolation.  Voxels that map outside the source extent
/// are set to zero (clamp-to-black boundary condition).
///
/// ## Algorithm
///
/// ```text
/// for each (i,j,k) in output:
///   [xs, ys, zs]ᵀ = T · [i, j, k, 1]ᵀ    // forward transform to source coords
///   output[i,j,k] = trilinear_sample(volume, xs, ys, zs)
/// ```
///
/// Trilinear sampling uses the 8 surrounding voxel corners with weights
/// `(1−α)(1−β)(1−γ)`, …, `αβγ` where `α = xs − floor(xs)`.
///
/// The forward transform interpretation (output ← source) is consistent with
/// how `apply_transform` and `transform_point` are used throughout this module.
pub(crate) fn apply_transform_to_volume(
    volume: &Array3<f64>,
    transform: &[f64; 16],
) -> Array3<f64> {
    let shape = volume.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let mut result = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let src = transform_point(transform, [i as f64, j as f64, k as f64]);
                let xs = src[0];
                let ys = src[1];
                let zs = src[2];

                // Integer corners
                let x0 = xs.floor() as isize;
                let y0 = ys.floor() as isize;
                let z0 = zs.floor() as isize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let z1 = z0 + 1;

                // All 8 corners must be within the source volume for valid trilinear
                if x0 < 0
                    || x1 >= nx as isize
                    || y0 < 0
                    || y1 >= ny as isize
                    || z0 < 0
                    || z1 >= nz as isize
                {
                    // Out of bounds — leave as zero
                    continue;
                }

                let (x0, y0, z0) = (x0 as usize, y0 as usize, z0 as usize);
                let (x1, y1, z1) = (x1 as usize, y1 as usize, z1 as usize);

                let ax = xs - xs.floor();
                let ay = ys - ys.floor();
                let az = zs - zs.floor();

                // Trilinear weights for 8 corners
                result[[i, j, k]] = volume[[x0, y0, z0]] * (1.0 - ax) * (1.0 - ay) * (1.0 - az)
                    + volume[[x1, y0, z0]] * ax * (1.0 - ay) * (1.0 - az)
                    + volume[[x0, y1, z0]] * (1.0 - ax) * ay * (1.0 - az)
                    + volume[[x1, y1, z0]] * ax * ay * (1.0 - az)
                    + volume[[x0, y0, z1]] * (1.0 - ax) * (1.0 - ay) * az
                    + volume[[x1, y0, z1]] * ax * (1.0 - ay) * az
                    + volume[[x0, y1, z1]] * (1.0 - ax) * ay * az
                    + volume[[x1, y1, z1]] * ax * ay * az;
            }
        }
    }

    result
}
