//! Atlas-typed tests for the affine transform sister adapter.
//!
//! Strict subtractive on test surface (ADR 0012 §Decision §Sub-batch #3.f):
//! every affine case exercises [`super::super::atlas_affine::AtlasAffineTransform`]
//! (sister API) + `super::super::atlas_affine::AtlasAffineError` over
//! host-slice params, rather than the legacy tensor-backed
//! `super::super::affine::AffineTransform<B, D>` (which carries the
//! `Module` derive and `Param<Tensor>` wrapping — sub-batch #5 owns the
//! `coeus_nn::Module` derive migration).
//!
//! The `test_atlas_affine_seeded_from_rigid_rotation_reproduces_rigid`
//! test uses the same closed-form Euler convention documented by
//! `RigidTransform::build_rotation_matrix()` (`R = R_z * R_y * R_x`) while
//! keeping this rewritten test file free of legacy tensor construction.

use super::super::atlas_affine::AtlasAffineTransform;
use coeus_core::MoiraiBackend;

// ── Sister-only tests ─────────────────────────────────────────────────────

/// Tolerance for near-zero assertions in identity and translation tests.
const NEAR_ZERO: f32 = 1e-6;

/// Approximate equality for atlas-side affine round-trip tests.
/// Euler-angle composition accumulates ~5 ULP; 1e-5 is tight but robust.
const RIGID_AFFINE_CONSISTENCY_TOL: f32 = 1e-5;

fn rigid_rotation_matrix(angles: [f32; 3]) -> [f32; 9] {
    let [alpha, beta, gamma] = angles;
    let (sx, cx) = alpha.sin_cos();
    let (sy, cy) = beta.sin_cos();
    let (sz, cz) = gamma.sin_cos();

    [
        cz * cy,
        cz * sy * sx - sz * cx,
        cz * sy * cx + sz * sx,
        sz * cy,
        sz * sy * sx + cz * cx,
        sz * sy * cx - cz * sx,
        -sy,
        cy * sx,
        cy * cx,
    ]
}

fn apply_affine(
    matrix: &[f32; 9],
    translation: &[f32; 3],
    center: &[f32; 3],
    point: [f32; 3],
) -> [f32; 3] {
    let shifted = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];

    [
        matrix[0] * shifted[0]
            + matrix[1] * shifted[1]
            + matrix[2] * shifted[2]
            + center[0]
            + translation[0],
        matrix[3] * shifted[0]
            + matrix[4] * shifted[1]
            + matrix[5] * shifted[2]
            + center[1]
            + translation[1],
        matrix[6] * shifted[0]
            + matrix[7] * shifted[1]
            + matrix[8] * shifted[2]
            + center[2]
            + translation[2],
    ]
}

#[test]
fn test_atlas_affine_transform_identity() {
    let transform = AtlasAffineTransform::<MoiraiBackend, 3>::identity(None);
    let matrix_values: Vec<f32> = transform.matrix().to_vec();
    let translation_values: Vec<f32> = transform.translation().to_vec();
    let center_values: Vec<f32> = transform.center().to_vec();

    let points_values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dims = [2usize, 3];
    let points = ritk_image::native::Image::<f32, MoiraiBackend, 2>::from_flat(
        points_values,
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
    )
    .expect("atlas points image");

    let transformed = transform
        .transform_points::<MoiraiBackend>(&points)
        .expect("atlas identity transform_points");
    let slice = transformed
        .data_slice()
        .expect("atlas identity output host-slice access");

    assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(matrix_values.len(), 9);
    assert_eq!(translation_values, &[0.0, 0.0, 0.0]);
    assert_eq!(center_values, &[0.0, 0.0, 0.0]);
}

#[test]
fn test_atlas_affine_transform_translation_with_center() {
    // Identity 3-D matrix.
    let matrix: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let translation: [f32; 3] = [1.0, 1.0, 0.0];
    let center: [f32; 3] = [10.0, 10.0, 0.0];

    let transform =
        AtlasAffineTransform::<MoiraiBackend, 3>::construct(&matrix, &translation, &center);

    // Point at center `[10, 10, 0]`: T(c) = c + t = `[11, 11, 0]`.
    let points_values: Vec<f32> = vec![10.0, 10.0, 0.0];
    let dims = [1usize, 3];
    let points = ritk_image::native::Image::<f32, MoiraiBackend, 2>::from_flat(
        points_values,
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
    )
    .expect("atlas centroid point image");
    let transformed = transform
        .transform_points::<MoiraiBackend>(&points)
        .expect("atlas translation transform_points");
    let slice = transformed
        .data_slice()
        .expect("atlas translation output host-slice access");

    assert_eq!(slice, &[11.0, 11.0, 0.0]);
}

#[test]
fn test_atlas_affine_transform_scale_with_center() {
    // 2-D scale-by-2 with center `[1, 1]`.
    let matrix: [f32; 4] = [2.0, 0.0, 0.0, 2.0];
    let translation: [f32; 2] = [0.0, 0.0];
    let center: [f32; 2] = [1.0, 1.0];

    let transform =
        AtlasAffineTransform::<MoiraiBackend, 2>::construct(&matrix, &translation, &center);

    // Point `[2, 1]` (1 unit right of center): T(x) = 2 · (x - c) + c = `[3, 1]`.
    let points_values: Vec<f32> = vec![2.0, 1.0];
    let dims = [1usize, 2];
    let points = ritk_image::native::Image::<f32, MoiraiBackend, 2>::from_flat(
        points_values,
        dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
    )
    .expect("atlas off-centre point image");
    let transformed = transform
        .transform_points::<MoiraiBackend>(&points)
        .expect("atlas scale transform_points");
    let slice = transformed
        .data_slice()
        .expect("atlas scale output host-slice access");

    assert!((slice[0] - 3.0).abs() < NEAR_ZERO);
    assert!((slice[1] - 1.0).abs() < NEAR_ZERO);
}

#[test]
fn test_atlas_affine_rejects_homogeneous_matrix_shape() {
    // 16-element buffer passed for a [3, 3] (D=3) affine.
    let matrix: [f32; 16] = [0.0; 16];
    let translation: [f32; 3] = [0.0; 3];
    let center: [f32; 3] = [0.0; 3];
    let err = AtlasAffineTransform::<MoiraiBackend, 3>::try_new(&matrix, &translation, &center)
        .expect_err("atlas affine must reject non-[D,D] matrix");
    let msg = err.to_string();
    assert!(
        msg.contains("[9]") && msg.contains("[16]"),
        "expected [9] vs [16] matrix-length rejection, got {msg}"
    );
}

#[test]
fn test_atlas_affine_seeded_from_rigid_rotation_reproduces_rigid() {
    let rotation_angles: [f32; 3] = [0.3, -0.2, 0.1];
    let translation_vec: [f32; 3] = [2.0, -1.0, 3.0];
    let center_vec: [f32; 3] = [5.0, 6.0, 7.0];

    // Matches `RigidTransform::build_rotation_matrix()` for 3-D:
    // Euler angles x/y/z with composition `R_z(gamma) * R_y(beta) * R_x(alpha)`.
    let rotation_matrix = rigid_rotation_matrix(rotation_angles);

    let atlas = AtlasAffineTransform::<MoiraiBackend, 3>::construct(
        &rotation_matrix,
        &translation_vec,
        &center_vec,
    );

    let pts_values: Vec<f32> = vec![1.0, 2.0, 3.0, 10.0, -4.0, 0.5];
    let pts_dims = [2usize, 3];
    let pts = ritk_image::native::Image::<f32, MoiraiBackend, 2>::from_flat(
        pts_values,
        pts_dims,
        ritk_spatial::Point::origin(),
        ritk_spatial::Spacing::uniform(1.0),
        ritk_spatial::Direction::identity(),
    )
    .expect("atlas points image");
    let atlas_ext = atlas
        .transform_points::<MoiraiBackend>(&pts)
        .expect("atlas rigid-seeded transform_points");

    let expected = [
        apply_affine(
            &rotation_matrix,
            &translation_vec,
            &center_vec,
            [1.0, 2.0, 3.0],
        ),
        apply_affine(
            &rotation_matrix,
            &translation_vec,
            &center_vec,
            [10.0, -4.0, 0.5],
        ),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();

    let atlas_slice = atlas_ext
        .data_slice()
        .expect("atlas rigid-seeded output host-slice access");

    for (i, (expected, actual)) in expected.iter().zip(atlas_slice.iter()).enumerate() {
        assert!(
            (expected - actual).abs() < RIGID_AFFINE_CONSISTENCY_TOL,
            "atlas[{i}] = {actual}, expected[{i}] = {expected}"
        );
    }
}
