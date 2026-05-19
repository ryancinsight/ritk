//! Tests for `jacobian_determinant` and `analyze_jacobian`.
//!
//! Extracted to respect the 500-line structural limit.

use super::{analyze_jacobian, jacobian_determinant};
use crate::filter::ops::extract_vec;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_disp(data: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<TestBackend, 3> {
    let device = Default::default();
    let tensor =
        Tensor::<TestBackend, 3>::from_data(TensorData::new(data, Shape::new(dims)), &device);
    Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new(spacing),
        Direction::identity(),
    )
}

fn zero_disp(dims: [usize; 3]) -> Image<TestBackend, 3> {
    let n = dims[0] * dims[1] * dims[2];
    make_disp(vec![0.0f32; n], dims, [1.0, 1.0, 1.0])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Zero displacement field → det(J) = 1.0 at every voxel.
///
/// Proof: u = 0 ⟹ ∀ i,j: ∂u_i/∂x_j = 0 (finite differences of zeros are
/// zero).  Therefore J = I + 0 = I and det(I) = 1.
#[test]
fn identity_field_gives_determinant_one() {
    let dims = [4, 4, 4];
    let disp = zero_disp(dims);

    let jac = jacobian_determinant(&disp, &disp, &disp)
        .expect("jacobian_determinant must succeed on zero field");

    let (vals, jac_dims) = extract_vec(&jac).unwrap();
    assert_eq!(jac_dims, dims, "output shape must match input shape");

    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-5,
            "voxel {i}: expected det = 1.0, got {v}"
        );
    }
}

/// Uniform expansion u_z = α·z, u_y = α·y, u_x = α·x with α = 0.5.
///
/// Analytical Jacobian at interior voxels (spacing = 1.0):
///   ∂u_z/∂z = central diff = (α(z+1) − α(z−1)) / 2 = α  (= 0.5)
///   ∂u_y/∂y = α,  ∂u_x/∂x = α
///   All cross-derivatives = 0 (each component depends on exactly one axis)
///
/// J = diag(1+α, 1+α, 1+α) = diag(1.5, 1.5, 1.5)
/// det(J) = 1.5³ = 3.375
///
/// Verification at interior index [4,4,4] with tolerance 1e-3 (f32 arithmetic).
#[test]
fn uniform_expansion_gives_correct_determinant() {
    let nz = 8usize;
    let ny = 8usize;
    let nx = 8usize;
    let alpha = 0.5f32;

    let mut data_z = vec![0.0f32; nz * ny * nx];
    let mut data_y = vec![0.0f32; nz * ny * nx];
    let mut data_x = vec![0.0f32; nz * ny * nx];

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = z * ny * nx + y * nx + x;
                data_z[i] = alpha * z as f32;
                data_y[i] = alpha * y as f32;
                data_x[i] = alpha * x as f32;
            }
        }
    }

    let disp_z = make_disp(data_z, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let disp_y = make_disp(data_y, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let disp_x = make_disp(data_x, [nz, ny, nx], [1.0, 1.0, 1.0]);

    let jac = jacobian_determinant(&disp_z, &disp_y, &disp_x)
        .expect("jacobian_determinant must succeed on uniform expansion field");

    let (vals, _) = extract_vec(&jac).unwrap();

    // Interior voxel [4, 4, 4] uses central differences on all three axes.
    let interior_idx = 4 * ny * nx + 4 * nx + 4;
    let det = vals[interior_idx];
    let expected = (1.0f32 + alpha).powi(3); // 3.375

    assert!(
        (det - expected).abs() < 1e-3,
        "interior voxel [4,4,4]: expected det = {expected:.6}, got {det:.6}"
    );
}

/// analyze_jacobian on a zero displacement field.
///
/// All det(J) = 1.0 exactly. Classification:
///   num_folded    = 0  (1.0 ≤ 0 is false)
///   num_compressed = 0  (1.0 < 1.0 is false)
///   num_expanded  = N  (1.0 ≥ 1.0 is true)
///   num_valid     = N
#[test]
fn analyze_jacobian_zero_field() {
    let dims = [4, 4, 4];
    let n = dims[0] * dims[1] * dims[2]; // 64
    let disp = zero_disp(dims);
    let jac = jacobian_determinant(&disp, &disp, &disp).unwrap();
    let stats = analyze_jacobian(&jac).unwrap();

    assert!(
        (stats.min - 1.0).abs() < 1e-5,
        "min should be ≈ 1.0, got {}",
        stats.min
    );
    assert!(
        (stats.max - 1.0).abs() < 1e-5,
        "max should be ≈ 1.0, got {}",
        stats.max
    );
    assert_eq!(
        stats.num_folded, 0,
        "num_folded must be 0 for identity field"
    );
    assert_eq!(
        stats.num_compressed, 0,
        "num_compressed must be 0: det = 1.0 is not < 1.0"
    );
    assert_eq!(
        stats.num_expanded, n,
        "num_expanded must equal total voxels (det ≥ 1.0)"
    );
    assert_eq!(stats.num_valid, n, "num_valid must equal total voxels");
}

/// Output shape of `jacobian_determinant` matches the input shape.
#[test]
fn output_shape_matches_input() {
    let dims = [5, 4, 3];
    let n = dims[0] * dims[1] * dims[2];
    let disp = make_disp(vec![0.0f32; n], dims, [1.0, 1.0, 1.0]);
    let jac = jacobian_determinant(&disp, &disp, &disp).unwrap();
    assert_eq!(jac.shape(), dims, "output shape must equal input shape");
}

/// analyze_jacobian on a [4,4,4] zero field: num_folded = 0, total_voxels = 64.
#[test]
fn analyze_jacobian_returns_correct_stats_for_identity() {
    let dims = [4, 4, 4];
    let disp = zero_disp(dims);
    let jac = jacobian_determinant(&disp, &disp, &disp).unwrap();
    let stats = analyze_jacobian(&jac).unwrap();

    assert_eq!(
        stats.num_folded, 0,
        "num_folded must be 0 for identity field"
    );
    assert_eq!(
        stats.total_voxels, 64,
        "total_voxels must be 64 for [4,4,4]"
    );
}
