use super::IterativeInverseDisplacementField;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::Direction;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn img(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, dims)
}

/// Direction of a conventional axial acquisition: index axis 0 (depth) along
/// world z, axis 1 (row) along y, axis 2 (column) along x.
fn axial() -> Direction<3> {
    Direction::from_rows([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
}

/// Rotation of the (x, y) plane by the exact 3-4-5 angle: orthonormal,
/// determinant 1, and not a permutation of the coordinate axes.
fn oblique_rotation() -> Direction<3> {
    Direction::from_rows([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
}

fn img_with_direction(
    data: Vec<f32>,
    dims: [usize; 3],
    direction: Direction<3>,
) -> Image<f32, B, 3> {
    ts::make_image_with::<f32, B, 3>(data, dims, None, None, Some(direction))
}

/// Deterministic spatially varying field. A *constant* field is useless here:
/// the inverse of a pure translation is its negation in every frame, so it
/// cannot distinguish a direction-aware implementation from a direction-blind
/// one. The field must vary with position for the sampling geometry to matter.
fn varying_components(dims: [usize; 3]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let mut ux = Vec::with_capacity(nz * ny * nx);
    let mut uy = Vec::with_capacity(nz * ny * nx);
    let mut uz = Vec::with_capacity(nz * ny * nx);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let (fz, fy, fx) = (iz as f32, iy as f32, ix as f32);
                ux.push(0.05 * fx - 0.02 * fy);
                uy.push(0.03 * fy + 0.01 * fz);
                uz.push(0.02 * fz - 0.01 * fx);
            }
        }
    }
    (ux, uy, uz)
}

/// The inverse of the zero field is the zero field (exactly).
#[test]
fn iterative_invert_zero_field_is_zero() {
    let dims = [4, 5, 6];
    let n: usize = dims.iter().product();
    let z = || img(vec![0.0f32; n], dims);
    let (vx, vy, vz) = IterativeInverseDisplacementField::default()
        .apply(&z(), &z(), &z())
        .expect("invariant: fixture is Cartesian with an invertible direction");
    for c in [&vx, &vy, &vz] {
        let (v, _) = extract_vec_infallible(c);
        assert!(v.iter().all(|&x| x == 0.0), "zero field inverts to zero");
    }
}

/// For a spatially constant displacement `u = (a, 0, 0)`, the interior of the
/// inverse approaches `−a`.
#[test]
fn iterative_invert_constant_field_interior_is_negated() {
    let dims = [7, 7, 7];
    let n: usize = dims.iter().product();
    let a = 0.5f32;
    let dx = img(vec![a; n], dims);
    let zero = img(vec![0.0f32; n], dims);
    let (vx, _, _) = IterativeInverseDisplacementField::default()
        .apply(&dx, &zero, &zero)
        .expect("invariant: fixture is Cartesian with an invertible direction");
    let (vxv, _) = extract_vec_infallible(&vx);
    let center = (3 * 7 + 3) * 7 + 3;
    assert!(
        (vxv[center] + a).abs() < 0.2,
        "interior inverse x = {} should approach -{a}",
        vxv[center]
    );
}

/// Output geometry matches the input field.
#[test]
fn iterative_invert_preserves_geometry() {
    let dims = [3, 4, 5];
    let n: usize = dims.iter().product();
    let z = || img(vec![0.0f32; n], dims);
    let (vx, _, _) = IterativeInverseDisplacementField::default()
        .apply(&z(), &z(), &z())
        .expect("invariant: fixture is Cartesian with an invertible direction");
    assert_eq!(vx.shape(), dims);
    assert_eq!(vx.spacing()[0], 1.0);
}

/// Regression for ATLAS-RITK-TRANSFORM-DIRECTION-081.
///
/// The index/world conversions composed origin and spacing but dropped the
/// direction matrix, so the search ran in the index frame regardless of how the
/// volume was actually oriented.
///
/// The oracle is rigid-motion equivariance, which is independent of this
/// filter's own arithmetic: inverting a displacement field and then rotating
/// the result must equal rotating the field first and inverting on the rotated
/// grid. Concretely, for a rotation `R`, a grid with direction `R·A` carrying
/// components `R·u` must yield exactly `R·v`, where `v` is the inverse computed
/// on the grid with direction `A` carrying `u`. Both runs execute an identical
/// iteration sequence — same step schedule, same interpolation weights, same
/// number of iterations — so the identity is exact to floating point rather
/// than approximate, and no convergence tolerance enters the assertion.
///
/// A direction-blind implementation returns the *same* `v` for both grids,
/// which is `R·v` only where `R` is the identity. With the 3-4-5 rotation used
/// here the discrepancy is a large fraction of the field magnitude.
#[test]
fn iterative_inverse_is_equivariant_under_grid_rotation() {
    let dims = [3, 3, 3];
    let (ux, uy, uz) = varying_components(dims);
    let rotation = oblique_rotation();

    // Reference run on the axial grid.
    let reference = IterativeInverseDisplacementField::default()
        .apply(
            &img_with_direction(ux.clone(), dims, axial()),
            &img_with_direction(uy.clone(), dims, axial()),
            &img_with_direction(uz.clone(), dims, axial()),
        )
        .expect("invariant: axial fixture is Cartesian and invertible");

    // Rotate the components pointwise: u' = R u.
    let rotate = |a: &[f32], b: &[f32], c: &[f32], row: usize| -> Vec<f32> {
        (0..a.len())
            .map(|i| {
                (rotation[(row, 0)] * f64::from(a[i])
                    + rotation[(row, 1)] * f64::from(b[i])
                    + rotation[(row, 2)] * f64::from(c[i])) as f32
            })
            .collect()
    };
    let rotated_direction = rotation * axial();
    let rotated = IterativeInverseDisplacementField::default()
        .apply(
            &img_with_direction(rotate(&ux, &uy, &uz, 0), dims, rotated_direction),
            &img_with_direction(rotate(&ux, &uy, &uz, 1), dims, rotated_direction),
            &img_with_direction(rotate(&ux, &uy, &uz, 2), dims, rotated_direction),
        )
        .expect("invariant: rotated fixture is Cartesian and invertible");

    let (rx, _) = extract_vec_infallible(&reference.0);
    let (ry, _) = extract_vec_infallible(&reference.1);
    let (rz, _) = extract_vec_infallible(&reference.2);
    let (gx, _) = extract_vec_infallible(&rotated.0);
    let (gy, _) = extract_vec_infallible(&rotated.1);
    let (gz, _) = extract_vec_infallible(&rotated.2);

    let mut largest_reference = 0.0_f64;
    for i in 0..rx.len() {
        let reference_vector = [f64::from(rx[i]), f64::from(ry[i]), f64::from(rz[i])];
        largest_reference = largest_reference.max(
            reference_vector
                .iter()
                .fold(0.0_f64, |acc, v| acc.max(v.abs())),
        );
        let got = [f64::from(gx[i]), f64::from(gy[i]), f64::from(gz[i])];
        for row in 0..3 {
            let want: f64 = (0..3)
                .map(|column| rotation[(row, column)] * reference_vector[column])
                .sum();
            // f32 storage of the components bounds the achievable agreement at
            // a few ulp of the field magnitude, not at f64 epsilon.
            assert!(
                (got[row] - want).abs() < 1e-5,
                "voxel {i} row {row}: rotated run gave {}, rotation of the \
                 reference is {want}",
                got[row]
            );
        }
    }
    assert!(
        largest_reference > 0.05,
        "fixture must produce a non-trivial inverse field, largest component \
         was {largest_reference}"
    );
}
