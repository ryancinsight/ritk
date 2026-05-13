use crate::deformable_field_ops::{flat, trilinear_interpolate};
use super::super::basis::init_control_grid;
use super::super::warp_image_bspline;
use super::make_test_image;

#[test]
fn zero_control_displacements_produce_identity_warp() {
    let dims = [8, 10, 12];
    let image = make_test_image(dims);
    let ctrl_spacing = [4.0, 4.0, 4.0];
    let ctrl_dims = init_control_grid(dims, &ctrl_spacing);
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

    let cp_z = vec![0.0_f32; cn];
    let cp_y = vec![0.0_f32; cn];
    let cp_x = vec![0.0_f32; cn];

    let warped =
        warp_image_bspline(&image, dims, &cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);

    for i in 0..image.len() {
        assert!(
            (warped[i] - image[i]).abs() < 1e-5,
            "identity warp mismatch at voxel {}: {} vs {}",
            i,
            warped[i],
            image[i]
        );
    }
}

#[test]
fn constant_displacement_translates_image() {
    let dims = [8, 10, 12];
    let ctrl_spacing = [4.0, 4.0, 4.0];
    let ctrl_dims = init_control_grid(dims, &ctrl_spacing);
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

    // Set all control points to constant displacement of +2 voxels in x.
    let cp_z = vec![0.0_f32; cn];
    let cp_y = vec![0.0_f32; cn];
    let cp_x = vec![2.0_f32; cn];

    // Create a simple ramp in x: I(z,y,x) = x.
    let [nz, ny, nx] = dims;
    let image: Vec<f32> = (0..nz * ny * nx).map(|fi| (fi % nx) as f32).collect();

    let warped =
        warp_image_bspline(&image, dims, &cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);

    // At interior voxels (away from boundary clamping), warped(z,y,x) ≈
    // moving(z, y, x + 2) = (x + 2). Near the right boundary, clamping
    // limits the sampled coordinate.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..(nx - 3) {
                let fi = flat(iz, iy, ix, ny, nx);
                let expected = (ix + 2) as f32;
                assert!(
                    (warped[fi] - expected).abs() < 0.5,
                    "translation mismatch at ({},{},{}): got {}, expected {}",
                    iz,
                    iy,
                    ix,
                    warped[fi],
                    expected
                );
            }
        }
    }
}

// Suppress unused import warning: trilinear_interpolate is used in integration tests
// but kept here to verify it's accessible from within bspline_ffd tests.
#[test]
fn trilinear_interpolate_accessible_from_bspline_tests() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let dims = [2, 2, 2];
    let v = trilinear_interpolate(&data, dims, 0.0, 0.0, 0.0);
    assert_eq!(v, 1.0);
}
