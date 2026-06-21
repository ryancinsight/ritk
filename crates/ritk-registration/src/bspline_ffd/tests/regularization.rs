use super::super::regularization::bending_energy;
use crate::deformable_field_ops::flat;

#[test]
fn bending_energy_of_zero_field_is_zero() {
    let ctrl_dims = [6, 6, 6];
    let ctrl_spacing = [4.0, 4.0, 4.0];
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

    let cp_z = vec![0.0_f32; cn];
    let cp_y = vec![0.0_f32; cn];
    let cp_x = vec![0.0_f32; cn];

    let be = bending_energy(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);
    assert!(
        be.abs() < 1e-12,
        "bending energy of zero field should be 0, got {}",
        be
    );
}

#[test]
fn bending_energy_of_constant_field_is_zero() {
    let ctrl_dims = [6, 6, 6];
    let ctrl_spacing = [4.0, 4.0, 4.0];
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

    let cp_z = vec![3.0_f32; cn];
    let cp_y = vec![-1.5_f32; cn];
    let cp_x = vec![2.0_f32; cn];

    let be = bending_energy(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);
    assert!(
        be.abs() < 1e-10,
        "bending energy of constant field should be ~0, got {}",
        be
    );
}

#[test]
fn bending_energy_positive_for_nonlinear_field() {
    let ctrl_dims = [6, 6, 6];
    let ctrl_spacing = [4.0, 4.0, 4.0];
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
    let [cnz, cny, cnx] = ctrl_dims;

    let mut cp_x = vec![0.0_f32; cn];
    let cp_y = vec![0.0_f32; cn];
    let cp_z = vec![0.0_f32; cn];

    for iz in 0..cnz {
        for iy in 0..cny {
            for ix in 0..cnx {
                cp_x[flat(iz, iy, ix, cny, cnx)] = (ix as f32) * (ix as f32);
            }
        }
    }

    let be = bending_energy(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);
    assert!(
        be > 0.0,
        "bending energy of quadratic field should be > 0, got {}",
        be
    );
}

#[test]
fn bending_energy_gradient_of_zero_field_is_zero() {
    use super::super::regularization::bending_energy_gradient;
    let ctrl_dims = [6, 6, 6];
    let ctrl_spacing = [4.0, 4.0, 4.0];
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

    let cp_z = vec![0.0_f32; cn];
    let cp_y = vec![0.0_f32; cn];
    let cp_x = vec![0.0_f32; cn];

    let be_grad = bending_energy_gradient(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);
    for i in 0..cn {
        assert!(be_grad.z[i].abs() < 1e-12);
        assert!(be_grad.y[i].abs() < 1e-12);
        assert!(be_grad.x[i].abs() < 1e-12);
    }
}
