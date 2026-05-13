use super::super::pyramid::refine_control_grid;

#[test]
fn refine_doubles_grid_points() {
    let ctrl_dims = [5, 6, 7];
    let ctrl_spacing = [8.0, 8.0, 8.0];
    let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];

    let cp_z = vec![0.0_f32; cn];
    let cp_y = vec![0.0_f32; cn];
    let cp_x = vec![0.0_f32; cn];

    let (_, _, _, new_dims, new_spacing) =
        refine_control_grid(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);

    // new_dims = 2 * old - 1
    assert_eq!(new_dims[0], 2 * ctrl_dims[0] - 1);
    assert_eq!(new_dims[1], 2 * ctrl_dims[1] - 1);
    assert_eq!(new_dims[2], 2 * ctrl_dims[2] - 1);

    // Spacing halved.
    for d in 0..3 {
        assert!(
            (new_spacing[d] - ctrl_spacing[d] / 2.0).abs() < 1e-12,
            "spacing mismatch at dim {}: {} vs {}",
            d,
            new_spacing[d],
            ctrl_spacing[d] / 2.0
        );
    }
}
