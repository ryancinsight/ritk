use super::*;

fn config() -> RigidSearchConfig {
    RigidSearchConfig::try_new(12.0, 8.0, 0.5, 0.75, 256).expect("valid search configuration")
}

#[test]
fn centroid_transform_maps_fixed_centroid_to_moving_centroid() {
    let fixed = [20.0, -10.0, 4.0];
    let moving = [-3.0, 12.0, 9.0];
    let transform = rigid_about_centroid(euler_zyx(0.2, -0.1, 0.3), fixed, moving);
    let matrix = transform.as_array();
    let mapped = [
        matrix[0] * fixed[0] + matrix[1] * fixed[1] + matrix[2] * fixed[2] + matrix[3],
        matrix[4] * fixed[0] + matrix[5] * fixed[1] + matrix[6] * fixed[2] + matrix[7],
        matrix[8] * fixed[0] + matrix[9] * fixed[1] + matrix[10] * fixed[2] + matrix[11],
    ];
    assert_eq!(mapped, moving);
}

#[test]
fn search_recovers_coupled_translation_optimum() {
    let target = [1.5, -2.0, 0.8];
    let objective = |transform: &AffineTransform| {
        let matrix = transform.as_array();
        let residual = [
            matrix[3] - target[0],
            matrix[7] - target[1],
            matrix[11] - target[2],
        ];
        Ok(-(residual[0] * residual[0]
            + residual[1] * residual[1]
            + residual[2] * residual[2]
            + 0.75 * (residual[0] + residual[1]).powi(2)
            + 0.75 * (residual[1] + residual[2]).powi(2)))
    };
    let result = search_rigid_pose([0.0; 3], [0.0; 3], config(), objective, objective)
        .expect("finite objective");
    let matrix = result.capture_transform.as_array();
    for (actual, expected) in [matrix[3], matrix[7], matrix[11]].into_iter().zip(target) {
        assert!(
            (actual - expected).abs() < 0.1,
            "got {actual}, expected {expected}"
        );
    }
}

#[test]
fn structural_search_cannot_leave_terminal_capture_cell() {
    let capture = |transform: &AffineTransform| {
        let matrix = transform.as_array();
        Ok(-(matrix[3].powi(2) + matrix[7].powi(2) + matrix[11].powi(2)))
    };
    let structural = |transform: &AffineTransform| {
        let translation = transform.as_array()[3];
        Ok(-(translation - 20.0).powi(2))
    };
    let result = search_rigid_pose([0.0; 3], [0.0; 3], config(), capture, structural)
        .expect("finite objectives");
    assert!(result.structural_transform.as_array()[3].abs() <= 0.75);
    assert!(result.structural_saturated);
}

#[test]
fn configuration_rejects_invalid_resource_bounds() {
    assert!(RigidSearchConfig::try_new(0.0, 8.0, 0.5, 0.75, 256).is_err());
    assert!(RigidSearchConfig::try_new(12.0, 8.0, 13.0, 0.75, 256).is_err());
    assert!(RigidSearchConfig::try_new(12.0, 8.0, 0.5, 0.75, 0).is_err());
}

#[test]
fn search_propagates_objective_failure() {
    let failure = |_transform: &AffineTransform| {
        Err(RegistrationError::InvalidInput(
            "fixture objective failure".to_owned(),
        ))
    };
    let result = search_rigid_pose([0.0; 3], [0.0; 3], config(), failure, failure);
    assert!(matches!(
        result,
        Err(RegistrationError::InvalidInput(message))
            if message == "fixture objective failure"
    ));
}
