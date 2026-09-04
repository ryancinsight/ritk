use super::*;

fn transformed(point: [f64; 3]) -> [f64; 3] {
    let angle = 20.0_f64.to_radians();
    let (sine, cosine) = angle.sin_cos();
    [
        cosine * point[0] - sine * point[1] + 4.0,
        sine * point[0] + cosine * point[1] - 3.0,
        point[2] + 2.0,
    ]
}

fn clean_pairs() -> Vec<RigidCorrespondence> {
    [
        [-2.0, -1.0, 0.0],
        [3.0, -1.0, 1.0],
        [-1.0, 4.0, 2.0],
        [2.0, 3.0, -2.0],
        [-3.0, 1.0, 4.0],
        [4.0, 2.0, 3.0],
    ]
    .into_iter()
    .map(|fixed| RigidCorrespondence::try_new(fixed, transformed(fixed)).expect("finite fixture"))
    .collect()
}

fn assert_maps_fixture(transform: &AffineTransform) {
    let gamma_512 = 512.0 * f64::EPSILON / (1.0 - 512.0 * f64::EPSILON);
    for pair in clean_pairs() {
        let residual = squared_residual(transform, &pair).sqrt();
        let scale = pair
            .moving_mm()
            .into_iter()
            .map(f64::abs)
            .fold(1.0, f64::max);
        assert!(
            residual <= gamma_512 * scale,
            "rigid residual {residual} exceeds derived floating-point bound"
        );
    }
}

#[test]
fn symmetric_trim_recovers_rigid_pose_with_sub_half_outliers() {
    let clean = clean_pairs();
    let mut forward = clean.clone();
    let mut reverse: Vec<_> = clean
        .iter()
        .map(|pair| {
            RigidCorrespondence::try_new(pair.moving_mm(), pair.fixed_mm()).expect("finite reverse")
        })
        .collect();
    for index in 0..4 {
        let fixed = [50.0 + index as f64, -80.0, 20.0];
        let moving = [-70.0, 40.0 + index as f64, -30.0];
        forward.push(RigidCorrespondence::try_new(fixed, moving).expect("finite outlier"));
        reverse.push(RigidCorrespondence::try_new(moving, fixed).expect("finite reverse outlier"));
    }

    let result = fit_symmetric_trimmed_rigid(&forward, &reverse).expect("robust rigid fit");

    assert_eq!(result.correspondence_count, 20);
    assert_eq!(result.inlier_count, 10);
    assert_maps_fixture(&result.transform);
}

#[test]
fn swapping_directions_returns_the_inverse_pose() {
    let forward = clean_pairs();
    let reverse: Vec<_> = forward
        .iter()
        .map(|pair| {
            RigidCorrespondence::try_new(pair.moving_mm(), pair.fixed_mm()).expect("finite reverse")
        })
        .collect();
    let direct = fit_symmetric_trimmed_rigid(&forward, &reverse).expect("direct fit");
    let swapped = fit_symmetric_trimmed_rigid(&reverse, &forward).expect("swapped fit");
    let product = multiply(direct.transform.as_array(), swapped.transform.as_array());
    let gamma_1024 = 1024.0 * f64::EPSILON / (1.0 - 1024.0 * f64::EPSILON);
    for (actual, expected) in product.into_iter().zip(AffineTransform::IDENTITY.0) {
        assert!((actual - expected).abs() <= gamma_1024);
    }
}

#[test]
fn correspondence_order_does_not_change_the_fit() {
    let forward = clean_pairs();
    let reverse: Vec<_> = forward
        .iter()
        .map(|pair| {
            RigidCorrespondence::try_new(pair.moving_mm(), pair.fixed_mm()).expect("finite reverse")
        })
        .collect();
    let expected = fit_symmetric_trimmed_rigid(&forward, &reverse).expect("ordered fit");
    let mut permuted_forward = forward;
    permuted_forward.rotate_left(2);
    let mut permuted_reverse = reverse;
    permuted_reverse.reverse();

    let actual =
        fit_symmetric_trimmed_rigid(&permuted_forward, &permuted_reverse).expect("permuted fit");

    assert_eq!(actual, expected);
}

#[test]
fn sampled_candidate_schedule_recovers_large_consensus() {
    let mut forward = Vec::new();
    let mut reverse = Vec::new();
    for index in 0_u32..30 {
        let layer = f64::from(index / 10);
        let fixed = [
            f64::from(index % 5) - 2.0,
            f64::from((index / 5) % 2) * 3.0 - 1.5,
            layer * 2.0 - 2.0,
        ];
        let moving = transformed(fixed);
        forward.push(RigidCorrespondence::try_new(fixed, moving).expect("finite fixture"));
        reverse.push(RigidCorrespondence::try_new(moving, fixed).expect("finite reverse fixture"));
    }
    for index in 0_u32..20 {
        let offset = f64::from(index);
        let fixed = [70.0 + offset, -30.0 + offset * 0.5, 40.0];
        let moving = [-50.0, 60.0 + offset, -20.0 + offset * 0.25];
        forward.push(RigidCorrespondence::try_new(fixed, moving).expect("finite outlier"));
        reverse.push(RigidCorrespondence::try_new(moving, fixed).expect("finite reverse outlier"));
    }

    let result = fit_symmetric_trimmed_rigid(&forward, &reverse).expect("sampled robust fit");

    assert_eq!(result.correspondence_count, 100);
    assert_eq!(result.inlier_count, 50);
    assert_maps_fixture(&result.transform);
}

#[test]
fn invalid_and_rank_deficient_inputs_fail_closed() {
    let nonfinite = RigidCorrespondence::try_new([f64::NAN, 0.0, 0.0], [0.0; 3])
        .expect_err("non-finite coordinates must fail");
    assert!(matches!(nonfinite, RegistrationError::InvalidInput(_)));

    let line: Vec<_> = (0..4)
        .map(|index| {
            let point = [index as f64, 0.0, 0.0];
            RigidCorrespondence::try_new(point, point).expect("finite line")
        })
        .collect();
    let error = fit_symmetric_trimmed_rigid(&line, &line)
        .expect_err("collinear correspondences cannot determine a rigid pose");
    assert!(
        matches!(error, RegistrationError::InvalidInput(message) if message == "rigid correspondences contain no non-collinear elemental subset")
    );
}

fn multiply(left: &[f64; 16], right: &[f64; 16]) -> [f64; 16] {
    std::array::from_fn(|index| {
        let row = index / 4;
        let column = index % 4;
        (0..4)
            .map(|inner| left[row * 4 + inner] * right[inner * 4 + column])
            .sum()
    })
}
