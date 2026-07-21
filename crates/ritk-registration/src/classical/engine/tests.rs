use super::*;
use leto::{Array2, Array3};

#[test]
fn test_rigid_landmark_identity() {
    let reg = ImageRegistration::default();

    let fixed = Array2::from_vec([3, 3], vec![0., 0., 0., 1., 0., 0., 0., 1., 0.])
        .expect("valid dimension");
    let result = reg
        .rigid_registration_landmarks(&fixed, &fixed)
        .expect("infallible: validated precondition");

    // Identity transform should have zero FRE
    let fre = result
        .quality
        .fre
        .expect("infallible: validated precondition");
    assert!(
        fre < 1e-10,
        "FRE for identity transform should be ~0, got {}",
        fre
    );
}

#[test]
fn test_rigid_landmark_known_rotation() {
    let reg = ImageRegistration::default();

    // Fixed points: unit vectors along axes
    let fixed = Array2::from_vec([3, 3], vec![1., 0., 0., 0., 1., 0., 0., 0., 1.])
        .expect("valid dimension");
    // Moving points: same points rotated 90 deg around Z-axis
    let moving = Array2::from_vec([3, 3], vec![0., 1., 0., -1., 0., 0., 0., 0., 1.])
        .expect("valid dimension");

    let result = reg
        .rigid_registration_landmarks(&fixed, &moving)
        .expect("infallible: validated precondition");

    let fre = result
        .quality
        .fre
        .expect("infallible: validated precondition");
    assert!(
        fre < 1e-6,
        "FRE for 90 deg rotation should be ~0, got {}",
        fre
    );
}

#[test]
fn test_mutual_information_identical() {
    let metric = MutualInformationMetric::default();
    let volume = Array3::from_elem([10, 10, 10], 128.0);
    let nmi = metric.compute(&volume, &volume);

    // Identical volumes have NMI = 1
    assert!(
        (nmi - 1.0).abs() < 1e-6,
        "NMI for identical volumes should be 1.0, got {}",
        nmi
    );
}

#[test]
fn test_mutual_information_different() {
    let metric = MutualInformationMetric::default();
    let vol1 = Array3::from_elem([10, 10, 10], 100.0);
    let vol2 = Array3::from_elem([10, 10, 10], 200.0);
    let nmi = metric.compute(&vol1, &vol2);

    // Different constant volumes have low NMI
    assert!(
        nmi < 1.0,
        "NMI for different constant volumes should be < 1, got {}",
        nmi
    );
}

#[test]
fn intensity_registration_reports_final_transform_metric() {
    let volume = Array3::from_vec([3, 3, 3], (0..27).map(|value| value as f64 * 8.0).collect())
        .expect("infallible: validated precondition");
    let initial = crate::types::AffineTransform::new([
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let config = ClassicalConfig {
        max_iterations: 0,
        ..ClassicalConfig::default()
    };
    let metric = MutualInformationMetric::default();
    let registration = ImageRegistration::with_config(config, metric.clone());
    let transformed = crate::classical::spatial::apply_transform(&volume, &initial);
    let expected = metric.compute(&transformed, &volume);
    let untransformed = metric.compute(&volume, &volume);

    let rigid = registration
        .rigid_registration_mutual_info(&volume, &volume, &initial)
        .expect("infallible: validated precondition");
    let affine = registration
        .affine_registration_mutual_info(&volume, &volume, &initial)
        .expect("infallible: validated precondition");

    assert_eq!(rigid.quality.mutual_information, expected);
    assert_eq!(affine.quality.mutual_information, expected);
    assert_ne!(expected, untransformed);
}

#[test]
fn translation_mutual_information_recovers_known_shift() {
    let fixed = Array3::from_vec(
        [5, 5, 5],
        (0..125)
            .map(|index| {
                let z = index / 25;
                let y = (index / 5) % 5;
                let x = index % 5;
                f64::from((z * z + 3 * y + 7 * x) as u32)
            })
            .collect(),
    )
    .expect("infallible: validated precondition");
    let generating_transform = crate::types::AffineTransform::new([
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let moving = crate::classical::spatial::apply_transform(&fixed, &generating_transform);
    let metric = MutualInformationMetric::new(16, 0.0, 60.0);
    let initial_similarity = metric.compute(&moving, &fixed);
    let registration = ImageRegistration::with_config(
        ClassicalConfig {
            max_iterations: 4,
            tolerance: 0.0,
            step_multiplier: 1.0,
        },
        metric,
    );

    let result = registration
        .translation_registration_mutual_info(
            &moving,
            &fixed,
            &crate::types::AffineTransform::IDENTITY,
        )
        .expect("infallible: validated precondition");

    assert_eq!(result.transform.0[3], -1.0);
    assert_eq!(result.transform.0[7], 0.0);
    assert_eq!(result.transform.0[11], 0.0);
    assert!(result.quality.mutual_information > initial_similarity);
}
