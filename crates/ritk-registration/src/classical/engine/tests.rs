use super::*;
use leto::{Array2, Array3};

#[test]
fn test_rigid_landmark_identity() {
    let reg = ImageRegistration::default();

    let fixed = Array2::from_vec([3, 3], vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
    let result = reg.rigid_registration_landmarks(&fixed, &fixed).unwrap();

    // Identity transform should have zero FRE
    let fre = result.quality.fre.unwrap();
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
    let fixed = Array2::from_vec([3, 3], vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]).unwrap();
    // Moving points: same points rotated 90 deg around Z-axis
    let moving = Array2::from_vec([3, 3], vec![0., 1., 0., -1., 0., 0., 0., 0., 1.]).unwrap();

    let result = reg.rigid_registration_landmarks(&fixed, &moving).unwrap();

    let fre = result.quality.fre.unwrap();
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
    let volume =
        Array3::from_vec([3, 3, 3], (0..27).map(|value| value as f64 * 8.0).collect()).unwrap();
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
        .unwrap();
    let affine = registration
        .affine_registration_mutual_info(&volume, &volume, &initial)
        .unwrap();

    assert_eq!(rigid.quality.mutual_information, expected);
    assert_eq!(affine.quality.mutual_information, expected);
    assert_ne!(expected, untransformed);
}
