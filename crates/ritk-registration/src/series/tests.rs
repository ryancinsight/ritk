//! Series alignment contracts.
//!
//! The registration fit itself is covered by the classical engine's own tests.
//! What matters here is the driver's contract: the reference is exact rather
//! than fitted, every volume gets an entry in acquisition order, the reported
//! rotation is the polar factor rather than the raw linear part, and a fit that
//! cannot yield a rotation is surfaced instead of passed on.

use leto::Array3;

use super::*;
use crate::classical::engine::ImageRegistration;

/// A small volume with a distinguishable intensity pattern.
fn volume(seed: f64) -> Array3<f64> {
    let dims = [4, 4, 4];
    let values: Vec<f64> = (0..64)
        .map(|index| seed + (index % 7) as f64 + ((index / 7) % 5) as f64 * 0.5)
        .collect();
    Array3::from_shape_vec(dims, values).expect("valid volume shape")
}

fn engine() -> ImageRegistration {
    ImageRegistration::new()
}

/// Build a row-major 4×4 affine from a 3×3 linear part.
fn affine_from_linear(linear: [[f64; 3]; 3]) -> AffineTransform {
    AffineTransform::new([
        linear[0][0],
        linear[0][1],
        linear[0][2],
        0.0, //
        linear[1][0],
        linear[1][1],
        linear[1][2],
        0.0, //
        linear[2][0],
        linear[2][1],
        linear[2][2],
        0.0, //
        0.0,
        0.0,
        0.0,
        1.0,
    ])
}

#[test]
fn every_volume_receives_an_entry_in_acquisition_order() {
    let volumes: Vec<Array3<f64>> = (0..5).map(|index| volume(index as f64)).collect();

    let alignment = register_series(&volumes, &engine(), &SeriesRegistrationConfig::default())
        .expect("a well-formed series aligns");

    assert_eq!(alignment.volumes.len(), 5, "one entry per volume");
    for (position, entry) in alignment.volumes.iter().enumerate() {
        assert_eq!(
            entry.index, position,
            "entries must stay in acquisition order"
        );
    }
}

#[test]
fn the_reference_is_exact_rather_than_fitted() {
    // Registering the reference to itself would return a near-identity fit
    // perturbed by optimizer noise, injecting a spurious rotation into the one
    // volume known to need none.
    let volumes: Vec<Array3<f64>> = (0..3).map(|index| volume(index as f64)).collect();

    let alignment =
        register_series(&volumes, &engine(), &SeriesRegistrationConfig::default()).expect("aligns");

    let reference = &alignment.volumes[alignment.reference_index];
    assert_eq!(
        reference.transform.as_array(),
        AffineTransform::IDENTITY.as_array(),
        "the reference transform must be exactly the identity"
    );
    assert_eq!(
        reference.rotation, IDENTITY_ROTATION,
        "the reference rotation must be exactly the identity"
    );
}

#[test]
fn an_explicit_reference_index_is_honoured() {
    let volumes: Vec<Array3<f64>> = (0..4).map(|index| volume(index as f64)).collect();
    let config = SeriesRegistrationConfig {
        reference: ReferenceVolume::Index(2),
        ..SeriesRegistrationConfig::default()
    };

    let alignment = register_series(&volumes, &engine(), &config).expect("aligns");

    assert_eq!(alignment.reference_index, 2);
    assert_eq!(
        alignment.volumes[2].rotation, IDENTITY_ROTATION,
        "the nominated volume is the one held fixed"
    );
}

#[test]
fn an_out_of_range_reference_is_rejected() {
    // Silently clamping to the last volume would align the series to a
    // different target than the caller asked for.
    let volumes: Vec<Array3<f64>> = (0..3).map(|index| volume(index as f64)).collect();
    let config = SeriesRegistrationConfig {
        reference: ReferenceVolume::Index(7),
        ..SeriesRegistrationConfig::default()
    };

    let error = register_series(&volumes, &engine(), &config)
        .expect_err("index 7 is outside a 3-volume series");
    assert!(
        format!("{error}").contains("outside a series of 3"),
        "error must name the series length, got {error}"
    );
}

#[test]
fn an_empty_series_is_rejected() {
    let error = register_series(&[], &engine(), &SeriesRegistrationConfig::default())
        .expect_err("an empty series has no reference");
    assert!(format!("{error}").contains("empty series"));
}

#[test]
fn rotations_are_returned_in_acquisition_order() {
    // The rotations list feeds per-volume reorientation directly, so its order
    // and length must match the series exactly.
    let volumes: Vec<Array3<f64>> = (0..4).map(|index| volume(index as f64)).collect();

    let alignment =
        register_series(&volumes, &engine(), &SeriesRegistrationConfig::default()).expect("aligns");
    let rotations = alignment.rotations();

    assert_eq!(rotations.len(), volumes.len());
    for (position, rotation) in rotations.iter().enumerate() {
        assert_eq!(
            *rotation, alignment.volumes[position].rotation,
            "rotation {position} must match its volume entry"
        );
    }
}

#[test]
fn every_reported_rotation_is_a_proper_rotation() {
    // Downstream reorientation validates orthonormality at 1e-9 and rejects
    // anything looser, so a rotation that failed this would be unusable exactly
    // where it is needed.
    let volumes: Vec<Array3<f64>> = (0..4).map(|index| volume(index as f64)).collect();

    let alignment =
        register_series(&volumes, &engine(), &SeriesRegistrationConfig::default()).expect("aligns");

    for entry in &alignment.volumes {
        let rotation = entry.rotation;
        for row in 0..3 {
            for column in 0..3 {
                let dot: f64 = (0..3)
                    .map(|axis| rotation[axis][row] * rotation[axis][column])
                    .sum();
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-9,
                    "volume {} RᵀR[{row}][{column}] is {dot}, expected {expected}",
                    entry.index
                );
            }
        }
    }
}

#[test]
fn rotation_of_strips_scale_from_the_linear_part() {
    // The contract that keeps a caller from using the raw upper-left 3×3: a
    // quarter turn scaled by 2 must report the quarter turn alone.
    let transform = affine_from_linear([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]);

    let rotation = rotation_of(&transform).expect("invertible");

    assert!(
        (rotation[0][1] + 1.0).abs() < 1e-12,
        "scale removed from -1"
    );
    assert!(
        (rotation[1][0] - 1.0).abs() < 1e-12,
        "scale removed from +1"
    );
    assert!((rotation[2][2] - 1.0).abs() < 1e-12, "z axis normalized");
}

#[test]
fn rotation_of_ignores_the_translation_column() {
    // Translation does not reorient anything, so it must not leak into the
    // rotation. Two transforms differing only in translation must agree.
    let linear = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let mut translated = affine_from_linear(linear);
    translated.as_array_mut()[3] = 17.5;
    translated.as_array_mut()[7] = -4.25;
    translated.as_array_mut()[11] = 3.0;

    assert_eq!(
        rotation_of(&translated).expect("invertible"),
        rotation_of(&affine_from_linear(linear)).expect("invertible"),
        "translation must not change the extracted rotation"
    );
}

#[test]
fn rotation_of_rejects_a_reflected_transform() {
    // A registration between two images of one subject cannot reverse
    // handedness; reporting a repaired rotation would hide the failed fit.
    let reflected = affine_from_linear([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    assert!(matches!(
        rotation_of(&reflected),
        Err(RotationExtractionError::OrientationReversing { .. })
    ));
}

#[test]
fn rotation_of_rejects_a_collapsed_transform() {
    let collapsed = affine_from_linear([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);

    assert!(matches!(
        rotation_of(&collapsed),
        Err(RotationExtractionError::RankDeficient { .. })
    ));
}

#[test]
fn a_single_volume_series_is_its_own_reference() {
    // Degenerate but legal: one volume, nothing to align it to.
    let alignment = register_series(
        &[volume(1.0)],
        &engine(),
        &SeriesRegistrationConfig::default(),
    )
    .expect("a one-volume series aligns trivially");

    assert_eq!(alignment.volumes.len(), 1);
    assert_eq!(alignment.reference_index, 0);
    assert_eq!(alignment.volumes[0].rotation, IDENTITY_ROTATION);
}

#[test]
fn the_default_model_is_rigid() {
    // Rigid cannot deform anatomy. A caller that has not considered eddy
    // currents must not silently receive a shape-changing fit.
    assert_eq!(
        SeriesRegistrationConfig::default().model,
        SeriesTransformModel::Rigid
    );
}
