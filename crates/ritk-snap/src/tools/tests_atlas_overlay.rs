use super::*;
use crate::LoadedVolume;
use ritk_io::{literal_arraystring, DicomReadMetadata};
use std::sync::Arc;

fn blob_volume(shape: [usize; 3], center: [f64; 3], radius: f64) -> LoadedVolume {
    let mut data = Vec::with_capacity(shape[0] * shape[1] * shape[2]);
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let dx = x as f64 - center[0];
                let dy = y as f64 - center[1];
                let dz = z as f64 - center[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let blob = (-r2 / (radius * radius)).exp();
                let ripple =
                    0.05 * (x as f64 * 0.7).sin() * (y as f64 * 1.1).sin() * (z as f64 * 0.9).sin();
                // Scale to [0, 255]: the default MI metric bins that range,
                // and [0, 1] data would collapse into a single bin.
                data.push(((blob + ripple) * 255.0) as f32);
            }
        }
    }
    LoadedVolume {
        data: Arc::new(data),
        shape,
        channels: 1,
        spacing: [1.0, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metadata: None,
        source: None,
        modality: None,
        patient_name: None,
        patient_id: None,
        study_date: None,
        series_description: None,
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    }
}

fn shift_x(volume: &LoadedVolume, shift: isize) -> LoadedVolume {
    let [d, r, c] = volume.shape;
    let mut data = vec![0.0_f32; d * r * c];
    for z in 0..d {
        for y in 0..r {
            for x in 0..c {
                let sx = x as isize - shift;
                if sx >= 0 && sx < c as isize {
                    data[(z * r + y) * c + x] = volume.data[(z * r + y) * c + sx as usize];
                }
            }
        }
    }
    LoadedVolume {
        data: Arc::new(data),
        ..clone_volume(volume)
    }
}

fn clone_volume(volume: &LoadedVolume) -> LoadedVolume {
    LoadedVolume {
        data: volume.data.clone(),
        shape: volume.shape,
        channels: volume.channels,
        spacing: volume.spacing,
        origin: volume.origin,
        direction: volume.direction,
        metadata: volume.metadata.clone(),
        source: volume.source.clone(),
        modality: volume.modality,
        patient_name: volume.patient_name.clone(),
        patient_id: volume.patient_id.clone(),
        study_date: volume.study_date,
        series_description: volume.series_description.clone(),
        series_time: volume.series_time,
        patient_weight_kg: volume.patient_weight_kg,
        injected_dose_bq: volume.injected_dose_bq,
        radionuclide_half_life_s: volume.radionuclide_half_life_s,
        radiopharmaceutical_start_time: volume.radiopharmaceutical_start_time,
        decay_correction: volume.decay_correction,
    }
}

/// Identical volumes: MI must not decrease and geometry must be preserved.
#[test]
fn identical_volumes_keep_mi_and_geometry() {
    let primary = blob_volume([32, 32, 32], [16.0, 16.0, 16.0], 6.0);
    let outcome = register_secondary_to_primary(&primary, &primary, 4)
        .expect("identical-volume registration succeeds");
    assert!(
        outcome.mi_after + 1e-9 >= outcome.mi_before,
        "MI must not decrease: {} -> {}",
        outcome.mi_before,
        outcome.mi_after
    );
    assert_eq!(outcome.resampled.shape, primary.shape);
    assert_eq!(outcome.resampled.spacing, primary.spacing);
    assert_eq!(outcome.resampled.origin, primary.origin);
    assert_eq!(outcome.resampled.direction, primary.direction);
    assert!(
        outcome
            .resampled
            .metadata
            .as_deref()
            .and_then(|meta| meta.frame_of_reference_uid.as_ref())
            .is_none(),
        "no primary UID means no output UID"
    );
}

/// A 4-voxel x-shift is recovered (coarse grid is 2x, so within 1.5 is exact).
#[test]
fn shifted_volume_registers_and_improves_mi() {
    let primary = blob_volume([32, 32, 32], [16.0, 16.0, 16.0], 6.0);
    let moving = shift_x(&primary, 4);
    let outcome = register_secondary_to_primary(&primary, &moving, 12)
        .expect("shifted-volume registration succeeds");
    assert!(
        outcome.mi_after > outcome.mi_before,
        "MI must strictly improve: {} -> {}",
        outcome.mi_before,
        outcome.mi_after
    );
    // Fixed→moving voxel map: the +4 shift means T translates +4 in columns.
    let tx = outcome.transform[3];
    assert!(
        (tx - 4.0).abs() <= 1.5,
        "recovered x translation {tx} near +4 voxels"
    );
    assert_eq!(outcome.resampled.shape, primary.shape);
}

/// Multi-channel input is rejected.
#[test]
fn multichannel_input_errors() {
    let primary = blob_volume([16, 16, 16], [8.0, 8.0, 8.0], 3.0);
    let mut secondary = clone_volume(&primary);
    secondary.channels = 3;
    let result = register_secondary_to_primary(&primary, &secondary, 4);
    assert!(result.is_err(), "expected single-channel rejection");
    if let Err(error) = result {
        assert!(format!("{error}").contains("single-channel"));
    }
}

/// Zero iteration budget is rejected.
#[test]
fn zero_iterations_errors() {
    let primary = blob_volume([16, 16, 16], [8.0, 8.0, 8.0], 3.0);
    let result = register_secondary_to_primary(&primary, &primary, 0);
    assert!(result.is_err(), "expected positive-budget rejection");
    if let Err(error) = result {
        assert!(format!("{error}").contains("positive"));
    }
}

/// The primary frame-of-reference UID is inherited by the output.
#[test]
fn frame_of_reference_uid_is_inherited() {
    let mut primary = blob_volume([16, 16, 16], [8.0, 8.0, 8.0], 3.0);
    let mut metadata = Box::<DicomReadMetadata>::default();
    metadata.frame_of_reference_uid = Some(literal_arraystring::<64>("2.25.99"));
    primary.metadata = Some(metadata);
    let outcome = register_secondary_to_primary(&primary, &primary, 2)
        .expect("registration with UID succeeds");
    let uid = outcome
        .resampled
        .metadata
        .as_deref()
        .and_then(|meta| meta.frame_of_reference_uid.as_ref())
        .expect("output carries a UID");
    assert_eq!(uid.as_str(), "2.25.99");
}
