use super::*;
use std::sync::Arc;

/// Create a test volume with known shape and pixel data.
fn test_volume(shape: [usize; 3], data: Vec<f32>) -> LoadedVolume {
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

/// In-bounds voxel lookup must return the correct value from the buffer.
///
/// Analytical: 3Ã—3Ã—3 volume (27 pixels total), uniform value 5.0.
/// Lookup [1, 1, 1] (center voxel) â†’ idx = 1 Ã— 9 + 1 Ã— 3 + 1 = 13.
/// pixels[13] = 5.0.
#[test]
fn test_intensity_at_voxel_in_bounds() {
    let vol = test_volume([3, 3, 3], vec![5.0; 27]);
    let intensity = intensity_at_voxel(&vol, [1, 1, 1]);
    assert_eq!(intensity, 5.0, "center voxel must return stored value");
}

/// Out-of-bounds depth coordinate must return 0.0.
///
/// Analytical: 2Ã—2Ã—2 volume, d=2 is out-of-bounds (shape[0]=2, valid: 0â€“1).
#[test]
fn test_intensity_at_voxel_out_of_bounds_depth() {
    let vol = test_volume([2, 2, 2], vec![42.0; 8]);
    let intensity = intensity_at_voxel(&vol, [2, 0, 0]);
    assert_eq!(intensity, 0.0, "out-of-bounds depth must return 0.0");
}

/// Out-of-bounds row coordinate must return 0.0.
#[test]
fn test_intensity_at_voxel_out_of_bounds_row() {
    let vol = test_volume([2, 2, 2], vec![42.0; 8]);
    let intensity = intensity_at_voxel(&vol, [0, 3, 0]);
    assert_eq!(intensity, 0.0, "out-of-bounds row must return 0.0");
}

/// Out-of-bounds column coordinate must return 0.0.
#[test]
fn test_intensity_at_voxel_out_of_bounds_column() {
    let vol = test_volume([2, 2, 2], vec![42.0; 8]);
    let intensity = intensity_at_voxel(&vol, [0, 0, 5]);
    assert_eq!(intensity, 0.0, "out-of-bounds column must return 0.0");
}

/// Boundary voxels (at edges) must return correct values.
///
/// Analytical: 2Ã—2Ã—2 volume with pixels [0,1,2,3,4,5,6,7] (row-major).
/// Voxel [0,0,0] â†’ idx = 0 â†’ pixels[0] = 0.0
/// Voxel [1,1,1] â†’ idx = 1Ã—4 + 1Ã—2 + 1 = 7 â†’ pixels[7] = 7.0
#[test]
fn test_intensity_at_voxel_boundary_corners() {
    let vol = test_volume([2, 2, 2], (0..8).map(|i| i as f32).collect());
    let intensity_origin = intensity_at_voxel(&vol, [0, 0, 0]);
    let intensity_far = intensity_at_voxel(&vol, [1, 1, 1]);
    assert_eq!(intensity_origin, 0.0, "origin corner must be 0.0");
    assert_eq!(intensity_far, 7.0, "far corner must be 7.0");
}
