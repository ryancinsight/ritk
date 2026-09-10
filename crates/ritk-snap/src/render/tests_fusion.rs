use super::*;
use crate::render::SliceRenderer;
use egui::Color32;
use iris::color::{ColorMap, Normalized};
use ritk_io::{literal_arraystring, DicomReadMetadata};

fn rgb8(map: NamedColorMap, value: f32) -> [u8; 3] {
    let value = Normalized::new(value).expect("test value is normalized");
    let [red, green, blue, _] = map.sample(value).to_rgba8();
    [red, green, blue]
}

fn test_volume(shape: [usize; 3], scale: f32) -> LoadedVolume {
    let [d, r, c] = shape;
    let mut data = Vec::with_capacity(d * r * c);
    for z in 0..d {
        for y in 0..r {
            for x in 0..c {
                data.push(scale * (z * r * c + y * c + x) as f32);
            }
        }
    }
    LoadedVolume {
        data: std::sync::Arc::new(data),
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
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        series_time: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    }
}

fn with_frame(mut volume: LoadedVolume, uid: &str) -> LoadedVolume {
    let mut metadata = DicomReadMetadata::default();
    metadata.frame_of_reference_uid = Some(
        uid.try_into()
            .expect("test frame identifier fits the DICOM UID field"),
    );
    volume.metadata = Some(Box::new(metadata));
    volume
}

#[test]
fn alpha_zero_equals_primary_render() {
    let p = test_volume([2, 3, 4], 1.0);
    let s = test_volume([2, 3, 4], 2.0);
    let wl = WindowLevel::new(128.0, 256.0);
    let fused = render_fused_slice(
        FusedSliceParams {
            volume: &p,
            axis: 0,
            slice: 1,
            wl,
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &s,
            axis: 0,
            slice: 1,
            wl,
            colormap: NamedColorMap::Hot,
        },
        0.0,
    )
    .expect("identical grids form a valid fusion");
    let primary = SliceRenderer::render(&p, 0, 1, wl, NamedColorMap::Grayscale);
    assert_eq!(fused.size, primary.size);
    assert_eq!(fused.pixels, primary.pixels);
}

#[test]
fn output_size_matches_primary_slice_geometry() {
    let p = test_volume([5, 7, 9], 1.0);
    let s = test_volume([3, 4, 6], 1.0);
    let wl = WindowLevel::new(64.0, 128.0);
    let fused = render_fused_slice(
        FusedSliceParams {
            volume: &p,
            axis: 1,
            slice: 2,
            wl,
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &s,
            axis: 1,
            slice: 2,
            wl,
            colormap: NamedColorMap::Jet,
        },
        0.5,
    )
    .expect("parallel planes form a valid fusion");
    // Axis 1 slice is [depth, cols] => [5, 9] in [rows, cols], egui [width, height].
    assert_eq!(fused.size, [9, 5]);
}

#[test]
fn pet_secondary_is_windowed_in_suv_units() {
    let primary = test_volume([1, 1, 1], 40.0);
    let mut pet = test_volume([1, 1, 1], 0.0);
    let injected_dose_bq = 370_000_000.0;
    let patient_weight_kg = 70.0;
    pet.data = std::sync::Arc::new(vec![
        (injected_dose_bq / (patient_weight_kg * 1_000.0)) as f32,
    ]);
    pet.modality = Some(literal_arraystring::<16>("PT"));
    pet.patient_weight_kg = Some(patient_weight_kg);
    pet.injected_dose_bq = Some(injected_dose_bq);
    pet.radionuclide_half_life_s = Some(6_586.2);
    pet.decay_correction = Some(literal_arraystring::<16>("START"));

    let fused = render_fused_slice(
        FusedSliceParams {
            volume: &primary,
            axis: 0,
            slice: 0,
            wl: WindowLevel::new(40.0, 400.0),
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &pet,
            axis: 0,
            slice: 0,
            wl: WindowLevel::new(3.0, 6.0),
            colormap: NamedColorMap::Hot,
        },
        1.0,
    )
    .expect("identical grids form a valid fusion");

    let expected_wl = WindowLevel::new(3.0, 6.0).apply(1.0);
    let [r, g, b] = rgb8(NamedColorMap::Hot, f32::from(expected_wl) / 255.0);
    assert_eq!(fused.size, [1, 1]);
    assert_eq!(
        fused.pixels[0],
        Color32::from_rgb(r, g, b),
        "PET fusion must apply SUVbw before the SUV window"
    );
}

#[test]
fn non_pet_secondary_with_pet_fields_uses_raw_window_units() {
    let primary = test_volume([1, 1, 1], 40.0);
    let mut secondary = test_volume([1, 1, 1], 0.0);
    let injected_dose_bq = 370_000_000.0;
    let patient_weight_kg = 70.0;
    let raw_bqml = (injected_dose_bq / (patient_weight_kg * 1_000.0)) as f32;
    secondary.data = std::sync::Arc::new(vec![raw_bqml]);
    secondary.modality = Some(literal_arraystring::<16>("CT"));
    secondary.patient_weight_kg = Some(patient_weight_kg);
    secondary.injected_dose_bq = Some(injected_dose_bq);
    secondary.radionuclide_half_life_s = Some(6_586.2);
    secondary.decay_correction = Some(literal_arraystring::<16>("START"));

    let fused = render_fused_slice(
        FusedSliceParams {
            volume: &primary,
            axis: 0,
            slice: 0,
            wl: WindowLevel::new(40.0, 400.0),
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &secondary,
            axis: 0,
            slice: 0,
            wl: WindowLevel::new(3.0, 6.0),
            colormap: NamedColorMap::Hot,
        },
        1.0,
    )
    .expect("identical grids form a valid fusion");

    let expected_wl = WindowLevel::new(3.0, 6.0).apply(f64::from(raw_bqml));
    let [r, g, b] = rgb8(NamedColorMap::Hot, f32::from(expected_wl) / 255.0);
    assert_eq!(
        fused.pixels[0],
        Color32::from_rgb(r, g, b),
        "non-PT fusion inputs must not apply SUV conversion"
    );
}

#[test]
fn rotated_anisotropic_grids_sample_by_patient_landmarks() {
    let mut primary = test_volume([1, 3, 2], 0.0);
    primary.spacing = [1.0, 2.0, 2.0];

    let mut secondary = test_volume([1, 5, 5], 0.0);
    secondary.origin = [0.0, 4.0, 0.0];
    secondary.direction = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
    secondary.data = std::sync::Arc::new(
        (0..5)
            .flat_map(|row| (0..5).map(move |col| (row * 10 + col) as f32))
            .collect(),
    );
    let primary = with_frame(primary, "1.2.3");
    let secondary = with_frame(secondary, "1.2.3");
    let wl = WindowLevel::new(20.0, 40.0);
    let image = render_fused_slice(
        FusedSliceParams {
            volume: &primary,
            axis: 0,
            slice: 0,
            wl,
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &secondary,
            axis: 0,
            slice: 0,
            wl,
            colormap: NamedColorMap::Grayscale,
        },
        1.0,
    )
    .expect("parallel grids with one frame should fuse");

    let expected_values = [4.0, 24.0, 2.0, 22.0, 0.0, 20.0];
    for (pixel, value) in image.pixels.iter().zip(expected_values) {
        let [red, green, blue] = rgb8(NamedColorMap::Grayscale, f32::from(wl.apply(value)) / 255.0);
        assert_eq!(*pixel, Color32::from_rgb(red, green, blue));
    }
}

#[test]
fn frame_mismatch_is_rejected_before_sampling() {
    let primary = with_frame(test_volume([1, 1, 1], 1.0), "1.2.3");
    let secondary = with_frame(test_volume([1, 1, 1], 2.0), "1.2.4");
    let result = render_fused_slice(
        FusedSliceParams {
            volume: &primary,
            axis: 0,
            slice: 0,
            wl: WindowLevel::new(0.0, 1.0),
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &secondary,
            axis: 0,
            slice: 0,
            wl: WindowLevel::new(0.0, 1.0),
            colormap: NamedColorMap::Grayscale,
        },
        1.0,
    );
    assert_eq!(result, Err(FusionError::IncompatibleFrameOfReference));
}

#[test]
fn differing_unknown_frames_are_rejected() {
    let primary = test_volume([1, 1, 1], 1.0);
    let mut secondary = test_volume([1, 1, 1], 1.0);
    secondary.origin = [1.0, 0.0, 0.0];
    let result = render_fused_slice(
        FusedSliceParams {
            volume: &primary,
            axis: 0,
            slice: 0,
            wl: WindowLevel::new(0.0, 1.0),
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &secondary,
            axis: 0,
            slice: 0,
            wl: WindowLevel::new(0.0, 1.0),
            colormap: NamedColorMap::Grayscale,
        },
        1.0,
    );
    assert_eq!(result, Err(FusionError::MissingFrameOfReference));
}

#[test]
fn out_of_field_in_plane_pixels_keep_primary_values() {
    let mut primary = test_volume([1, 1, 2], 0.0);
    primary.data = std::sync::Arc::new(vec![10.0, 20.0]);
    let mut secondary = test_volume([1, 1, 2], 0.0);
    secondary.origin = [0.0, 100.0, 0.0];
    let primary = with_frame(primary, "1.2.3");
    let secondary = with_frame(secondary, "1.2.3");
    let wl = WindowLevel::new(0.0, 40.0);
    let fused = render_fused_slice(
        FusedSliceParams {
            volume: &primary,
            axis: 0,
            slice: 0,
            wl,
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &secondary,
            axis: 0,
            slice: 0,
            wl,
            colormap: NamedColorMap::Grayscale,
        },
        1.0,
    )
    .expect("in-plane field gaps are a valid partial fusion");
    let primary_image = SliceRenderer::render(&primary, 0, 0, wl, NamedColorMap::Grayscale);
    assert_eq!(fused.pixels, primary_image.pixels);
}

#[test]
fn non_parallel_planes_are_rejected() {
    let primary = test_volume([2, 2, 2], 1.0);
    let secondary = test_volume([2, 2, 2], 1.0);
    let result = render_fused_slice(
        FusedSliceParams {
            volume: &primary,
            axis: 0,
            slice: 0,
            wl: WindowLevel::new(0.0, 1.0),
            colormap: NamedColorMap::Grayscale,
        },
        FusedSliceParams {
            volume: &secondary,
            axis: 1,
            slice: 0,
            wl: WindowLevel::new(0.0, 1.0),
            colormap: NamedColorMap::Grayscale,
        },
        1.0,
    );
    assert_eq!(result, Err(FusionError::NonParallelPlanes));
}

#[test]
fn secondary_slice_selection_uses_patient_coordinate() {
    let mut primary = test_volume([3, 1, 1], 1.0);
    primary.spacing = [2.0, 1.0, 1.0];
    let mut secondary = test_volume([5, 1, 1], 1.0);
    secondary.origin = [-1.0, 0.0, 0.0];
    let primary = with_frame(primary, "1.2.3");
    let secondary = with_frame(secondary, "1.2.3");
    assert_eq!(
        secondary_slice_for_primary(&primary, 0, 1, &secondary, 0)
            .expect("physical slice should be in the secondary extent"),
        3
    );
}

#[test]
fn normal_out_of_field_is_reported() {
    let primary = with_frame(test_volume([1, 1, 1], 1.0), "1.2.3");
    let mut secondary = test_volume([1, 1, 1], 1.0);
    secondary.origin = [100.0, 0.0, 0.0];
    let secondary = with_frame(secondary, "1.2.3");
    let result = secondary_slice_for_primary(&primary, 0, 0, &secondary, 0);
    assert_eq!(result, Err(FusionError::NoPhysicalOverlap));
}
