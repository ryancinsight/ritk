use super::*;
use crate::render::SliceRenderer;
use egui::Color32;
use iris::color::{ColorMap, Normalized};
use ritk_io::literal_arraystring;

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
    );
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
            axis: 2,
            slice: 3,
            wl,
            colormap: NamedColorMap::Jet,
        },
        0.5,
    );
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
    );

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
    );

    let expected_wl = WindowLevel::new(3.0, 6.0).apply(f64::from(raw_bqml));
    let [r, g, b] = rgb8(NamedColorMap::Hot, f32::from(expected_wl) / 255.0);
    assert_eq!(
        fused.pixels[0],
        Color32::from_rgb(r, g, b),
        "non-PT fusion inputs must not apply SUV conversion"
    );
}
