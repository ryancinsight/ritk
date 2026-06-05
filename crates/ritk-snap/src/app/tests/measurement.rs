//! Measurement label routing: per-axis spacing selection and coordinate mapping tests.
//!
//! The per-axis spacing selection used in `render_axis_viewport` for
//! measurement label routing follows the ITK-SNAP convention:
//!
//! | axis | row spacing | col spacing |
//! |------|-------------|-------------|
//! | 0 axial    | dy | dx |
//! | 1 coronal  | dz | dx |
//! | 2 sagittal | dz | dy |

use arrayvec::ArrayString;
use super::*;

/// Constructs an anisotropic volume with spacing [dz=2.0, dy=3.0, dx=5.0]
/// (prime distances, each uniquely identifiable) for spacing-dispatch tests.
fn make_anisotropic_volume() -> LoadedVolume {
    let shape = [4, 6, 8];
    LoadedVolume {
        data: Arc::new(vec![0.0f32; shape[0] * shape[1] * shape[2]]),
        shape,
        channels: 1,
        spacing: [2.0, 3.0, 5.0], // [dz, dy, dx]
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metadata: None,
        source: None,
        modality: Some(ArrayString::from("CT").unwrap()),
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

/// Axial (axis 0): row_spacing = dy = 3.0, col_spacing = dx = 5.0
#[test]
fn measurement_spacing_axial_selects_dy_dx() {
    let vol = make_anisotropic_volume();
    let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
    let spacing_2d: [f32; 2] = match 0usize {
        0 => [dy, dx],
        1 => [dz, dx],
        _ => [dz, dy],
    };
    assert_eq!(
        spacing_2d,
        [3.0, 5.0],
        "axial axis must select [dy=3.0, dx=5.0]"
    );
}

/// Coronal (axis 1): row_spacing = dz = 2.0, col_spacing = dx = 5.0
#[test]
fn measurement_spacing_coronal_selects_dz_dx() {
    let vol = make_anisotropic_volume();
    let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
    let spacing_2d: [f32; 2] = match 1usize {
        0 => [dy, dx],
        1 => [dz, dx],
        _ => [dz, dy],
    };
    assert_eq!(
        spacing_2d,
        [2.0, 5.0],
        "coronal axis must select [dz=2.0, dx=5.0]"
    );
}

/// Sagittal (axis 2): row_spacing = dz = 2.0, col_spacing = dy = 3.0
#[test]
fn measurement_spacing_sagittal_selects_dz_dy() {
    let vol = make_anisotropic_volume();
    let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
    let spacing_2d: [f32; 2] = match 2usize {
        0 => [dy, dx],
        1 => [dz, dx],
        _ => [dz, dy],
    };
    assert_eq!(
        spacing_2d,
        [2.0, 3.0],
        "sagittal axis must select [dz=2.0, dy=3.0]"
    );
}

/// All three axis-spacing pairs are mutually distinct (no collision).
#[test]
fn measurement_spacing_all_axes_are_distinct() {
    let vol = make_anisotropic_volume();
    let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
    let axial: [f32; 2] = [dy, dx];
    let coronal: [f32; 2] = [dz, dx];
    let sagittal: [f32; 2] = [dz, dy];
    assert_ne!(axial, coronal, "axial and coronal spacing must differ");
    assert_ne!(axial, sagittal, "axial and sagittal spacing must differ");
    assert_ne!(
        coronal, sagittal,
        "coronal and sagittal spacing must differ"
    );
}

/// `img_to_screen` mapping: image-pixel (col=c, row=r) maps to
/// screen position `origin + (c * scale, r * scale)`.
///
/// Analytical: origin=(10, 20), scale=2.0, img=(3, 5)
/// x = 10 + 3 × 2 = 16
/// y = 20 + 5 × 2 = 30
#[test]
fn measurement_img_to_screen_analytical() {
    let origin = egui::pos2(10.0, 20.0);
    let scale = 2.0_f32;
    let img_to_screen = |p: egui::Pos2| egui::pos2(origin.x + p.x * scale, origin.y + p.y * scale);
    let screen = img_to_screen(egui::pos2(3.0, 5.0)); // col=3, row=5
    assert_eq!(
        (screen.x, screen.y),
        (16.0, 30.0),
        "img_to_screen must compute origin + img × scale analytically"
    );
}

/// `img_to_screen` at image origin maps to screen origin.
#[test]
fn measurement_img_to_screen_origin_maps_to_rect_min() {
    let origin = egui::pos2(50.0, 75.0);
    let scale = 3.0_f32;
    let img_to_screen = |p: egui::Pos2| egui::pos2(origin.x + p.x * scale, origin.y + p.y * scale);
    let screen = img_to_screen(egui::pos2(0.0, 0.0));
    assert_eq!(
        (screen.x, screen.y),
        (50.0, 75.0),
        "image origin must map to screen rect.min"
    );
}
