//! Viewport unit tests.

use super::state::{img_to_screen, img_to_volume, screen_to_img, screen_to_img_f32, slice_dims};
use super::*;
use crate::render::slice_render::WindowLevel;
use crate::LoadedVolume;
use egui::{pos2, vec2, Rect, Vec2};

fn make_volume(depth: usize, rows: usize, cols: usize) -> LoadedVolume {
    let n = depth * rows * cols;
    LoadedVolume {
        data: std::sync::Arc::new((0..n).map(|i| i as f32).collect()),
        shape: [depth, rows, cols],
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

// ── image_transform ───────────────────────────────────────────────────────

/// For zoom=1.0 and a square image fitting exactly in the viewport,
/// `scale` must equal `vp_size / img_size`.
///
/// Analytical: viewport 100×100, image 50×50 → base_scale = 2.0,
/// zoom = 1.0 → scale = 2.0.
#[test]
fn test_image_transform_scale_fit_square() {
    let state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
    let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
    let (_, scale) = state.image_transform(rect, 50, 50);
    assert!(
        (scale - 2.0).abs() < 1e-4,
        "scale must be 2.0 for 100×100 viewport and 50×50 image, got {scale}"
    );
}

/// For a landscape viewport (200×100) and a square image (50×50),
/// the constraining dimension is the height, so `scale = 100/50 = 2.0`.
#[test]
fn test_image_transform_scale_fit_landscape_viewport() {
    let state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
    let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(200.0, 100.0));
    let (_, scale) = state.image_transform(rect, 50, 50);
    // min(200/50, 100/50) = min(4.0, 2.0) = 2.0
    assert!(
        (scale - 2.0).abs() < 1e-4,
        "scale must be 2.0 (height-constrained), got {scale}"
    );
}

/// Zoom doubles the scale.
#[test]
fn test_image_transform_zoom_doubles_scale() {
    let mut state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
    state.zoom = 2.0;
    let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
    let (_, scale) = state.image_transform(rect, 50, 50);
    // base=2.0, zoom=2.0 → scale=4.0
    assert!(
        (scale - 4.0).abs() < 1e-4,
        "zoom=2.0 must double scale to 4.0, got {scale}"
    );
}

/// For a zero-size image `image_transform` must return scale=1.0 without
/// panic (defensive path).
#[test]
fn test_image_transform_zero_image_no_panic() {
    let state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
    let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 100.0));
    let (_, scale) = state.image_transform(rect, 0, 0);
    assert_eq!(scale, 1.0, "zero-size image must return scale=1.0");
}

// ── slice_dims ────────────────────────────────────────────────────────────

/// Axial slice of [D=4, R=5, C=6] must have (width, height) = (6, 5).
#[test]
fn test_slice_dims_axial() {
    let vol = make_volume(4, 5, 6);
    let (w, h) = slice_dims(&vol, 0);
    assert_eq!(w, 6, "axial width must equal cols=6");
    assert_eq!(h, 5, "axial height must equal rows=5");
}

/// Coronal slice of [D=4, R=5, C=6] must have (width, height) = (6, 4).
#[test]
fn test_slice_dims_coronal() {
    let vol = make_volume(4, 5, 6);
    let (w, h) = slice_dims(&vol, 1);
    assert_eq!(w, 6, "coronal width must equal cols=6");
    assert_eq!(h, 4, "coronal height must equal depth=4");
}

/// Sagittal slice of [D=4, R=5, C=6] must have (width, height) = (5, 4).
#[test]
fn test_slice_dims_sagittal() {
    let vol = make_volume(4, 5, 6);
    let (w, h) = slice_dims(&vol, 2);
    assert_eq!(w, 5, "sagittal width must equal rows=5");
    assert_eq!(h, 4, "sagittal height must equal depth=4");
}

// ── img_to_volume ─────────────────────────────────────────────────────────

/// Axial axis: img_col → volume col, img_row → volume row,
/// slice → volume depth.
#[test]
fn test_img_to_volume_axial() {
    let vox = img_to_volume(3, 7, 10, 0);
    assert_eq!(vox, [10, 7, 3], "axial: [slice, img_row, img_col]");
}

/// Coronal axis: img_col → volume col, img_row → volume depth,
/// slice → volume row.
#[test]
fn test_img_to_volume_coronal() {
    let vox = img_to_volume(3, 7, 10, 1);
    assert_eq!(vox, [7, 10, 3], "coronal: [img_row, slice, img_col]");
}

/// Sagittal axis: img_col → volume row, img_row → volume depth,
/// slice → volume col.
#[test]
fn test_img_to_volume_sagittal() {
    let vox = img_to_volume(3, 7, 10, 2);
    assert_eq!(vox, [7, 3, 10], "sagittal: [img_row, img_col, slice]");
}

// ── screen_to_img ─────────────────────────────────────────────────────────

/// screen_to_img must return `None` for positions outside the image
/// bounds.
#[test]
fn test_screen_to_img_out_of_bounds() {
    let offset = Vec2::new(0.0, 0.0);
    // Position at col=10, row=0 with img_w=8 is out of bounds.
    let result = screen_to_img(pos2(10.0, 0.0), offset, 1.0, 8, 8);
    assert!(
        result.is_none(),
        "position col=10 must be out of bounds for img_w=8"
    );
}

/// screen_to_img must correctly round-down to integer coordinates for
/// an in-bounds position.
///
/// Analytical: offset=(0,0), scale=2.0, screen=(7.9, 5.0)
/// → col_f = 3.95 → col = 3; row_f = 2.5 → row = 2.
#[test]
fn test_screen_to_img_in_bounds() {
    let offset = Vec2::new(0.0, 0.0);
    let result = screen_to_img(pos2(7.9, 5.0), offset, 2.0, 10, 10);
    assert!(result.is_some(), "position must be in bounds");
    let (col, row) = result.unwrap();
    assert_eq!(col, 3, "col must be floor(7.9 / 2.0) = 3");
    assert_eq!(row, 2, "row must be floor(5.0 / 2.0) = 2");
}

/// Forward + inverse affine transforms must compose to identity when
/// `scale > 0`.
#[test]
fn test_img_screen_transform_round_trip_identity() {
    let offset = Vec2::new(13.0, -7.5);
    let scale = 2.75;
    let img = pos2(19.25, 4.5);
    let screen = img_to_screen(img, offset, scale);
    let restored =
        screen_to_img_f32(screen, offset, scale).expect("inverse mapping must exist for scale > 0");
    assert!(
        (restored.0 - img.x).abs() < 1e-6,
        "round-trip col mismatch: expected {}, got {}",
        img.x,
        restored.0
    );
    assert!(
        (restored.1 - img.y).abs() < 1e-6,
        "round-trip row mismatch: expected {}, got {}",
        img.y,
        restored.1
    );
}

/// Integer `screen_to_img` must agree with `floor(screen_to_img_f32)` for
/// in-bounds points.
#[test]
fn test_screen_to_img_matches_f32_floor_for_in_bounds_points() {
    let offset = Vec2::new(5.0, 9.0);
    let scale = 1.25;
    let img_w = 32;
    let img_h = 18;
    // Choose a point that maps inside bounds with non-integer image coords.
    let screen = pos2(17.2, 23.6);
    let int_coord =
        screen_to_img(screen, offset, scale, img_w, img_h).expect("chosen point must be in bounds");
    let float_coord =
        screen_to_img_f32(screen, offset, scale).expect("scale > 0 must produce f32 coordinate");
    assert_eq!(int_coord.0, float_coord.0 as usize, "col floor mismatch");
    assert_eq!(int_coord.1, float_coord.1 as usize, "row floor mismatch");
}

/// `screen_to_img_f32` must reject non-positive scales to preserve the
/// affine inverse precondition.
#[test]
fn test_screen_to_img_f32_rejects_non_positive_scale() {
    let p = pos2(1.0, 2.0);
    let o = Vec2::new(0.0, 0.0);
    assert!(
        screen_to_img_f32(p, o, 0.0).is_none(),
        "scale=0 must reject"
    );
    assert!(
        screen_to_img_f32(p, o, -1.0).is_none(),
        "negative scale must reject"
    );
}

// ── clamp_slice_index ─────────────────────────────────────────────────────

/// `clamp_slice_index` must reduce an out-of-range index to `dim − 1`.
#[test]
fn test_clamp_slice_index() {
    let vol = make_volume(10, 20, 30);
    let mut state = ViewportState::new(0, WindowLevel::new(0.0, 100.0));
    state.slice_index = 999;
    state.clamp_slice_index(&vol);
    assert_eq!(
        state.slice_index, 9,
        "clamp must reduce 999 to depth-1=9 for axial axis"
    );
}

/// `for_mip` must initialize the viewport with axial axis and MIP mode.
#[test]
fn test_for_mip_initializes_mip_mode() {
    let state = ViewportState::for_mip(WindowLevel::new(0.0, 1.0));
    assert_eq!(state.axis, 0, "MIP viewport must use axial axis=0");
    assert_eq!(
        state.render_mode,
        ViewportRenderMode::Mip,
        "MIP viewport must default to Mip render mode"
    );
}

/// Generic constructor must default to slice rendering mode.
#[test]
fn test_new_defaults_to_slice_mode() {
    let state = ViewportState::new(1, WindowLevel::new(0.0, 1.0));
    assert_eq!(
        state.render_mode,
        ViewportRenderMode::Slice,
        "standard viewport must default to slice mode"
    );
}

// ── invalidate_texture ────────────────────────────────────────────────────

/// `invalidate_texture` must clear both `texture` and `texture_slice_key`.
#[test]
fn test_invalidate_texture_clears_key() {
    let mut state = ViewportState::new(0, WindowLevel::new(0.0, 1.0));
    state.texture_slice_key = Some((0, 5));
    state.invalidate_texture();
    assert!(
        state.texture.is_none(),
        "texture must be None after invalidation"
    );
    assert!(
        state.texture_slice_key.is_none(),
        "texture_slice_key must be None after invalidation"
    );
}
