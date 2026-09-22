//! Behavioral tests for the presentation frame boundary.

use super::*;
use arrayvec::ArrayString;
use std::sync::Arc;

fn test_volume() -> LoadedVolume {
    LoadedVolume {
        data: Arc::new(vec![0.0, 255.0]),
        shape: [1, 1, 2],
        channels: 1,
        spacing: [1.0, 1.0, 1.0],
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metadata: None,
        source: None,
        modality: Some(ArrayString::from("CT").expect("bounded modality")),
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

#[test]
fn frame_preserves_rendered_pixel_order_and_alpha() {
    let frame =
        PresentationFrame::from_rgba(2, 1, &[0, 0, 0, 0, 200, 150, 100, 255]).expect("valid frame");
    assert_eq!(frame.width(), 2);
    assert_eq!(frame.height(), 1);
    assert_eq!(frame.rgba(), &[0, 0, 0, 0, 200, 150, 100, 255]);
    assert_eq!(frame.display_spacing().values(), [1.0, 1.0]);
}

#[test]
fn frame_rejects_inconsistent_rgba_storage() {
    let error = PresentationFrame::from_rgba(2, 1, &[255; 4]).expect_err("mismatched bytes");
    assert!(error.to_string().contains("byte count"));
}

#[test]
fn frame_rejects_zero_dimensions() {
    let error = PresentationFrame::from_rgba(0, 1, &[]).expect_err("zero width");
    assert!(error.to_string().contains("dimensions must be nonzero"));
}

#[test]
fn frame_rejects_host_oversized_storage_before_copying_pixels() {
    let width = u32::try_from(MAX_PIXELS + 1).expect("test width fits u32");
    let error = PresentationFrame::from_rgba(width, 1, &[]).expect_err("oversized frame");
    assert!(error.to_string().contains("exceeds host limit"));
}

#[test]
fn frame_uses_ritk_slice_display_semantics() {
    let frame = PresentationFrame::from_slice(
        &test_volume(),
        0,
        0,
        WindowLevel::new(127.5, 255.0),
        NamedColorMap::Grayscale,
    )
    .expect("rendered slice frame");
    assert_eq!(frame.width(), 2);
    assert_eq!(frame.height(), 1);
    assert_eq!(frame.rgba(), &[0, 0, 0, 255, 255, 255, 255, 255]);
    assert_eq!(frame.display_spacing().values(), [1.0, 1.0]);
}

#[test]
fn orthogonal_frames_preserve_axis_order_and_dimensions() {
    let frames = PresentationFrame::from_orthogonal_slices(
        &test_volume(),
        [0, 0, 0],
        WindowLevel::new(127.5, 255.0),
        NamedColorMap::Grayscale,
    )
    .expect("orthogonal frames");

    assert_eq!(frames[0].width(), 2);
    assert_eq!(frames[0].height(), 1);
    assert_eq!(frames[1].width(), 2);
    assert_eq!(frames[1].height(), 1);
    assert_eq!(frames[2].width(), 1);
    assert_eq!(frames[2].height(), 1);
    assert_eq!(frames[0].rgba(), frames[1].rgba());
    assert_eq!(frames[2].rgba(), &[0, 0, 0, 255]);
    assert_eq!(frames[0].display_spacing().values(), [1.0, 1.0]);
    assert_eq!(frames[1].display_spacing().values(), [1.0, 1.0]);
    assert_eq!(frames[2].display_spacing().values(), [1.0, 1.0]);
}

#[test]
fn slice_frame_carries_axis_specific_spacing() {
    let volume = LoadedVolume {
        spacing: [2.0, 3.0, 5.0],
        ..test_volume()
    };
    let frames = [0_usize, 1, 2].map(|axis| {
        PresentationFrame::from_slice(
            &volume,
            axis,
            0,
            WindowLevel::new(127.5, 255.0),
            NamedColorMap::Grayscale,
        )
        .expect("rendered spacing frame")
    });
    assert_eq!(frames[0].display_spacing().values(), [3.0, 5.0]);
    assert_eq!(frames[1].display_spacing().values(), [2.0, 5.0]);
    assert_eq!(frames[2].display_spacing().values(), [2.0, 3.0]);
}

#[test]
fn malformed_display_spacing_is_rejected() {
    for [row, column] in [
        [0.0, 1.0],
        [-1.0, 1.0],
        [f64::NAN, 1.0],
        [1.0, f64::INFINITY],
    ] {
        let error = PresentationSpacing::try_new(row, column).expect_err("invalid display spacing");
        assert!(error.to_string().contains("display spacing"));
    }
}

#[test]
fn slice_rejects_invalid_volume_spacing() {
    let volume = LoadedVolume {
        spacing: [1.0, 0.0, 1.0],
        ..test_volume()
    };
    let error = PresentationFrame::from_slice(
        &volume,
        0,
        0,
        WindowLevel::new(127.5, 255.0),
        NamedColorMap::Grayscale,
    )
    .expect_err("invalid volume spacing");
    assert!(error.to_string().contains("display spacing"));
}

#[test]
fn repeated_slice_updates_reuse_rgba_capacity() {
    let volume = test_volume();
    let mut scratch = FrameRenderScratch::default();
    let mut frame = PresentationFrame::empty();
    let window_level = WindowLevel::new(127.5, 255.0);

    frame
        .render_slice_into(
            &volume,
            0,
            0,
            window_level,
            NamedColorMap::Grayscale,
            &mut scratch,
        )
        .expect("first reusable frame");
    let expected =
        PresentationFrame::from_slice(&volume, 0, 0, window_level, NamedColorMap::Grayscale)
            .expect("reference frame");
    assert_eq!(frame.rgba(), expected.rgba());

    frame
        .render_slice_into(
            &volume,
            0,
            0,
            window_level,
            NamedColorMap::Grayscale,
            &mut scratch,
        )
        .expect("second reusable frame");
    assert_eq!(frame.rgba(), expected.rgba());
    let frame_capacity = frame.rgba.capacity();
    let scratch_capacity = scratch.rgba.capacity();

    frame
        .render_slice_into(
            &volume,
            0,
            0,
            window_level,
            NamedColorMap::Grayscale,
            &mut scratch,
        )
        .expect("third reusable frame");
    assert_eq!(frame.rgba(), expected.rgba());
    assert_eq!(frame.rgba.capacity(), frame_capacity);
    assert_eq!(scratch.rgba.capacity(), scratch_capacity);
}
