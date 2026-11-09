//! A frame scratch reused across renders yields the pixels of a fresh one.

use crate::render::mip_vr::{
    render_mip_axial_rgba, render_mip_axial_rgba_with_scratch, render_vr_axial_rgba,
};
use crate::render::slice_render::{FrameRenderScratch, RgbaImage};
use crate::render::{NamedColorMap, SliceRenderer, WindowLevel};
use crate::LoadedVolume;
use std::sync::Arc;

/// Construct a volume whose voxel `(d, r, c)` holds `d × R × C + r × C + c`.
fn make_volume(shape: [usize; 3], data: Vec<f32>) -> LoadedVolume {
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

fn ramp_volume() -> LoadedVolume {
    let data = std::iter::successors(Some(0.0_f32), |value| Some(value + 1.0))
        .take(4 * 5 * 6)
        .collect();
    make_volume([4, 5, 6], data)
}

fn frame_image(
    scratch: &mut FrameRenderScratch,
    volume: &LoadedVolume,
    axis: usize,
    index: usize,
    wl: WindowLevel,
) -> RgbaImage {
    let size =
        SliceRenderer::render_rgba_into(scratch, volume, axis, index, wl, NamedColorMap::Grayscale);
    RgbaImage::new(size, scratch.rgba.clone())
}

/// A scratch reused across axes renders each axis as a fresh scratch does.
///
/// Volume `[D=4, R=5, C=6]` with window centre 12 and width 24 keeps the ramp
/// within the display range.
#[test]
fn a_scratch_reused_across_axes_renders_each_as_a_fresh_scratch_does() {
    let volume = ramp_volume();
    let wl = WindowLevel::new(12.0, 24.0);
    let mut scratch = FrameRenderScratch::default();
    for (axis, index) in [(0, 2), (1, 2), (2, 1)] {
        let expected = SliceRenderer::render(&volume, axis, index, wl, NamedColorMap::Grayscale);
        assert_eq!(
            frame_image(&mut scratch, &volume, axis, index, wl),
            expected,
            "axis {axis}"
        );
    }
}

/// Reusing the scratch storage leaves no state from the previous frame.
#[test]
fn reused_frame_scratch_renders_consistent_results() {
    let volume = ramp_volume();
    let wl = WindowLevel::new(12.0, 24.0);
    let mut scratch = FrameRenderScratch::default();
    let first = frame_image(&mut scratch, &volume, 0, 2, wl);
    let other_axis = frame_image(&mut scratch, &volume, 2, 1, wl);
    let second = frame_image(&mut scratch, &volume, 0, 2, wl);
    assert_ne!(first, other_axis);
    assert_eq!(first, second);
}

/// Smaller frames reuse the storage of the largest frame rendered so far.
///
/// Volume `[D=4, R=5, C=6]` renders planes of 30, 24 and 20 samples on axes 0, 1
/// and 2, so each render after the first needs no more than the first left.
#[test]
fn frame_scratch_capacity_does_not_fall_across_smaller_frames() {
    let volume = ramp_volume();
    let wl = WindowLevel::new(12.0, 24.0);
    let mut scratch = FrameRenderScratch::default();
    let mut samples = 0;
    let mut bytes = 0;
    for axis in [0, 0, 1, 2] {
        frame_image(&mut scratch, &volume, axis, 1, wl);
        assert!(
            scratch.pixel_f32.capacity() >= samples.max(20),
            "axis {axis}"
        );
        assert!(scratch.rgba.capacity() >= bytes.max(80), "axis {axis}");
        samples = scratch.pixel_f32.capacity();
        bytes = scratch.rgba.capacity();
    }
    assert!(samples >= 30);
    assert!(bytes >= 120);
}

/// A projection leaves nothing in the scratch that a later projection shows.
#[test]
fn projection_scratch_keeps_no_state_from_an_earlier_projection() {
    let volume = ramp_volume();
    let wl = WindowLevel::new(12.0, 24.0);
    let expected = render_mip_axial_rgba(&volume, wl, NamedColorMap::Grayscale);
    let mut scratch = mnemosyne::AlignedVec::default();
    let earlier = make_volume([1, 6, 8], vec![36.0; 48]);
    let warm =
        render_mip_axial_rgba_with_scratch(&mut scratch, &earlier, wl, NamedColorMap::Grayscale);
    assert_ne!(warm, expected);
    let actual =
        render_mip_axial_rgba_with_scratch(&mut scratch, &volume, wl, NamedColorMap::Grayscale);
    assert_eq!(actual, expected);
}

/// NaN and negative infinity map to the lower window endpoint and positive
/// infinity to the upper one.
#[test]
fn volume_renderers_map_non_finite_voxels_to_window_endpoints() {
    let volume = make_volume([1, 1, 3], vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY]);
    let wl = WindowLevel::new(0.0, 2.0);

    let mip = render_mip_axial_rgba(&volume, wl, NamedColorMap::Grayscale);
    let vr = render_vr_axial_rgba(&volume, wl, NamedColorMap::Grayscale, 1.0);

    let black = [0, 0, 0, 255];
    let white = [255, 255, 255, 255];
    let transparent = [0, 0, 0, 0];
    assert_eq!(mip.rgba_pixels().collect::<Vec<_>>(), [black, black, white]);
    assert_eq!(
        vr.rgba_pixels().collect::<Vec<_>>(),
        [transparent, transparent, white]
    );
}

/// Scalar projections reject RGB input rather than select its first channel.
#[test]
fn volume_renderers_reject_rgb_input() {
    let mut volume = make_volume([1, 1, 1], vec![255.0, 0.0, 0.0]);
    volume.channels = 3;
    let wl = WindowLevel::new(0.0, 1.0);

    let mip = render_mip_axial_rgba(&volume, wl, NamedColorMap::Grayscale);
    let vr = render_vr_axial_rgba(&volume, wl, NamedColorMap::Grayscale, 1.0);

    let rejected = [255, 0, 255, 255];
    assert_eq!(mip.rgba_pixels().collect::<Vec<_>>(), [rejected]);
    assert_eq!(vr.rgba_pixels().collect::<Vec<_>>(), [rejected]);
}
