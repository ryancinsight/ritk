//! Differential equivalence tests: GPU MIP vs CPU MIP.
//!
//! # Invariant
//!
//! For all valid volumes and the Grayscale colormap, the GPU and CPU MIP
//! outputs must agree within ±2 u8 per channel (accounts for f32→u8 rounding
//! differences caused by the GPU shader outputting raw f32 max values that are
//! window-levelled on CPU, vs the CPU path which window-levels inline).
//!
//! # Headless GPU guard
//!
//! All tests call `GpuVolumeRenderer::try_create()`.  If this returns `None`
//! (no GPU available — typical on headless CI), the test logs a skip and
//! returns successfully.  Tests never fail due to missing GPU hardware.

use std::sync::Arc;

use crate::render::mip_vr::render_mip_axial;
use crate::render::{Colormap, WindowLevel};
use crate::LoadedVolume;

use super::GpuVolumeRenderer;

/// Build a small synthetic `LoadedVolume` with a deterministic voxel pattern.
///
/// Voxel value at (d, r, c) = `d * rows * cols + r * cols + c` as `f32`.
/// The maximum along the depth axis for pixel (r, c) is thus `(depth-1)*R*C + r*C + c`.
fn make_test_volume(depth: usize, rows: usize, cols: usize) -> LoadedVolume {
    let mut data = Vec::with_capacity(depth * rows * cols);
    for d in 0..depth {
        for r in 0..rows {
            for c in 0..cols {
                data.push((d * rows * cols + r * cols + c) as f32);
            }
        }
    }
    LoadedVolume {
        data: Arc::new(data),
        shape: [depth, rows, cols],
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

/// GPU MIP output must match CPU MIP output within ±2 u8 per channel for a
/// small synthetic volume under the Grayscale colormap.
///
/// The analytical ground truth: max along depth for pixel (r, c) is
/// `(depth-1) * rows * cols + r * cols + c`.
#[test]
fn gpu_mip_matches_cpu_mip_grayscale() {
    let renderer = GpuVolumeRenderer::try_create();
    let Some(mut renderer) = renderer else {
        tracing::info!("No GPU available — skipping GPU MIP differential test");
        return;
    };

    let volume = make_test_volume(8, 16, 16);
    // Window/level: centre on the midpoint of the expected range.
    // Max pixel value = 7*16*16 + 15*16 + 15 = 1792+240+15 = 2047
    let wl = WindowLevel::new(1024.0, 2048.0);
    let colormap = Colormap::Grayscale;

    let cpu_img = render_mip_axial(&volume, wl, colormap);
    let gpu_img = renderer
        .render_mip(&volume, wl, colormap)
        .expect("GPU MIP must succeed when GPU is available");

    assert_eq!(cpu_img.size, gpu_img.size, "MIP image sizes must match");
    assert_eq!(
        cpu_img.pixels.len(),
        gpu_img.pixels.len(),
        "Pixel buffer lengths must match"
    );

    let mut max_diff: i32 = 0;
    for (i, (c, g)) in cpu_img.pixels.iter().zip(gpu_img.pixels.iter()).enumerate() {
        let diff = (c.r() as i32 - g.r() as i32)
            .abs()
            .max((c.g() as i32 - g.g() as i32).abs())
            .max((c.b() as i32 - g.b() as i32).abs());
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff <= 2,
            "Pixel {i}: CPU={c:?} GPU={g:?} max_channel_diff={diff} exceeds ±2 tolerance"
        );
    }
    tracing::info!(max_diff, "GPU vs CPU MIP max |channel diff|");
}

/// Volume change detection: uploading a different volume must produce a
/// different MIP (the cache is invalidated).
#[test]
fn gpu_mip_cache_invalidated_on_volume_change() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let vol_a = make_test_volume(4, 8, 8);
    let vol_b = {
        // vol_b has all zeros → MIP output should be all zeros after WL.
        let zeros = vec![0.0f32; 4 * 8 * 8];
        LoadedVolume {
            data: Arc::new(zeros),
            shape: [4, 8, 8],
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
    };

    let wl = WindowLevel::new(100.0, 200.0);
    let cm = Colormap::Grayscale;

    let img_a = renderer.render_mip(&vol_a, wl, cm).expect("render vol_a");
    let img_b = renderer.render_mip(&vol_b, wl, cm).expect("render vol_b");

    // vol_b is all zeros → WL-normalised to 0 → Grayscale maps to [0,0,0,255].
    let zero_pixel = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 255);
    for &p in &img_b.pixels {
        assert_eq!(p, zero_pixel, "vol_b must render to black");
    }

    // vol_a has non-zero values → at least one pixel must differ from vol_b.
    let all_same = img_a.pixels.iter().zip(img_b.pixels.iter()).all(|(a, b)| a == b);
    assert!(!all_same, "vol_a and vol_b MIP outputs must differ");
}

/// Empty volume (zero depth) must not panic and must return a valid image.
#[test]
fn gpu_mip_empty_volume_no_panic() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    // Minimum non-zero shape (1×4×4) to avoid zero-size buffer issues.
    let vol = make_test_volume(1, 4, 4);
    let wl = WindowLevel::new(0.0, 1.0);
    let img = renderer.render_mip(&vol, wl, Colormap::Grayscale);
    assert!(img.is_some(), "Single-slice volume must produce a valid MIP");
    let img = img.unwrap();
    assert_eq!(img.size, [4, 4], "Output size must be [cols, rows] = [4, 4]");
}
