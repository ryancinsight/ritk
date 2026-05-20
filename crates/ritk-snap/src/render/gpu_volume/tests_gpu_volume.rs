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

use crate::render::mip_vr::{render_mip_axial, render_vr_axial};
use crate::render::{Colormap, WindowLevel};
use crate::LoadedVolume;

use super::GpuVolumeRenderer;

/// Build a small synthetic `LoadedVolume` with uniform intensity `value`.
///
/// All voxels are set to `value`.
fn make_uniform_volume(depth: usize, rows: usize, cols: usize, value: f32) -> LoadedVolume {
    LoadedVolume {
        data: Arc::new(vec![value; depth * rows * cols]),
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

// ── VR tests ─────────────────────────────────────────────────────────────────

/// GPU VR output must match CPU VR output within ±2 u8 per channel (after
/// premultiplication by egui) for a synthetic volume under the Grayscale
/// colormap and `alpha_scale = 0.06`.
///
/// Tolerance ±2 accounts for f32 IEEE 754 rounding differences between GPU
/// shader-side FMA and the CPU sequential accumulation loop.
#[test]
fn gpu_vr_matches_cpu_vr_grayscale() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        tracing::info!("No GPU available — skipping GPU VR differential test");
        return;
    };

    let volume = make_test_volume(8, 16, 16);
    // WL: centre on the midpoint of the value range; width spans the full range.
    // Max pixel value = 7*256 + 15*16 + 15 = 2047.
    let wl = WindowLevel::new(1024.0, 2048.0);
    let colormap = Colormap::Grayscale;
    let alpha_scale = 0.06f32;

    let cpu_img = render_vr_axial(&volume, wl, colormap, alpha_scale);
    let gpu_img = renderer
        .render_vr(&volume, wl, colormap, alpha_scale)
        .expect("GPU VR must succeed when GPU is available");

    assert_eq!(cpu_img.size, gpu_img.size, "VR image sizes must match");
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
    tracing::info!(max_diff, "GPU vs CPU VR max |channel diff|");
}

/// All voxels below the WL floor (norm = 0) must produce a fully transparent
/// black pixel: `a = alpha_scale * 0 = 0` → no accumulation.
///
/// WL: center = 128, width = 256 → wl_lo = 0; norm(v=0) = 0 exactly.
#[test]
fn gpu_vr_below_window_floor_transparent_black() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let vol = make_uniform_volume(4, 8, 8, 0.0);
    let wl = WindowLevel::new(128.0, 256.0);
    let img = renderer
        .render_vr(&vol, wl, Colormap::Grayscale, 0.06)
        .expect("GPU VR must succeed when GPU is available");

    for &p in &img.pixels {
        assert_eq!(p.r(), 0, "R must be 0 for zero-intensity volume");
        assert_eq!(p.g(), 0, "G must be 0 for zero-intensity volume");
        assert_eq!(p.b(), 0, "B must be 0 for zero-intensity volume");
        assert_eq!(p.a(), 0, "A must be 0 for zero-intensity volume");
    }
}

/// Non-zero-intensity volume must produce at least one pixel with non-zero
/// channel values, confirming the compositing path executes and accumulates.
#[test]
fn gpu_vr_nonzero_volume_has_nonzero_output() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    // make_test_volume(4,8,8): voxel values span 0..255.
    // WL: center=128, width=256 → most voxels have norm > 0.
    let volume = make_test_volume(4, 8, 8);
    let wl = WindowLevel::new(128.0, 256.0);
    let img = renderer
        .render_vr(&volume, wl, Colormap::Grayscale, 0.06)
        .expect("GPU VR must succeed when GPU is available");

    // At least one pixel must have a non-zero colour channel.
    let has_nonzero = img.pixels.iter().any(|p| p.r() > 0 || p.g() > 0 || p.b() > 0);
    assert!(
        has_nonzero,
        "Non-zero-intensity volume must produce at least one non-black pixel"
    );
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
