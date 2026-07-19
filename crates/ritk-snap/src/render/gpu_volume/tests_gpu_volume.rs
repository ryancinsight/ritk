//! Differential equivalence tests: GPU MIP vs CPU reference path. //!
//! # Invariants under test
//!
//! ```text
//! âˆ€ pixel p: |gpu_mip(p) âˆ’ cpu_mip(p)| â‰¤ 2 (u8 channel value)
//! ```
//!
//! The Â±2 tolerance accounts for:
//! - LUT index truncation (`floor(norm * 255)`) vs CPU `colormap.map(norm)`.
//! - `pack4x8unorm` rounding vs CPU integer truncation (`as u8`).
//!
//! # Sprint 272 additions
//!
//! - `gpu_mip_wl_clamps_below_floor_all_black`, `gpu_mip_wl_clamps_above_ceiling_all_white`,
//!   `gpu_mip_repeated_render_identical`.
//!
//! # Headless GPU guard
//!
//! All tests call `GpuVolumeRenderer::try_create()`. If this returns `None`
//! (no GPU available â€” typical on headless CI), the test logs a skip and
//! returns successfully. Tests never fail due to missing GPU hardware.

use std::sync::Arc;

use egui::ColorImage;

use crate::render::mip_vr::render_mip_axial;
use crate::render::{Colormap, WindowLevel};
use crate::LoadedVolume;

use super::GpuVolumeRenderer;

// â”€â”€ Test helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Submit MIP work, block until the GPU completes, then collect and return
/// the result. Two flush rounds guarantee the returned image was rendered
/// from the given `volume`, regardless of any in-flight work from a previous
/// render call with a different volume or params.
///
/// # Protocol
///
/// Round 1: `render_mip` (flushes any previously pending work) â†’ `poll_blocking`.
/// Round 2: `render_mip` (submits work for the current volume) â†’ `poll_blocking`.
/// Round 3: `render_mip` (collects and returns the current volume's result).
///
/// # Invariant
///
/// The returned `Option` is `Some(img)` where `img` was rendered from `volume`
/// iff a GPU is available. Returns `None` only on headless CI (no GPU).
fn render_mip_sync(
    renderer: &mut GpuVolumeRenderer,
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: Colormap,
) -> Option<ColorImage> {
    // Round 1: flush any in-flight work from a previous render call.
    let _ = renderer.render_mip(volume, wl, colormap);
    renderer.poll_blocking();

    // Round 2: submit work for the current volume and wait for completion.
    let _ = renderer.render_mip(volume, wl, colormap);
    renderer.poll_blocking();

    // Round 3: collect the current volume's result.
    renderer.render_mip(volume, wl, colormap)
}

/// Build a small synthetic `LoadedVolume` with uniform intensity `value`.
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

/// GPU MIP vs CPU MIP: Grayscale colormap, synthetic ramp volume.
///
/// # Invariant
///
/// For all pixels: |gpu_r âˆ’ cpu_r| â‰¤ 2 âˆ§ |gpu_g âˆ’ cpu_g| â‰¤ 2 âˆ§ |gpu_b âˆ’ cpu_b| â‰¤ 2.
/// Bound: LUT truncation (â‰¤1) + pack4x8unorm rounding (â‰¤1) = â‰¤2 total.
#[test]
fn gpu_mip_matches_cpu_mip_grayscale() {
    let renderer = GpuVolumeRenderer::try_create();
    let Some(mut renderer) = renderer else {
        tracing::info!("No GPU available â€” skipping GPU MIP differential test");
        return;
    };

    let volume = make_test_volume(8, 16, 16);
    let wl = WindowLevel::new(1024.0, 2048.0);
    let colormap = Colormap::Grayscale;

    let cpu_img = render_mip_axial(&volume, wl, colormap);
    let gpu_img = render_mip_sync(&mut renderer, &volume, wl, colormap)
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
            "Pixel {i}: CPU={c:?} GPU={g:?} max_channel_diff={diff} exceeds Â±2 tolerance"
        );
    }
    tracing::info!(max_diff, "GPU vs CPU MIP max |channel diff|");
}

/// GPU MIP cache: rendering different volumes produces different output.
///
/// Also verifies that zero-intensity volume with WL(0, 200) â†’ all black pixels
/// (norm = 0, Grayscale LUT index 0 = black, alpha = 255).
#[test]
fn gpu_mip_cache_invalidated_on_volume_change() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let vol_a = make_test_volume(4, 8, 8);
    let vol_b = {
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

    // wl_lo = 100 - 0.5*200 = 0; wl_range = 200.
    // vol_b norm = (0 - 0) / 200 = 0 â†’ black; alpha always 255 (MIP).
    let wl = WindowLevel::new(100.0, 200.0);
    let cm = Colormap::Grayscale;

    let img_a = render_mip_sync(&mut renderer, &vol_a, wl, cm).expect("render vol_a");
    let img_b = render_mip_sync(&mut renderer, &vol_b, wl, cm).expect("render vol_b");

    let zero_pixel = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 255);
    for &p in &img_b.pixels {
        assert_eq!(
            p, zero_pixel,
            "vol_b must render to black (norm=0 â†’ Grayscale LUT[0]=black)"
        );
    }

    let all_same = img_a
        .pixels
        .iter()
        .zip(img_b.pixels.iter())
        .all(|(a, b)| a == b);
    assert!(!all_same, "vol_a and vol_b MIP outputs must differ");
}

/// GPU MIP: uniform volume with intensity below WL floor â†’ all black pixels.
///
/// # Derivation
///
/// wl_lo = 128 - 0.5*256 = 0; wl_range = 256.
/// voxel = -100.0 â†’ norm = clamp((-100 - 0)/256, 0, 1) = 0 â†’ LUT[0] = black.
/// Alpha always 255 for MIP (pack4x8unorm(*, *, *, 1.0)).
#[test]
fn gpu_mip_wl_clamps_below_floor_all_black() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let vol = make_uniform_volume(4, 8, 8, -100.0);
    let wl = WindowLevel::new(128.0, 256.0);
    let img = render_mip_sync(&mut renderer, &vol, wl, Colormap::Grayscale)
        .expect("GPU MIP must succeed when GPU is available");

    assert_eq!(
        img.size,
        [8, 8],
        "Output size must be [cols, rows] = [8, 8]"
    );
    for &p in &img.pixels {
        assert_eq!(p.r(), 0, "R must be 0 for below-floor volume");
        assert_eq!(p.g(), 0, "G must be 0 for below-floor volume");
        assert_eq!(p.b(), 0, "B must be 0 for below-floor volume");
        assert_eq!(p.a(), 255, "A must be 255 (MIP: fully opaque)");
    }
}

/// GPU MIP: uniform volume with intensity above WL ceiling â†’ all white pixels.
///
/// # Derivation
///
/// wl_lo = 128 - 0.5*256 = 0; wl_range = 256.
/// voxel = 5000.0 â†’ norm = clamp((5000 - 0)/256, 0, 1) = 1.0 â†’ LUT[255].
/// Grayscale LUT[255] = [255/255, 255/255, 255/255] â†’ pack4x8unorm gives white.
#[test]
fn gpu_mip_wl_clamps_above_ceiling_all_white() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let vol = make_uniform_volume(4, 8, 8, 5000.0);
    let wl = WindowLevel::new(128.0, 256.0);
    let img = render_mip_sync(&mut renderer, &vol, wl, Colormap::Grayscale)
        .expect("GPU MIP must succeed when GPU is available");

    assert_eq!(
        img.size,
        [8, 8],
        "Output size must be [cols, rows] = [8, 8]"
    );
    for &p in &img.pixels {
        // Grayscale LUT[255]: pack4x8unorm(1.0, 1.0, 1.0, 1.0) = round(255) = 255 each.
        assert_eq!(p.r(), 255, "R must be 255 for above-ceiling Grayscale MIP");
        assert_eq!(p.g(), 255, "G must be 255 for above-ceiling Grayscale MIP");
        assert_eq!(p.b(), 255, "B must be 255 for above-ceiling Grayscale MIP");
        assert_eq!(p.a(), 255, "A must be 255 (MIP: fully opaque)");
    }
}

/// GPU MIP: two consecutive renders of the same volume with the same params
/// produce pixel-identical output, verifying that frame buffer reuse (caching)
/// does not corrupt results.
///
/// # Protocol
///
/// Both frames are rendered via `render_mip_sync` (submit â†’ poll_blocking â†’
/// collect) to get deterministic, value-verified results.
#[test]
fn gpu_mip_repeated_render_identical() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let vol = make_test_volume(8, 16, 16);
    let wl = WindowLevel::new(1024.0, 2048.0);
    let cm = Colormap::Grayscale;

    let img1 = render_mip_sync(&mut renderer, &vol, wl, cm).expect("first MIP render");
    let img2 =
        render_mip_sync(&mut renderer, &vol, wl, cm).expect("second MIP render (cache reuse)");

    assert_eq!(img1.size, img2.size, "Sizes must match on repeated render");
    for (i, (a, b)) in img1.pixels.iter().zip(img2.pixels.iter()).enumerate() {
        assert_eq!(
            a, b,
            "Pixel {i}: repeated MIP render must be pixel-identical"
        );
    }
}

/// GPU MIP: single-slice volume (depth=1) produces valid output without panic.
#[test]
fn gpu_mip_empty_volume_no_panic() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let vol = make_test_volume(1, 4, 4);
    let wl = WindowLevel::new(0.0, 1.0);

    let img = render_mip_sync(&mut renderer, &vol, wl, Colormap::Grayscale);
    assert!(
        img.is_some(),
        "Single-slice volume must produce a valid MIP"
    );

    let img = img.unwrap();
    assert_eq!(
        img.size,
        [4, 4],
        "Output size must be [cols, rows] = [4, 4]"
    );
}
