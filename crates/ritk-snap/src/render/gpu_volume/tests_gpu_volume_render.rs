//! VR rendering and async pipeline tests for GpuVolumeRenderer.
//!
//! Split from `tests_gpu_volume.rs` to keep file sizes under 500 lines.
//!
//! # Headless GPU guard
//!
//! All tests call `GpuVolumeRenderer::try_create()`. If this returns `None`
//! (no GPU available — typical on headless CI), the test logs a skip and
//! returns successfully. Tests never fail due to missing GPU hardware.

use std::sync::Arc;

use crate::render::mip_vr::render_vr_axial;
use crate::render::{NamedColorMap, WindowLevel};
use crate::LoadedVolume;

use super::GpuVolumeRenderer;

// ── Test helpers (VR-specific) ──────────────────────────────────────────────

/// Submit VR work, block until the GPU completes, then collect and return
/// the result. Mirrors the two-round protocol of `render_mip_sync`.
fn render_vr_sync(
    renderer: &mut GpuVolumeRenderer,
    volume: &LoadedVolume,
    wl: WindowLevel,
    colormap: NamedColorMap,
    alpha_scale: f32,
) -> Option<egui::ColorImage> {
    let _ = renderer.render_vr(volume, wl, colormap, alpha_scale);
    renderer.poll_blocking();
    let _ = renderer.render_vr(volume, wl, colormap, alpha_scale);
    renderer.poll_blocking();
    renderer.render_vr(volume, wl, colormap, alpha_scale)
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

/// GPU VR vs CPU VR: Grayscale colormap, synthetic ramp volume.
///
/// # Invariant
///
/// For all pixels: max(|Δr|, |Δg|, |Δb|) ≤ 2.
/// Bound: LUT truncation (≤1) + pack4x8unorm rounding (≤1) = ≤2 total.
#[test]
fn gpu_vr_matches_cpu_vr_grayscale() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        tracing::info!("No GPU available — skipping GPU VR differential test");
        return;
    };

    let volume = make_test_volume(8, 16, 16);
    let wl = WindowLevel::new(1024.0, 2048.0);
    let colormap = NamedColorMap::Grayscale;
    let alpha_scale = 0.06f32;

    let cpu_img = render_vr_axial(&volume, wl, colormap, alpha_scale);
    let gpu_img = render_vr_sync(&mut renderer, &volume, wl, colormap, alpha_scale)
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

/// GPU VR: zero-intensity uniform volume → transparent black pixels.
///
/// # Derivation
///
/// wl_lo = 0; wl_range = 256. voxel = 0.0 → norm = 0 → a = 0.06 × 0 = 0.
/// No accumulation → acc_r=0, acc_g=0, acc_b=0, acc_alpha=0.
/// pack4x8unorm(0,0,0,0) → bytes [0,0,0,0] → from_rgba_unmultiplied(0,0,0,0)
/// → Color32::TRANSPARENT.
#[test]
fn gpu_vr_below_window_floor_transparent_black() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let vol = make_uniform_volume(4, 8, 8, 0.0);
    let wl = WindowLevel::new(128.0, 256.0);
    let img = render_vr_sync(&mut renderer, &vol, wl, NamedColorMap::Grayscale, 0.06)
        .expect("GPU VR must succeed when GPU is available");

    for &p in &img.pixels {
        assert_eq!(p.r(), 0, "R must be 0 for zero-intensity volume");
        assert_eq!(p.g(), 0, "G must be 0 for zero-intensity volume");
        assert_eq!(p.b(), 0, "B must be 0 for zero-intensity volume");
        assert_eq!(p.a(), 0, "A must be 0 for zero-intensity volume");
    }
}

/// GPU VR: non-zero volume produces at least one non-black pixel.
#[test]
fn gpu_vr_nonzero_volume_has_nonzero_output() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let volume = make_test_volume(4, 8, 8);
    let wl = WindowLevel::new(128.0, 256.0);
    let img = render_vr_sync(&mut renderer, &volume, wl, NamedColorMap::Grayscale, 0.06)
        .expect("GPU VR must succeed when GPU is available");

    let has_nonzero = img
        .pixels
        .iter()
        .any(|p| p.r() > 0 || p.g() > 0 || p.b() > 0);
    assert!(
        has_nonzero,
        "Non-zero-intensity volume must produce at least one non-black pixel"
    );
}

/// GPU VR nonzero: uses render_vr_sync to get a concrete result to inspect.
#[test]
fn gpu_vr_nonzero_volume_has_nonzero_output_sync() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let volume = make_test_volume(4, 8, 8);
    let wl = WindowLevel::new(128.0, 256.0);
    let img = render_vr_sync(&mut renderer, &volume, wl, NamedColorMap::Grayscale, 0.06)
        .expect("GPU VR must succeed when GPU is available");

    let has_nonzero = img
        .pixels
        .iter()
        .any(|p| p.r() > 0 || p.g() > 0 || p.b() > 0);
    assert!(
        has_nonzero,
        "Non-zero-intensity volume must produce at least one non-black pixel"
    );
}

/// GPU VR: two consecutive renders of the same volume produce pixel-identical
/// output, verifying that frame buffer reuse (caching) does not corrupt results.
#[test]
fn gpu_vr_repeated_render_identical() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        return;
    };

    let vol = make_test_volume(8, 16, 16);
    let wl = WindowLevel::new(1024.0, 2048.0);
    let cm = NamedColorMap::Grayscale;

    let img1 = render_vr_sync(&mut renderer, &vol, wl, cm, 0.06).expect("first VR render");
    let img2 =
        render_vr_sync(&mut renderer, &vol, wl, cm, 0.06).expect("second VR render (cache reuse)");

    assert_eq!(
        img1.size, img2.size,
        "Sizes must match on repeated VR render"
    );
    for (i, (a, b)) in img1.pixels.iter().zip(img2.pixels.iter()).enumerate() {
        assert_eq!(
            a, b,
            "Pixel {i}: repeated VR render must be pixel-identical"
        );
    }
}

// ── Sprint 274: async contract tests ─────────────────────────────────────────

/// Async readback contract: first `render_mip` call returns `None`; after
/// `poll_blocking`, the second call returns `Some` with valid pixel data.
///
/// # Formal contract
///
/// Let `r₀ = render_mip(v, wl, cm)` (first call, no cached result).
/// Let `r₁ = render_mip(v, wl, cm)` (after `poll_blocking`).
///
/// Invariant 1: `r₀ = None` — no blocking of the calling thread.
/// Invariant 2: `r₁ = Some(img)` where `img.size = [cols, rows]`.
/// Invariant 3: `img` contains ≥1 non-zero pixel for a non-zero input volume.
#[test]
fn gpu_mip_async_first_call_none_then_yields_result() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        tracing::info!("No GPU available — skipping async contract test");
        return;
    };

    let vol = make_test_volume(4, 8, 8);
    let wl = WindowLevel::new(128.0, 256.0);
    let cm = NamedColorMap::Grayscale;

    // Invariant 1: first call submits GPU work and returns None immediately.
    let r0 = renderer.render_mip(&vol, wl, cm);
    assert!(
        r0.is_none(),
        "First render_mip call must return None (GPU work in-flight, no cached result)"
    );

    // Drive GPU completion without blocking the render thread in production.
    renderer.poll_blocking();

    // Invariant 2 + 3: second call collects the completed result.
    let r1 = renderer
        .render_mip(&vol, wl, cm)
        .expect("Second render_mip must return Some after poll_blocking");

    assert_eq!(r1.size, [8, 8], "Output size must be [cols=8, rows=8]");
    assert_eq!(
        r1.pixels.len(),
        8 * 8,
        "Pixel buffer length must equal rows × cols = 64"
    );
    let has_nonzero = r1
        .pixels
        .iter()
        .any(|p| p.r() > 0 || p.g() > 0 || p.b() > 0);
    assert!(
        has_nonzero,
        "Non-zero test volume must produce at least one non-black MIP pixel"
    );
    tracing::info!(
        "Async MIP contract verified: r0=None, r1=Some({} pixels)",
        r1.pixels.len()
    );
}

/// Async readback contract for VR: first call returns `None`; after
/// `poll_blocking`, the second call returns `Some` with valid pixel data.
///
/// # Formal contract (parallel to MIP contract)
///
/// Invariant 1: `render_vr(v, wl, cm, α)` on first call = `None`.
/// Invariant 2: after `poll_blocking`, `render_vr(v, wl, cm, α)` = `Some(img)`.
/// Invariant 3: `img.size = [cols, rows]`.
#[test]
fn gpu_vr_async_first_call_none_then_yields_result() {
    let Some(mut renderer) = GpuVolumeRenderer::try_create() else {
        tracing::info!("No GPU available — skipping async VR contract test");
        return;
    };

    let vol = make_test_volume(4, 8, 8);
    let wl = WindowLevel::new(128.0, 256.0);
    let cm = NamedColorMap::Grayscale;
    let alpha = 0.06f32;

    // Invariant 1.
    let r0 = renderer.render_vr(&vol, wl, cm, alpha);
    assert!(
        r0.is_none(),
        "First render_vr call must return None (GPU work in-flight, no cached result)"
    );

    renderer.poll_blocking();

    // Invariant 2 + 3.
    let r1 = renderer
        .render_vr(&vol, wl, cm, alpha)
        .expect("Second render_vr must return Some after poll_blocking");

    assert_eq!(r1.size, [8, 8], "Output size must be [cols=8, rows=8]");
    assert_eq!(r1.pixels.len(), 8 * 8, "Pixel buffer length must equal 64");
    tracing::info!(
        "Async VR contract verified: r0=None, r1=Some({} pixels)",
        r1.pixels.len()
    );
}
