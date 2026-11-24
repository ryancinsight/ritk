//! Atlas-on-MRI overlay demo: MNI152 atlas registered onto a T1 MRI.
//!
//! Pipeline (all through public crate APIs, no test-only shortcuts):
//! 1. Load both volumes through the unified [`ritk_io::read_image_native`]
//!    dispatch (`.nii.gz` in both cases — the same entry point serves DICOM
//!    directories, MGH, NRRD, and the other dispatch formats).
//! 2. Register the T1 onto the atlas grid with
//!    [`ritk_snap::tools::atlas_overlay::register_secondary_to_primary`]
//!    (affine MI; the same routine backs the viewer tool).
//! 3. Render a fused compare slice with the snap fusion path.
//!
//! Writes `fixed.png`, `moving.png`, `fusion.png`, and `report.json` into the
//! output directory. Run from the workspace root, e.g.
//! `cargo run -p ritk-snap --example atlas_overlay --features eframe-shell --
//! <test_data dir> <output dir>`.

use anyhow::{ensure, Context, Result};
use ritk_io::{literal_arraystring, read_image_native, DicomReadMetadata, NativeImage};
use ritk_snap::render::{
    render_fused_slice, secondary_slice_for_primary, FusedSliceParams, NamedColorMap,
    SliceRenderer, WindowLevel,
};
use ritk_snap::tools::atlas_overlay::register_secondary_to_primary;
use ritk_snap::LoadedVolume;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Window/level from the 2nd/98th intensity percentiles of `values`.
fn window_from_percentiles(values: &[f32]) -> WindowLevel {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| {
        if sorted.is_empty() {
            0.0
        } else {
            sorted[((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1)] as f64
        }
    };
    let (lo, hi) = (at(0.02), at(0.98));
    WindowLevel::new((lo + hi) / 2.0, (hi - lo).max(1.0))
}

/// Wrap f32 voxels in `reference` geometry as a snap volume.
fn to_loaded_volume(data: Vec<f32>, reference: &NativeImage, frame_uid: &'static str) -> Result<LoadedVolume> {
    ensure!(
        data.len()
            == reference.shape()[0] * reference.shape()[1] * reference.shape()[2],
        "overlay voxel count matches its grid"
    );
    let mut metadata = Box::<DicomReadMetadata>::default();
    metadata.frame_of_reference_uid = Some(literal_arraystring::<64>(frame_uid));
    Ok(LoadedVolume {
        data: Arc::new(data),
        shape: reference.shape(),
        channels: 1,
        spacing: reference.spacing().to_array(),
        origin: reference.origin().to_array(),
        direction: reference.direction().to_row_major(),
        metadata: Some(metadata),
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
    })
}

fn save_rgba(path: &Path, pixels: &egui::ColorImage) -> Result<()> {
    let rgba: Vec<_> = pixels.pixels.iter().flat_map(|pixel| pixel.to_array()).collect();
    let width = u32::try_from(pixels.size[0]).context("render width fits u32")?;
    let height = u32::try_from(pixels.size[1]).context("render height fits u32")?;
    image::RgbaImage::from_raw(width, height, rgba)
        .context("rendered RGBA dimensions")
        .map(|buffer| buffer.save(path))??;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let data_dir = PathBuf::from(args.next().context("expected test_data directory")?);
    let out_dir = PathBuf::from(args.next().context("expected output directory")?);
    std::fs::create_dir_all(&out_dir)?;
    let frame_uid = "2.25.2026112401";

    // 1. Unified import: the same entry point loads both NIfTI volumes.
    let fixed_native = read_image_native(data_dir.join("ants_example/mni152.nii.gz"))?;
    let moving_native = read_image_native(data_dir.join("openneuro/sub-01_T1w.nii.gz"))?;
    let fixed_shape = fixed_native.shape();
    ensure!(fixed_shape.iter().all(|extent| *extent > 0), "fixed shape is non-degenerate");
    println!("fixed shape {fixed_shape:?} spacing {:?}", fixed_native.spacing().to_array());
    println!("moving shape {:?} spacing {:?}", moving_native.shape(), moving_native.spacing().to_array());
    let fixed_f32: Vec<f32> = fixed_native
        .data_slice()
        .map_err(|error| anyhow::anyhow!("fixed voxel access: {error}"))?
        .to_vec();
    let moving_f32: Vec<f32> = moving_native
        .data_slice()
        .map_err(|error| anyhow::anyhow!("moving voxel access: {error}"))?
        .to_vec();

    // 2. Register the T1 onto the atlas grid (the viewer tool routine).
    let fixed_volume = to_loaded_volume(fixed_f32.clone(), &fixed_native, frame_uid)?;
    let moving_volume = to_loaded_volume(moving_f32, &moving_native, frame_uid)?;
    let outcome = register_secondary_to_primary(&fixed_volume, &moving_volume, 20)
        .map_err(|error| anyhow::anyhow!("atlas overlay: {error}"))?;
    println!(
        "MI {:.4} -> {:.4} in {} iterations",
        outcome.mi_before, outcome.mi_after, outcome.iterations
    );
    let wl = window_from_percentiles(&fixed_f32);
    println!("window center {:.1} width {:.1}", wl.center, wl.width);

    // 3. Fused compare slice through the snap fusion path.
    let axis = 0_usize;
    let slice = fixed_shape[axis] / 2;
    let fixed_render = SliceRenderer::render(&fixed_volume, axis, slice, wl, NamedColorMap::Grayscale);
    ensure!(!fixed_render.pixels.is_empty(), "fixed slice renders pixels");
    save_rgba(&out_dir.join("fixed.png"), &fixed_render)?;
    let moving_render = SliceRenderer::render(&outcome.resampled, axis, slice, wl, NamedColorMap::Grayscale);
    save_rgba(&out_dir.join("moving.png"), &moving_render)?;
    let secondary_slice =
        secondary_slice_for_primary(&fixed_volume, axis, slice, &outcome.resampled, axis)
            .map_err(|error| anyhow::anyhow!("fused slice index: {error}"))?;
    let fused = render_fused_slice(
        FusedSliceParams { volume: &fixed_volume, axis, slice, wl, colormap: NamedColorMap::Grayscale },
        FusedSliceParams { volume: &outcome.resampled, axis, slice: secondary_slice, wl, colormap: NamedColorMap::Hot },
        0.5,
    )
    .map_err(|error| anyhow::anyhow!("fused render: {error}"))?;
    save_rgba(&out_dir.join("fusion.png"), &fused)?;

    let report = serde_json::json!({
        "fixed": {"path": "ants_example/mni152.nii.gz", "shape": fixed_shape},
        "moving": {"path": "openneuro/sub-01_T1w.nii.gz", "shape": moving_native.shape()},
        "mi_before": outcome.mi_before,
        "mi_after": outcome.mi_after,
        "iterations": outcome.iterations,
        "voxel_transform_full_res": outcome.transform,
        "slice": {"axis": axis, "index": slice, "secondary_index": secondary_slice},
        "window": {"center": wl.center, "width": wl.width},
    });
    std::fs::write(out_dir.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {}", out_dir.display());
    Ok(())
}
