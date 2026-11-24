//! Atlas-on-MRI overlay demo: MNI152 atlas registered onto a T1 MRI.
//!
//! Pipeline (all through public crate APIs, no test-only shortcuts):
//! 1. Load both volumes through the unified [`ritk_io::read_image_native`]
//!    dispatch (`.nii.gz` in both cases — the same entry point serves DICOM
//!    directories, MGH, NRRD, and the other dispatch formats).
//! 2. Resample the moving T1 onto the fixed atlas grid (identity), downsample
//!    both to a coarse grid, and run classical affine MI registration
//!    ([`ritk_registration`]) on the coarse pair. Affine (9 DOF with
//!    anisotropic scales) is required because the inputs have different voxel
//!    spacings (0.737 mm vs 1.0 mm); a rigid model cannot absorb that scale
//!    difference in voxel space.
//! 3. Scale the coarse voxel-space transform back to full resolution,
//!    resample the moving volume onto the fixed grid, and render a fused
//!    compare slice with [`ritk_snap`]’s `render_fused_slice`.
//!
//! Writes `fixed.png`, `moving.png`, `fusion.png`, and `report.json` into the
//! output directory. Run from the workspace root, e.g.
//! `cargo run -p ritk-snap --example atlas_overlay --features eframe-shell --
//! <test_data dir> <output dir>`.

use anyhow::{ensure, Context, Result};
use leto::Array3;
use ritk_io::{literal_arraystring, read_image_native, DicomReadMetadata, NativeImage};
use ritk_registration::classical::engine::config::ClassicalConfig;
use ritk_registration::classical::native::image_to_leto_volume;
use ritk_registration::classical::{ImageRegistration, MutualInformationMetric};
use ritk_registration::AffineTransform;
use ritk_snap::render::{
    render_fused_slice, secondary_slice_for_primary, FusedSliceParams, NamedColorMap,
    SliceRenderer, WindowLevel,
};
use ritk_snap::LoadedVolume;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Nearest-neighbour resample of `moving` onto a fixed grid.
///
/// `fixed_to_moving` maps fixed-grid voxel coordinates to moving voxel
/// coordinates (the same convention as the registration engine’s
/// `apply_transform`: `out[x] = moving[T(x)]`). Out-of-field samples are zero.
fn resample_nearest(
    moving: &Array3<f64>,
    out_shape: [usize; 3],
    fixed_to_moving: &[f64; 16],
) -> Array3<f64> {
    let moving_shape = moving.shape();
    let mut out = Array3::zeros(out_shape);
    for z in 0..out_shape[0] {
        for y in 0..out_shape[1] {
            for x in 0..out_shape[2] {
                let homogeneous = [x as f64, y as f64, z as f64, 1.0];
                let mut src = [0.0; 3];
                for row in 0..3 {
                    src[row] = fixed_to_moving[row * 4]
                        * homogeneous[0]
                        + fixed_to_moving[row * 4 + 1] * homogeneous[1]
                        + fixed_to_moving[row * 4 + 2] * homogeneous[2]
                        + fixed_to_moving[row * 4 + 3] * homogeneous[3];
                }
                let (sx, sy, sz) = (src[0].round() as isize, src[1].round() as isize, src[2].round() as isize);
                let sample = if sx >= 0
                    && sy >= 0
                    && sz >= 0
                    && (sx as usize) < moving_shape[2]
                    && (sy as usize) < moving_shape[1]
                    && (sz as usize) < moving_shape[0]
                {
                    moving[[sz as usize, sy as usize, sx as usize]]
                } else {
                    0.0
                };
                out[[z, y, x]] = sample;
            }
        }
    }
    out
}

/// Block-mean downsample by a factor of two in every axis (floor; remainders dropped).
fn downsample_half(volume: &Array3<f64>) -> Array3<f64> {
    let shape = volume.shape();
    let out_shape = [shape[0] / 2, shape[1] / 2, shape[2] / 2];
    let mut out = Array3::zeros(out_shape);
    for z in 0..out_shape[0] {
        for y in 0..out_shape[1] {
            for x in 0..out_shape[2] {
                let mut sum = 0.0;
                let mut count = 0_usize;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (sz, sy, sx) = (2 * z + dz, 2 * y + dy, 2 * x + dx);
                            if sz < shape[0] && sy < shape[1] && sx < shape[2] {
                                sum += volume[[sz, sy, sx]];
                                count += 1;
                            }
                        }
                    }
                }
                out[[z, y, x]] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }
    }
    out
}

/// Scale a voxel-space affine by an isotropic voxel-size ratio `s`.
///
/// Full-resolution voxels relate to coarse voxels by `x_full = s * x_coarse`,
/// so `T_full = S * T_coarse * S^-1` with `S = diag(s, s, s, 1)`.
fn scale_voxel_transform(transform: &[f64; 16], s: f64) -> [f64; 16] {
    let mut out = [0.0; 16];
    for row in 0..4 {
        let row_scale = if row < 3 { s } else { 1.0 };
        for col in 0..4 {
            let col_scale = if col < 3 { s } else { 1.0 };
            out[row * 4 + col] = transform[row * 4 + col] * row_scale / col_scale;
        }
    }
    out
}

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

/// Wrap full-resolution f32 voxels in the fixed geometry as a snap volume.
fn to_loaded_volume(data: Vec<f32>, reference: &NativeImage, frame_uid: &'static str) -> Result<LoadedVolume> {
    ensure!(
        data.len()
            == reference.shape()[0] * reference.shape()[1] * reference.shape()[2],
        "overlay voxel count matches the fixed grid"
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
    let moving_shape = moving_native.shape();
    println!("fixed shape {fixed_shape:?} spacing {:?}", fixed_native.spacing().to_array());
    println!("moving shape {moving_shape:?} spacing {:?}", moving_native.spacing().to_array());

    // 2. Coarse pair on the fixed grid, then rigid MI registration.
    let fixed_full = image_to_leto_volume(&fixed_native)
        .map_err(|error| anyhow::anyhow!("fixed Leto conversion: {error}"))?;
    let moving_full = image_to_leto_volume(&moving_native)
        .map_err(|error| anyhow::anyhow!("moving Leto conversion: {error}"))?;
    let moving_on_fixed = resample_nearest(&moving_full, fixed_shape, &AffineTransform::IDENTITY.0);
    let fixed_coarse = downsample_half(&fixed_full);
    let moving_coarse = downsample_half(&moving_on_fixed);
    ensure!(fixed_coarse.shape() == moving_coarse.shape(), "coarse pair shares one grid");
    let metric = MutualInformationMetric::default();
    let mi_before = metric
        .compute(&fixed_coarse, &moving_coarse)
        .map_err(|error| anyhow::anyhow!("baseline MI: {error}"))?;
    let engine = ImageRegistration::with_config(
        ClassicalConfig { max_iterations: 20, tolerance: 1e-6, step_multiplier: 1.0 },
        MutualInformationMetric::default(),
    );
    let result = engine
        .affine_registration_mutual_info(&moving_coarse, &fixed_coarse, &AffineTransform::IDENTITY)
        .map_err(|error| anyhow::anyhow!("affine MI registration: {error}"))?;
    let coarse_t = result.transform.0;
    let mi_after = metric
        .compute(&fixed_coarse, &resample_nearest(&moving_coarse, fixed_coarse.shape(), &coarse_t))
        .map_err(|error| anyhow::anyhow!("registered MI: {error}"))?;
    ensure!(
        mi_after + 1e-9 >= mi_before,
        "registration must not reduce coarse-grid MI ({mi_before:.4} -> {mi_after:.4})"
    );
    println!("MI {mi_before:.4} -> {mi_after:.4} in {} iterations", result.quality.iterations);

    // 3. Full-resolution resample with the exact scaled transform.
    let full_t = scale_voxel_transform(&coarse_t, 2.0);
    let moved_full = resample_nearest(&moving_full, fixed_shape, &full_t);
    let moved_f32: Vec<f32> = moved_full.iter().copied().map(|value| value as f32).collect();
    let fixed_f32: Vec<f32> = fixed_native
        .data_slice()
        .map_err(|error| anyhow::anyhow!("fixed voxel access: {error}"))?
        .to_vec();
    let wl = window_from_percentiles(&fixed_f32);
    println!("window center {:.1} width {:.1}", wl.center, wl.width);

    // 4. Fused compare slice through the snap fusion path.
    let fixed_volume = to_loaded_volume(fixed_f32, &fixed_native, frame_uid)?;
    let moved_volume = to_loaded_volume(moved_f32, &fixed_native, frame_uid)?;
    let axis = 0_usize;
    let slice = fixed_shape[axis] / 2;
    let fixed_render = SliceRenderer::render(&fixed_volume, axis, slice, wl, NamedColorMap::Grayscale);
    ensure!(!fixed_render.pixels.is_empty(), "fixed slice renders pixels");
    save_rgba(&out_dir.join("fixed.png"), &fixed_render)?;
    let moving_render = SliceRenderer::render(&moved_volume, axis, slice, wl, NamedColorMap::Grayscale);
    save_rgba(&out_dir.join("moving.png"), &moving_render)?;
    let secondary_slice =
        secondary_slice_for_primary(&fixed_volume, axis, slice, &moved_volume, axis)
            .map_err(|error| anyhow::anyhow!("fused slice index: {error}"))?;
    let fused = render_fused_slice(
        FusedSliceParams { volume: &fixed_volume, axis, slice, wl, colormap: NamedColorMap::Grayscale },
        FusedSliceParams { volume: &moved_volume, axis, slice: secondary_slice, wl, colormap: NamedColorMap::Hot },
        0.5,
    )
    .map_err(|error| anyhow::anyhow!("fused render: {error}"))?;
    save_rgba(&out_dir.join("fusion.png"), &fused)?;

    let report = serde_json::json!({
        "fixed": {"path": "ants_example/mni152.nii.gz", "shape": fixed_shape},
        "moving": {"path": "openneuro/sub-01_T1w.nii.gz", "shape": moving_shape},
        "coarse_shape": fixed_coarse.shape(),
        "mi_before": mi_before,
        "mi_after": mi_after,
        "iterations": result.quality.iterations,
        "voxel_transform_full_res": full_t,
        "slice": {"axis": axis, "index": slice, "secondary_index": secondary_slice},
        "window": {"center": wl.center, "width": wl.width},
    });
    std::fs::write(out_dir.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {}", out_dir.display());
    Ok(())
}
