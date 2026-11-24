//! Atlas-overlay registration tool.
//!
//! Registers the compare (secondary) volume onto the primary grid with the
//! classical affine MI engine and returns a fused-ready volume that shares
//! the primary grid exactly, so [`crate::render::fusion::render_fused_slice`]
//!//! accepts the pair.
//!
//! The pipeline mirrors `examples/atlas_overlay.rs` (which calls this
//! function): identity resample of the moving volume onto the fixed grid,
//! block-mean downsample of both to a coarse grid, affine MI registration on
//! the coarse pair, exact scaling of the voxel-space transform back to full
//! resolution, and a final nearest-neighbour resample. Affine (9 DOF with
//! anisotropic scales) is required because clinical pairs routinely differ
//! in voxel spacing; a rigid model cannot absorb that scale gap in voxel
//! space.
//!
//! The output carries the primary frame of reference: when the primary has a
//! frame-of-reference UID it is stamped onto the output, otherwise the output
//! is left without one so the fusion validator accepts the pair through its
//! identical-geometry arm.

use anyhow::{ensure, Context, Result};
use leto::Array3;
use ritk_registration::classical::engine::config::ClassicalConfig;
use ritk_registration::classical::ImageRegistration;
use ritk_registration::classical::MutualInformationMetric;
use ritk_registration::AffineTransform;
use std::sync::Arc;

use super::super::LoadedVolume;

/// Outcome of registering the compare volume onto the primary grid.
#[derive(Debug)]
pub struct AtlasOverlayOutcome {
    /// Full-resolution voxel-space transform mapping fixed voxels to moving
    /// voxels (`out[x] = moving[T(x)]`, the registration engine convention).
    pub transform: [f64; 16],
    /// Coarse-grid mutual information before registration (identity).
    pub mi_before: f64,
    /// Coarse-grid mutual information after registration.
    pub mi_after: f64,
    /// Hill-climbing iterations executed.
    pub iterations: usize,
    /// Moving volume resampled onto the primary grid and geometry, fused-ready.
    pub resampled: LoadedVolume,
}

/// Register `secondary` onto the `primary` grid.
///
/// Both volumes must be single-channel with non-degenerate shapes and
/// `max_iterations` must be positive. The returned volume shares the primary
/// shape, spacing, origin, and direction exactly, and carries the primary
/// frame of reference.
///
/// # Errors
///
/// Returns an error for multi-channel input, degenerate shapes, a non-positive
/// iteration budget, or a registration failure from the classical engine. A
/// converged-but-worse MI (the engine only accepts improvements, so this
/// indicates a numerical fault) is also rejected.
pub fn register_secondary_to_primary(
    primary: &LoadedVolume,
    secondary: &LoadedVolume,
    max_iterations: usize,
) -> Result<AtlasOverlayOutcome> {
    ensure!(primary.channels == 1, "primary must be single-channel");
    ensure!(secondary.channels == 1, "secondary must be single-channel");
    ensure!(
        primary.shape.iter().all(|extent| *extent > 0),
        "primary shape is degenerate"
    );
    ensure!(
        secondary.shape.iter().all(|extent| *extent > 0),
        "secondary shape is degenerate"
    );
    ensure!(max_iterations > 0, "iteration budget must be positive");

    let fixed_full = to_f64_array(&primary.data, primary.shape).context("primary array")?;
    let moving_full = to_f64_array(&secondary.data, secondary.shape).context("secondary array")?;
    let moving_on_fixed =
        resample_nearest(&moving_full, primary.shape, &AffineTransform::IDENTITY.0)
            .context("identity resample")?;
    let fixed_coarse = downsample_half(&fixed_full).context("fixed coarse grid")?;
    let moving_coarse = downsample_half(&moving_on_fixed).context("moving coarse grid")?;
    ensure!(
        fixed_coarse.shape() == moving_coarse.shape(),
        "coarse pair shares one grid"
    );

    let metric = MutualInformationMetric::default();
    let mi_before = metric
        .compute(&fixed_coarse, &moving_coarse)
        .map_err(|error| anyhow::anyhow!("baseline MI: {error}"))?;
    let engine = ImageRegistration::with_config(
        ClassicalConfig {
            max_iterations,
            tolerance: 1e-6,
            step_multiplier: 1.0,
        },
        MutualInformationMetric::default(),
    );
    let result = engine
        .affine_registration_mutual_info(&moving_coarse, &fixed_coarse, &AffineTransform::IDENTITY)
        .map_err(|error| anyhow::anyhow!("affine MI registration: {error}"))?;
    let coarse_t = result.transform.0;
    let moved_coarse = resample_nearest(&moving_coarse, fixed_coarse.shape(), &coarse_t)
        .context("registered coarse resample")?;
    let mi_after = metric
        .compute(&fixed_coarse, &moved_coarse)
        .map_err(|error| anyhow::anyhow!("registered MI: {error}"))?;
    ensure!(
        mi_after + 1e-9 >= mi_before,
        "registration must not reduce coarse-grid MI ({mi_before:.4} -> {mi_after:.4})"
    );

    let full_t = scale_voxel_transform(&coarse_t, 2.0);
    let moved_full = resample_nearest(&moving_full, primary.shape, &full_t)
        .context("full-resolution resample")?;
    let moved_f32: Vec<f32> = moved_full
        .iter()
        .copied()
        .map(|value| value as f32)
        .collect();

    // Frame-of-reference rule: inherit the primary UID when present so the
    // fusion validator sees one shared frame; leave it absent otherwise so
    // the identical-geometry arm applies.
    let mut metadata = Box::<ritk_io::DicomReadMetadata>::default();
    metadata.frame_of_reference_uid = primary
        .metadata
        .as_deref()
        .and_then(|meta| meta.frame_of_reference_uid.clone());

    Ok(AtlasOverlayOutcome {
        transform: full_t,
        mi_before,
        mi_after,
        iterations: result.quality.iterations,
        resampled: LoadedVolume {
            data: Arc::new(moved_f32),
            shape: primary.shape,
            channels: 1,
            spacing: primary.spacing,
            origin: primary.origin,
            direction: primary.direction,
            metadata: Some(metadata),
            source: secondary.source.clone(),
            modality: secondary.modality,
            patient_name: secondary.patient_name.clone(),
            patient_id: secondary.patient_id.clone(),
            study_date: secondary.study_date,
            series_description: secondary.series_description.clone(),
            series_time: secondary.series_time,
            patient_weight_kg: secondary.patient_weight_kg,
            injected_dose_bq: secondary.injected_dose_bq,
            radionuclide_half_life_s: secondary.radionuclide_half_life_s,
            radiopharmaceutical_start_time: secondary.radiopharmaceutical_start_time,
            decay_correction: secondary.decay_correction,
        },
    })
}

fn to_f64_array(values: &[f32], shape: [usize; 3]) -> Result<Array3<f64>> {
    let wide: Vec<f64> = values.iter().copied().map(f64::from).collect();
    Array3::from_shape_vec(shape, wide).map_err(|error| anyhow::anyhow!("Leto shape: {error}"))
}

/// Nearest-neighbour resample of `moving` onto `out_shape`.
///
/// `fixed_to_moving` maps fixed-grid voxels to moving voxels (the engine
/// convention `out[x] = moving[T(x)]`). Out-of-field samples are zero.
///
/// This is the workhorse behind [`register_secondary_to_primary`] and is
/// public so its throughput is directly benchmarked (`benches/atlas_resample.rs`).
///
/// Parallelized over output z-slices with moirai (house pattern from
/// `ritk_registration::deformable_field_ops::warp`): every output voxel is a
/// pure function of the immutable input, so slices write to disjoint ranges
/// with no synchronization.
///
/// # Errors
///
/// Returns an error when the input uses non-contiguous storage or the output
/// buffer length does not match the requested shape.
pub fn resample_nearest(
    moving: &Array3<f64>,
    out_shape: [usize; 3],
    fixed_to_moving: &[f64; 16],
) -> Result<Array3<f64>> {
    let moving_flat = moving
        .as_slice()
        .context("resample input must be contiguous")?;
    let moving_shape = moving.shape();
    let n = out_shape[0] * out_shape[1] * out_shape[2];
    let mut out = vec![0.0_f64; n];
    resample_flat(
        moving_flat,
        moving_shape,
        &mut out,
        out_shape,
        fixed_to_moving,
    );
    Array3::from_shape_vec(out_shape, out)
        .map_err(|error| anyhow::anyhow!("resample output: {error}"))
}

/// Flat-slice nearest-neighbour resample; the parallel kernel behind
/// [`resample_nearest`].
///
/// Parallelized over output z-slices: every output voxel is a pure function
/// of the immutable input, so slices write to disjoint ranges with no
/// synchronization.
fn resample_flat(
    moving: &[f64],
    moving_shape: [usize; 3],
    out: &mut [f64],
    out_shape: [usize; 3],
    fixed_to_moving: &[f64; 16],
) {
    let [md, mh, mw] = moving_shape;
    let [_od, oh, ow] = out_shape;
    let slice_len = oh * ow;
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        out,
        slice_len,
        |oz, out_s| {
            for oy in 0..oh {
                for ox in 0..ow {
                    let wx = ox as f64;
                    let wy = oy as f64;
                    let wz = oz as f64;
                    let sx = (fixed_to_moving[0] * wx
                        + fixed_to_moving[1] * wy
                        + fixed_to_moving[2] * wz
                        + fixed_to_moving[3])
                        .round() as isize;
                    let sy = (fixed_to_moving[4] * wx
                        + fixed_to_moving[5] * wy
                        + fixed_to_moving[6] * wz
                        + fixed_to_moving[7])
                        .round() as isize;
                    let sz = (fixed_to_moving[8] * wx
                        + fixed_to_moving[9] * wy
                        + fixed_to_moving[10] * wz
                        + fixed_to_moving[11])
                        .round() as isize;
                    out_s[oy * ow + ox] = if sx >= 0
                        && sy >= 0
                        && sz >= 0
                        && (sx as usize) < mw
                        && (sy as usize) < mh
                        && (sz as usize) < md
                    {
                        moving[(sz as usize * mh + sy as usize) * mw + sx as usize]
                    } else {
                        0.0
                    };
                }
            }
        },
    );
}

/// Block-mean downsample by a factor of two in every axis (remainders dropped).
///
/// Public alongside [`resample_nearest`] so the coarse-grid construction cost
/// is directly benchmarked (`benches/atlas_resample.rs`).
pub fn downsample_half(volume: &Array3<f64>) -> Result<Array3<f64>> {
    let flat = volume
        .as_slice()
        .context("downsample input must be contiguous")?;
    let shape = volume.shape();
    let out_shape = [shape[0] / 2, shape[1] / 2, shape[2] / 2];
    let n = out_shape[0] * out_shape[1] * out_shape[2];
    let mut out = vec![0.0_f64; n];
    downsample_flat(flat, shape, &mut out, out_shape);
    Array3::from_shape_vec(out_shape, out)
        .map_err(|error| anyhow::anyhow!("downsample output: {error}"))
}

/// Flat-slice block-mean downsample; the parallel kernel behind
/// [`downsample_half`].
///
/// Parallelized over output z-slices: each output voxel reads at most eight
/// input voxels and writes to a disjoint range.
fn downsample_flat(volume: &[f64], shape: [usize; 3], out: &mut [f64], out_shape: [usize; 3]) {
    let [nz, ny, nx] = shape;
    let [_oz, oh, ow] = out_shape;
    let slice_len = oh * ow;
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        out,
        slice_len,
        |oz, out_s| {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut sum = 0.0;
                    let mut count = 0_usize;
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let (sz, sy, sx) = (2 * oz + dz, 2 * oy + dy, 2 * ox + dx);
                                if sz < nz && sy < ny && sx < nx {
                                    sum += volume[(sz * ny + sy) * nx + sx];
                                    count += 1;
                                }
                            }
                        }
                    }
                    out_s[oy * ow + ox] = if count > 0 { sum / count as f64 } else { 0.0 };
                }
            }
        },
    );
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

#[cfg(test)]
#[path = "tests_atlas_overlay.rs"]
mod tests;
