//! Intensity projection filters for 3-D images.
//!
//! # Shape convention
//! All operations follow the RITK convention: `shape = [Z, Y, X]`
//! (axis 0 = Z, axis 1 = Y, axis 2 = X). The projected axis is collapsed
//! to size 1 in the output while all other axes are unchanged.
//!
//! # Precision
//! Max and min accumulation uses native `f32`. Mean, sum, and std-dev
//! accumulation uses `f64` to prevent catastrophic cancellation across
//! large slabs.
//!
//! # Parallelization
//! Each filter parallelises over the output pixels (the non-collapsed axes)
//! using `rayon::into_par_iter`. The inner reduction over the collapsed axis
//! is a sequential `fold` per output pixel.

use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── ProjectionAxis ────────────────────────────────────────────────────────────

/// Axis along which the projection is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionAxis {
    /// Project along axis 0 (Z): output shape `[1, Y, X]`.
    Z = 0,
    /// Project along axis 1 (Y): output shape `[Z, 1, X]`.
    Y = 1,
    /// Project along axis 2 (X): output shape `[Z, Y, 1]`.
    X = 2,
}

// ── MaxIntensityProjectionFilter ──────────────────────────────────────────────

/// Maximum intensity projection along a chosen axis.
pub struct MaxIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl MaxIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        fold_native(self.axis, image, f32::NEG_INFINITY, |a, b| {
            if b > a {
                b
            } else {
                a
            }
        })
    }
}

// ── MinIntensityProjectionFilter ──────────────────────────────────────────────

/// Minimum intensity projection along a chosen axis.
pub struct MinIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl MinIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        fold_native(
            self.axis,
            image,
            f32::INFINITY,
            |a, b| {
                if b < a {
                    b
                } else {
                    a
                }
            },
        )
    }
}

// ── MeanIntensityProjectionFilter ─────────────────────────────────────────────

/// Mean intensity projection along a chosen axis (f64 accumulation).
pub struct MeanIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl MeanIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        fold_wide(self.axis, image, |sum, n| (sum / n as f64) as f32)
    }
}

// ── SumIntensityProjectionFilter ──────────────────────────────────────────────

/// Sum intensity projection along a chosen axis (f64 accumulation).
pub struct SumIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl SumIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        fold_wide(self.axis, image, |sum, _n| sum as f32)
    }
}

// ── StdDevIntensityProjectionFilter ───────────────────────────────────────────

/// Population standard-deviation projection along a chosen axis (f64 accumulation).
///
/// # Formula
/// For each output pixel, let `v₀ … v_{n-1}` be the values along the
/// collapsed axis.
/// `μ = Σvᵢ / n`
/// `σ = sqrt(Σ(vᵢ − μ)² / (n − 1))`  (sample standard deviation, matching ITK's
/// `StandardDeviationProjectionImageFilter`; a length-1 axis yields σ = 0)
pub struct StdDevIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl StdDevIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        project_stddev(self.axis, image)
    }
}

// ── Private: fold_native ──────────────────────────────────────────────────────

/// Reduce the collapsed axis using a native-`f32` fold (max / min).
///
/// `combine(accumulator, next_value) -> new_accumulator`
fn fold_native<B, F>(
    axis: ProjectionAxis,
    image: &Image<B, 3>,
    init: f32,
    combine: F,
) -> Result<Image<B, 3>>
where
    B: Backend,
    F: Fn(f32, f32) -> f32 + Sync,
{
    let [nz, ny, nx] = image.shape();
    let (vals, _) = extract_vec(image)?;
    match axis {
        ProjectionAxis::Z => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(ny * nx, |idx| {
                    let y = idx / nx;
                    let x = idx % nx;
                    (0..nz).fold(init, |acc, z| combine(acc, vals[z * ny * nx + y * nx + x]))
                });
            Ok(rebuild(out, [1, ny, nx], image))
        }
        ProjectionAxis::Y => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * nx, |idx| {
                    let z = idx / nx;
                    let x = idx % nx;
                    (0..ny).fold(init, |acc, y| combine(acc, vals[z * ny * nx + y * nx + x]))
                });
            Ok(rebuild(out, [nz, 1, nx], image))
        }
        ProjectionAxis::X => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny, |idx| {
                    let z = idx / ny;
                    let y = idx % ny;
                    (0..nx).fold(init, |acc, x| combine(acc, vals[z * ny * nx + y * nx + x]))
                });
            Ok(rebuild(out, [nz, ny, 1], image))
        }
    }
}

// ── Private: fold_wide ───────────────────────────────────────────────────────

/// Reduce the collapsed axis using `f64` accumulation then finalise to `f32`.
///
/// `finalize(f64_sum, count) -> f32_output_pixel`
fn fold_wide<B, F>(axis: ProjectionAxis, image: &Image<B, 3>, finalize: F) -> Result<Image<B, 3>>
where
    B: Backend,
    F: Fn(f64, usize) -> f32 + Sync,
{
    let [nz, ny, nx] = image.shape();
    let (vals, _) = extract_vec(image)?;
    match axis {
        ProjectionAxis::Z => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(ny * nx, |idx| {
                    let y = idx / nx;
                    let x = idx % nx;
                    let s = (0..nz).fold(0.0_f64, |acc, z| {
                        acc + vals[z * ny * nx + y * nx + x] as f64
                    });
                    finalize(s, nz)
                });
            Ok(rebuild(out, [1, ny, nx], image))
        }
        ProjectionAxis::Y => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * nx, |idx| {
                    let z = idx / nx;
                    let x = idx % nx;
                    let s = (0..ny).fold(0.0_f64, |acc, y| {
                        acc + vals[z * ny * nx + y * nx + x] as f64
                    });
                    finalize(s, ny)
                });
            Ok(rebuild(out, [nz, 1, nx], image))
        }
        ProjectionAxis::X => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny, |idx| {
                    let z = idx / ny;
                    let y = idx % ny;
                    let s = (0..nx).fold(0.0_f64, |acc, x| {
                        acc + vals[z * ny * nx + y * nx + x] as f64
                    });
                    finalize(s, nx)
                });
            Ok(rebuild(out, [nz, ny, 1], image))
        }
    }
}

// ── Private: project_stddev ───────────────────────────────────────────────────

/// Sample (N−1) std-dev projection: two-pass (mean then variance) per output
/// pixel, matching ITK's `StandardDeviationProjectionImageFilter`. A length-1
/// projection axis yields σ = 0.
fn project_stddev<B: Backend>(axis: ProjectionAxis, image: &Image<B, 3>) -> Result<Image<B, 3>> {
    let [nz, ny, nx] = image.shape();
    let (vals, _) = extract_vec(image)?;
    match axis {
        ProjectionAxis::Z => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(ny * nx, |idx| {
                    let y = idx / nx;
                    let x = idx % nx;
                    let mean = (0..nz).fold(0.0_f64, |acc, z| {
                        acc + vals[z * ny * nx + y * nx + x] as f64
                    }) / nz as f64;
                    let var = (0..nz).fold(0.0_f64, |acc, z| {
                        let d = vals[z * ny * nx + y * nx + x] as f64 - mean;
                        acc + d * d
                    }) / (nz - 1).max(1) as f64;
                    var.sqrt() as f32
                });
            Ok(rebuild(out, [1, ny, nx], image))
        }
        ProjectionAxis::Y => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * nx, |idx| {
                    let z = idx / nx;
                    let x = idx % nx;
                    let mean = (0..ny).fold(0.0_f64, |acc, y| {
                        acc + vals[z * ny * nx + y * nx + x] as f64
                    }) / ny as f64;
                    let var = (0..ny).fold(0.0_f64, |acc, y| {
                        let d = vals[z * ny * nx + y * nx + x] as f64 - mean;
                        acc + d * d
                    }) / (ny - 1).max(1) as f64;
                    var.sqrt() as f32
                });
            Ok(rebuild(out, [nz, 1, nx], image))
        }
        ProjectionAxis::X => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny, |idx| {
                    let z = idx / ny;
                    let y = idx % ny;
                    let mean = (0..nx).fold(0.0_f64, |acc, x| {
                        acc + vals[z * ny * nx + y * nx + x] as f64
                    }) / nx as f64;
                    let var = (0..nx).fold(0.0_f64, |acc, x| {
                        let d = vals[z * ny * nx + y * nx + x] as f64 - mean;
                        acc + d * d
                    }) / (nx - 1).max(1) as f64;
                    var.sqrt() as f32
                });
            Ok(rebuild(out, [nz, ny, 1], image))
        }
    }
}

// ── MedianIntensityProjectionFilter ───────────────────────────────────────────

/// Median intensity projection along a chosen axis.
///
/// For each output pixel the median of the collapsed-axis values is taken via
/// `select_nth_unstable` at index `n/2` (the upper-middle for even `n`), matching
/// ITK `MedianProjectionImageFilter` (`std::nth_element` at `size/2`).
pub struct MedianIntensityProjectionFilter {
    axis: ProjectionAxis,
}

impl MedianIntensityProjectionFilter {
    pub fn new(axis: ProjectionAxis) -> Self {
        Self { axis }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        project_median(self.axis, image)
    }
}

/// Median of a slice via `select_nth_unstable` at `len/2` (ITK convention).
#[inline]
fn median_at_half(buf: &mut [f32]) -> f32 {
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    let k = n / 2;
    let (_, m, _) = buf.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    *m
}

fn project_median<B: Backend>(axis: ProjectionAxis, image: &Image<B, 3>) -> Result<Image<B, 3>> {
    let [nz, ny, nx] = image.shape();
    let (vals, _) = extract_vec(image)?;
    match axis {
        // Z-projection: output [1, ny, nx]. Parallelise over ny output rows
        // (chunk_size = nx). `col` allocated once per row, reused for all nx
        // pixels in that row — avoids the per-pixel Vec allocation (P-3).
        ProjectionAxis::Z => {
            let mut out = vec![0.0_f32; ny * nx];
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                &mut out,
                nx,
                |iy, row_slice| {
                    let mut col = Vec::with_capacity(nz);
                    for ix in 0..nx {
                        col.clear();
                        for z in 0..nz {
                            col.push(vals[z * ny * nx + iy * nx + ix]);
                        }
                        row_slice[ix] = median_at_half(&mut col);
                    }
                },
            );
            Ok(rebuild(out, [1, ny, nx], image))
        }
        // Y-projection: output [nz, 1, nx]. Parallelise over nz slices.
        ProjectionAxis::Y => {
            let mut out = vec![0.0_f32; nz * nx];
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                &mut out,
                nx,
                |iz, row_slice| {
                    let mut col = Vec::with_capacity(ny);
                    for ix in 0..nx {
                        col.clear();
                        for y in 0..ny {
                            col.push(vals[iz * ny * nx + y * nx + ix]);
                        }
                        row_slice[ix] = median_at_half(&mut col);
                    }
                },
            );
            Ok(rebuild(out, [nz, 1, nx], image))
        }
        // X-projection: output [nz, ny, 1]. Parallelise over nz slices;
        // each row vals[iz*ny*nx + iy*nx ..] is contiguous in memory.
        ProjectionAxis::X => {
            let mut out = vec![0.0_f32; nz * ny];
            moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
                &mut out,
                ny,
                |iz, col_slice| {
                    let mut row = Vec::with_capacity(nx);
                    for (iy, cell) in col_slice.iter_mut().enumerate() {
                        row.clear();
                        let base = iz * ny * nx + iy * nx;
                        row.extend_from_slice(&vals[base..base + nx]);
                        *cell = median_at_half(&mut row);
                    }
                },
            );
            Ok(rebuild(out, [nz, ny, 1], image))
        }
    }
}

// ── BinaryProjectionFilter ────────────────────────────────────────────────────

/// Binary projection along a chosen axis: a result pixel is `foreground` if
/// **any** voxel along the collapsed axis equals `foreground`, else `background`.
///
/// Matches ITK `BinaryProjectionImageFilter` (`sitk.BinaryProjection`).
pub struct BinaryProjectionFilter {
    axis: ProjectionAxis,
    foreground: f32,
    background: f32,
}

impl BinaryProjectionFilter {
    pub fn new(axis: ProjectionAxis, foreground: f32, background: f32) -> Self {
        Self {
            axis,
            foreground,
            background,
        }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let fg = self.foreground;
        project_any(
            self.axis,
            image,
            self.foreground,
            self.background,
            move |v| v == fg,
        )
    }
}

// ── BinaryThresholdProjectionFilter ───────────────────────────────────────────

/// Binary-threshold projection: a result pixel is `foreground` if **any** voxel
/// along the collapsed axis is `>= threshold`, else `background`.
///
/// Matches ITK `BinaryThresholdProjectionImageFilter` (`sitk.BinaryThresholdProjection`).
pub struct BinaryThresholdProjectionFilter {
    axis: ProjectionAxis,
    threshold: f32,
    foreground: f32,
    background: f32,
}

impl BinaryThresholdProjectionFilter {
    pub fn new(axis: ProjectionAxis, threshold: f32, foreground: f32, background: f32) -> Self {
        Self {
            axis,
            threshold,
            foreground,
            background,
        }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let thr = self.threshold;
        project_any(
            self.axis,
            image,
            self.foreground,
            self.background,
            move |v| v >= thr,
        )
    }
}

/// Project the collapsed axis to `foreground` if any voxel satisfies `pred`,
/// else `background`.
fn project_any<B, P>(
    axis: ProjectionAxis,
    image: &Image<B, 3>,
    foreground: f32,
    background: f32,
    pred: P,
) -> Result<Image<B, 3>>
where
    B: Backend,
    P: Fn(f32) -> bool + Sync,
{
    let pick = |hit: bool| if hit { foreground } else { background };
    let [nz, ny, nx] = image.shape();
    let (vals, _) = extract_vec(image)?;
    match axis {
        ProjectionAxis::Z => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(ny * nx, |idx| {
                    let (y, x) = (idx / nx, idx % nx);
                    pick((0..nz).any(|z| pred(vals[z * ny * nx + y * nx + x])))
                });
            Ok(rebuild(out, [1, ny, nx], image))
        }
        ProjectionAxis::Y => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * nx, |idx| {
                    let (z, x) = (idx / nx, idx % nx);
                    pick((0..ny).any(|y| pred(vals[z * ny * nx + y * nx + x])))
                });
            Ok(rebuild(out, [nz, 1, nx], image))
        }
        ProjectionAxis::X => {
            let out: Vec<f32> =
                moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny, |idx| {
                    let (z, y) = (idx / ny, idx % ny);
                    pick((0..nx).any(|x| pred(vals[z * ny * nx + y * nx + x])))
                });
            Ok(rebuild(out, [nz, ny, 1], image))
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_projection.rs"]
mod tests;
