use super::ProjectionAxis;
use anyhow::Result;
use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Reduce the collapsed axis using a native-`f32` fold (max / min).
pub(super) fn fold_native<B, F>(
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

/// Reduce the collapsed axis using `f64` accumulation then finalise to `f32`.
pub(super) fn fold_wide<B, F>(
    axis: ProjectionAxis,
    image: &Image<B, 3>,
    finalize: F,
) -> Result<Image<B, 3>>
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

/// Sample (N−1) std-dev projection: two-pass (mean then variance) per output
/// pixel, matching ITK's `StandardDeviationProjectionImageFilter`.
pub(super) fn project_stddev<B: Backend>(
    axis: ProjectionAxis,
    image: &Image<B, 3>,
) -> Result<Image<B, 3>> {
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

pub(super) fn project_median<B: Backend>(
    axis: ProjectionAxis,
    image: &Image<B, 3>,
) -> Result<Image<B, 3>> {
    let [nz, ny, nx] = image.shape();
    let (vals, _) = extract_vec(image)?;
    match axis {
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

/// Project the collapsed axis to `foreground` if any voxel satisfies `pred`,
/// else `background`.
pub(super) fn project_any<B, P>(
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
