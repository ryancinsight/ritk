//! Meijster–Roerdink–Hesselink (2000) core EDT routines.
//!
//! Pure mathematical functions: no image I/O, no coeus dependency.
#![forbid(unsafe_code)]

pub(super) const INF: f64 = 1e30_f64;

/// 1-D first pass: squared 1-D distance to nearest foreground along one axis.
/// `fg_row[i] = true` iff voxel i is foreground. `s` is the voxel spacing in mm.
/// Writes squared distances `(dist_mm)²` into `out[0..n]`.
pub(super) fn phase1_row(fg_row: &[bool], n: usize, s: f64, out: &mut [f64]) {
    // Forward scan: accumulate distance from left-most foreground
    if fg_row[0] {
        out[0] = 0.0;
    } else {
        out[0] = INF;
    }
    for i in 1..n {
        if fg_row[i] {
            out[i] = 0.0;
        } else if out[i - 1] < INF {
            out[i] = out[i - 1] + s;
        } else {
            out[i] = INF;
        }
    }
    // Backward scan: correct for foreground to the right
    for i in (0..n - 1).rev() {
        let d = out[i + 1] + s;
        if d < out[i] {
            out[i] = d;
        }
    }
    // Square to get squared distances
    for v in out.iter_mut() {
        *v = *v * *v;
    }
}

/// Meijster separability function: voxel index `x` at which parabola centered at `u`
/// overtakes the one centered at `i`, given accumulated squared distances `gi` and `gu`
/// and spacing `s` in mm.
///
/// Derivation: solve `(x−i)²s² + gi = (x−u)²s² + gu` for x:
/// `x = (s²(u²−i²) + gu − gi) / (2s²(u−i))`
#[inline]
pub(super) fn sep(i: isize, u: isize, gi: f64, gu: f64, s: f64) -> isize {
    let s2 = s * s;
    let num = s2 * (u * u - i * i) as f64 + gu - gi;
    let den = 2.0 * s2 * (u - i) as f64;
    (num / den).floor() as isize
}

/// Distance contribution of parabola centered at index `i` evaluated at `x`.
/// `gi` = accumulated squared distance from previous axis.
#[inline]
pub(super) fn f_dt(x: isize, i: isize, gi: f64, s: f64) -> f64 {
    let d = (x - i) as f64 * s;
    d * d + gi
}

/// Meijster parabolic lower-envelope pass for one row.
/// Input: `g[i]` = accumulated squared distance from all previous axes at position i.
/// Writes updated squared distances into `dt[0..n]`.
/// `s_stack[0..n]` and `t_stack[0..n]` are scratch buffers for the envelope stack.
pub(super) fn meijster_row(
    g: &[f64],
    n: usize,
    s: f64,
    s_stack: &mut [isize],
    t_stack: &mut [isize],
    dt: &mut [f64],
) {
    debug_assert!(s_stack.len() >= n);
    debug_assert!(t_stack.len() >= n);
    debug_assert!(dt.len() >= n);

    if n == 0 {
        return;
    }
    if n == 1 {
        dt[0] = g[0];
        return;
    }

    let mut q: usize = 0;

    // Seed the stack with the first finite entry.
    // INF parabolas (no foreground in upstream pass) are skipped — they never
    // contribute a finite minimum so they can never become dominant.
    let mut initialized = false;
    for u0 in 0..n {
        if g[u0] < INF {
            s_stack[0] = u0 as isize;
            t_stack[0] = 0;
            q = 0;
            initialized = true;
            // Process remaining voxels
            for u in (u0 + 1)..n {
                let gu = g[u];
                if gu >= INF {
                    continue; // skip all-background parabolas
                }
                loop {
                    if q == 0 {
                        if f_dt(t_stack[0], s_stack[0], g[s_stack[0] as usize], s)
                            >= f_dt(t_stack[0], u as isize, gu, s)
                        {
                            s_stack[0] = u as isize;
                            t_stack[0] = 0;
                        } else {
                            let w = sep(s_stack[0], u as isize, g[s_stack[0] as usize], gu, s)
                                .saturating_add(1);
                            if w < n as isize {
                                q += 1;
                                s_stack[q] = u as isize;
                                t_stack[q] = w;
                            }
                        }
                        break;
                    }
                    if f_dt(t_stack[q], s_stack[q], g[s_stack[q] as usize], s)
                        >= f_dt(t_stack[q], u as isize, gu, s)
                    {
                        q -= 1;
                    } else {
                        let w = sep(s_stack[q], u as isize, g[s_stack[q] as usize], gu, s)
                            .saturating_add(1);
                        if w < n as isize {
                            q += 1;
                            s_stack[q] = u as isize;
                            t_stack[q] = w;
                        }
                        break;
                    }
                }
            }
            break;
        }
    }
    // If no foreground found in this row, all distances remain INF.
    if !initialized {
        for v in dt.iter_mut().take(n) {
            *v = INF;
        }
        return;
    }

    // Backward pass: assign distances
    for u in (0..n).rev() {
        dt[u] = f_dt(u as isize, s_stack[q], g[s_stack[q] as usize], s);
        if q > 0 && u as isize == t_stack[q] {
            q -= 1;
        }
    }
}

/// Compute the unsigned squared Euclidean distance transform for a 3-D binary volume.
/// `fg[iz*ny*nx + iy*nx + ix] = true` for foreground voxels.
/// `spacing = [sz, sy, sx]` in mm.
/// Returns `Vec<f32>` of Euclidean distances (not squared) in mm.
pub(crate) fn euclidean_dt(fg: &[bool], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    euclidean_dt_with_measure(
        fg,
        dims,
        spacing,
        crate::distance::DistanceMeasure::Euclidean,
    )
}

pub(crate) fn euclidean_dt_with_measure(
    fg: &[bool],
    dims: [usize; 3],
    spacing: [f64; 3],
    measure: crate::distance::DistanceMeasure,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let [sz, sy, sx] = spacing;
    let n_total = nz * ny * nx;

    // Phase 1: 1-D DT along X for each (iz, iy) row — parallel over rows.
    //
    // Each (iz, iy) row is an independent nx-element contiguous chunk of g1:
    // chunk_idx = iz * ny + iy, so iz = chunk_idx / ny, iy = chunk_idx % ny.
    //
    // Safety: `fg` is `&[bool]` (Sync); each closure writes to a disjoint
    // nx-element chunk of g1 — no aliasing across threads.
    let mut g1 = vec![INF; n_total];
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut g1,
        nx,
        |chunk_idx, out_chunk| {
            let iz = chunk_idx / ny;
            let iy = chunk_idx % ny;
            let base = iz * ny * nx + iy * nx;
            let row = &fg[base..(base + nx)];
            let mut phase1_buf = vec![0.0f64; nx];
            phase1_row(row, nx, sx, &mut phase1_buf);
            out_chunk.copy_from_slice(&phase1_buf[..nx]);
        },
    );

    // Phase 2: parabolic envelope along Y for each (iz, ix) column — parallel over z-slices.
    //
    // Each z-slice is an independent ny*nx-element contiguous chunk of g2:
    // chunk_idx = iz; within the closure all nx Y-columns for that slice are processed.
    //
    // Safety: `g1` is `&[f64]` (Sync); each closure writes to a disjoint
    // ny*nx-element chunk of g2 — no aliasing across threads.
    let mut g2 = vec![INF; n_total];
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut g2,
        ny * nx,
        |iz, out_slice| {
            let mut col_buf = vec![0.0f64; ny];
            let mut dt_buf = vec![0.0f64; ny];
            let mut s_stack = vec![0isize; ny];
            let mut t_stack = vec![0isize; ny];
            for ix in 0..nx {
                // Gather Y-column from g1 for this (iz, ix)
                for iy in 0..ny {
                    col_buf[iy] = g1[iz * ny * nx + iy * nx + ix];
                }
                meijster_row(
                    &col_buf[..ny],
                    ny,
                    sy,
                    &mut s_stack,
                    &mut t_stack,
                    &mut dt_buf,
                );
                // Scatter results into this z-slice's output chunk
                for iy in 0..ny {
                    out_slice[iy * nx + ix] = dt_buf[iy];
                }
            }
        },
    );

    // Phase 3: parabolic envelope along Z for each (iy, ix) column — parallel.
    //
    // Strategy: transpose g2 from [nz, ny, nx] to [ny*nx, nz] layout so that
    // each Z-column is a contiguous nz-element chunk in g2_t. moirai then
    // processes ny*nx independent columns in parallel. The result is stored in
    // edt2_t (same layout), then scattered back to z-major order.
    //
    // Safety: each closure receives a disjoint nz-element chunk of edt2_t
    // (assigned by moirai's chunk logic); g2_t is captured immutably.
    let n_cols = ny * nx;
    // Forward transpose: g2_t[col*nz + iz] = g2[iz*n_cols + col]
    let mut g2_t = vec![INF; n_cols * nz];
    for iz in 0..nz {
        for col in 0..n_cols {
            g2_t[col * nz + iz] = g2[iz * n_cols + col];
        }
    }
    // Parallel Z-column processing: output in the same transposed layout (edt2_t)
    let mut edt2_t = vec![0.0f64; n_cols * nz];
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut edt2_t,
        nz,
        |col, out_chunk| {
            let col_data = &g2_t[col * nz..(col + 1) * nz];
            let mut dt_buf = vec![0.0f64; nz];
            let mut s_stack = vec![0isize; nz];
            let mut t_stack = vec![0isize; nz];
            meijster_row(col_data, nz, sz, &mut s_stack, &mut t_stack, &mut dt_buf);
            out_chunk.copy_from_slice(&dt_buf[..nz]);
        },
    );
    // Scatter into z-major order, applying the selected measure once at the
    // operation boundary.
    (0..n_total)
        .map(|flat| {
            let iz = flat / n_cols;
            let col = flat % n_cols;
            let squared = edt2_t[col * nz + iz];
            match measure {
                crate::distance::DistanceMeasure::Euclidean => squared.sqrt() as f32,
                crate::distance::DistanceMeasure::Squared => squared as f32,
            }
        })
        .collect()
}
