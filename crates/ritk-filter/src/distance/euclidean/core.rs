//! Meijster–Roerdink–Hesselink (2000) core EDT routines.
//!
//! Pure mathematical functions: no image I/O, no burn dependency.
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
    let [nz, ny, nx] = dims;
    let [sz, sy, sx] = spacing;
    let n_total = nz * ny * nx;

    // Pre-allocate scratch buffers (one allocation each, reused across all iterations)
    let max_col = ny.max(nz);
    let mut col_buf = vec![0.0f64; max_col];
    let mut dt_buf = vec![0.0f64; max_col];
    let mut s_stack = vec![0isize; max_col];
    let mut t_stack = vec![0isize; max_col];
    let mut phase1_buf = vec![0.0f64; nx];

    // Phase 1: 1-D DT along X for each (iz, iy) row
    let mut g1 = vec![INF; n_total];
    for iz in 0..nz {
        for iy in 0..ny {
            let base = iz * ny * nx + iy * nx;
            // fg row is contiguous — pass the slice directly, no allocation
            let row = &fg[base..(base + nx)];
            phase1_row(row, nx, sx, &mut phase1_buf);
            g1[base..(base + nx)].copy_from_slice(&phase1_buf[..nx]);
        }
    }

    // Phase 2: parabolic envelope along Y for each (iz, ix) column
    let mut g2 = vec![INF; n_total];
    for iz in 0..nz {
        for ix in 0..nx {
            // Gather column into pre-allocated buffer
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
            // Scatter results back
            for iy in 0..ny {
                g2[iz * ny * nx + iy * nx + ix] = dt_buf[iy];
            }
        }
    }

    // Phase 3: parabolic envelope along Z for each (iy, ix) column
    let mut edt2 = vec![INF; n_total];
    for iy in 0..ny {
        for ix in 0..nx {
            // Gather column into pre-allocated buffer
            for iz in 0..nz {
                col_buf[iz] = g2[iz * ny * nx + iy * nx + ix];
            }
            meijster_row(
                &col_buf[..nz],
                nz,
                sz,
                &mut s_stack,
                &mut t_stack,
                &mut dt_buf,
            );
            // Scatter results back
            for iz in 0..nz {
                edt2[iz * ny * nx + iy * nx + ix] = dt_buf[iz];
            }
        }
    }

    edt2.iter().map(|&v| v.sqrt() as f32).collect()
}
