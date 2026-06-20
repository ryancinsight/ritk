//! Summed-area tables for O(1) local CC window statistics.
//!
//! # Numerical contract
//! Inputs must be approximately [0, 1]-normalized (typical for SyN after
//! intensity normalization). For f64 SATs on [0,1] data, the König–Huygens
//! cancellation error is bounded by 2·cnt·ε_f64 ≈ 2·343·2.2×10⁻¹⁶ ≈ 1.5×10⁻¹³,
//! well below the 1×10⁻¹⁰ force guard used by callers.

/// 3-D summed-area tables for local cross-correlation window queries.
///
/// Build once from `(i_w, j_w)` at construction; then query any window
/// in O(1) with `query_at`. Replaces the two-pass O(w³) `window_cc_stats`.
///
/// # Memory
/// 5 × (nz+2r)(ny+2r)(nx+2r) f64 values.
/// At r=3, 192³ input: ~315 MB. Caller should drop after the force pass.
pub(crate) struct CcSats {
    sat_f: Vec<f64>,
    sat_m: Vec<f64>,
    sat_f2: Vec<f64>,
    sat_m2: Vec<f64>,
    sat_fm: Vec<f64>,
    /// SAT dimensions = (pnz+1, pny+1, pnx+1) where pnd = dims[d] + 2*r
    sdims: [usize; 3],
    r: usize,
    cnt: f64,
}

impl CcSats {
    /// Build all 5 SATs from `i_w` and `j_w` (both length nz*ny*nx, z-major).
    ///
    /// `r` is the Chebyshev radius (window side = 2r+1).
    pub(crate) fn build(i_w: &[f32], j_w: &[f32], dims: [usize; 3], r: usize) -> Self {
        let [nz, ny, nx] = dims;
        let w = 2 * r + 1;
        let cnt = (w * w * w) as f64;

        // Padded dimensions: each axis has r extra on each side.
        let pnz = nz + 2 * r;
        let pny = ny + 2 * r;
        let pnx = nx + 2 * r;

        // Phase A: build padded channel arrays (replicate-clamp padding).
        let padded_size = pnz * pny * pnx;
        let mut pad_f = vec![0.0_f64; padded_size];
        let mut pad_m = vec![0.0_f64; padded_size];
        let mut pad_f2 = vec![0.0_f64; padded_size];
        let mut pad_m2 = vec![0.0_f64; padded_size];
        let mut pad_fm = vec![0.0_f64; padded_size];

        for kz in 0..pnz {
            let sz = kz.saturating_sub(r).min(nz - 1);
            for ky in 0..pny {
                let sy = ky.saturating_sub(r).min(ny - 1);
                for kx in 0..pnx {
                    let sx = kx.saturating_sub(r).min(nx - 1);
                    let src = sz * ny * nx + sy * nx + sx;
                    let dst = kz * pny * pnx + ky * pnx + kx;
                    let f = i_w[src] as f64;
                    let m = j_w[src] as f64;
                    pad_f[dst] = f;
                    pad_m[dst] = m;
                    pad_f2[dst] = f * f;
                    pad_m2[dst] = m * m;
                    pad_fm[dst] = f * m;
                }
            }
        }

        // Phase B: build 1-based SATs. Allocate (pnz+1)*(pny+1)*(pnx+1).
        let snz = pnz + 1;
        let sny = pny + 1;
        let snx = pnx + 1;
        let sat_size = snz * sny * snx;

        let build_sat = |pad: &[f64]| -> Vec<f64> {
            let mut sat = vec![0.0_f64; sat_size];
            // Copy into 1-based positions: sat[z+1, y+1, x+1] = pad[z, y, x].
            for kz in 0..pnz {
                for ky in 0..pny {
                    for kx in 0..pnx {
                        sat[(kz + 1) * sny * snx + (ky + 1) * snx + (kx + 1)] =
                            pad[kz * pny * pnx + ky * pnx + kx];
                    }
                }
            }
            // X prefix pass (innermost dimension, contiguous in memory).
            for z in 1..=pnz {
                for y in 1..=pny {
                    for x in 1..=pnx {
                        sat[z * sny * snx + y * snx + x] += sat[z * sny * snx + y * snx + x - 1];
                    }
                }
            }
            // Y prefix pass.
            for z in 1..=pnz {
                for x in 1..=pnx {
                    for y in 1..=pny {
                        sat[z * sny * snx + y * snx + x] += sat[z * sny * snx + (y - 1) * snx + x];
                    }
                }
            }
            // Z prefix pass (outermost dimension).
            for y in 1..=pny {
                for x in 1..=pnx {
                    for z in 1..=pnz {
                        sat[z * sny * snx + y * snx + x] += sat[(z - 1) * sny * snx + y * snx + x];
                    }
                }
            }
            sat
        };

        let sat_f = build_sat(&pad_f);
        let sat_m = build_sat(&pad_m);
        let sat_f2 = build_sat(&pad_f2);
        let sat_m2 = build_sat(&pad_m2);
        let sat_fm = build_sat(&pad_fm);

        Self {
            sat_f,
            sat_m,
            sat_f2,
            sat_m2,
            sat_fm,
            sdims: [snz, sny, snx],
            r,
            cnt,
        }
    }

    /// O(1) query matching the `window_cc_stats` interface exactly.
    ///
    /// In padded space, original voxel `(iz, iy, ix)` maps to center
    /// `(iz+r, iy+r, ix+r)`. The window `[iz..=iz+2r] × [iy..=iy+2r] ×
    /// [ix..=ix+2r]` in padded coordinates is always interior (no boundary
    /// checks needed), because of the `r`-wide replicate border.
    ///
    /// Returns `(mu_i, mu_j, cc_numerator, var_i, var_j, count)` — identical
    /// convention to `window_cc_stats`.
    #[inline]
    pub(crate) fn query_at(
        &self,
        iz: usize,
        iy: usize,
        ix: usize,
    ) -> (f64, f64, f64, f64, f64, u32) {
        let r = self.r;
        let [_snz, sny, snx] = self.sdims;

        // Window in padded SAT 1-based coordinates.
        let (z0, z1) = (iz, iz + 2 * r);
        let (y0, y1) = (iy, iy + 2 * r);
        let (x0, x1) = (ix, ix + 2 * r);

        let box_q = |sat: &[f64]| -> f64 {
            // 3-D inclusion-exclusion on 1-based SAT.
            macro_rules! s {
                ($z:expr, $y:expr, $x:expr) => {
                    sat[$z * sny * snx + $y * snx + $x]
                };
            }
            s!(z1 + 1, y1 + 1, x1 + 1)
                - s!(z0, y1 + 1, x1 + 1)
                - s!(z1 + 1, y0, x1 + 1)
                - s!(z1 + 1, y1 + 1, x0)
                + s!(z0, y0, x1 + 1)
                + s!(z0, y1 + 1, x0)
                + s!(z1 + 1, y0, x0)
                - s!(z0, y0, x0)
        };

        let sum_f = box_q(&self.sat_f);
        let sum_m = box_q(&self.sat_m);
        let sum_f2 = box_q(&self.sat_f2);
        let sum_m2 = box_q(&self.sat_m2);
        let sum_fm = box_q(&self.sat_fm);

        let cnt = self.cnt;
        let mu_f = sum_f / cnt;
        let mu_m = sum_m / cnt;

        // König–Huygens form. .max(0.0) absorbs residual f64 rounding toward −ε.
        let vi = (sum_f2 - cnt * mu_f * mu_f).max(0.0);
        let vj = (sum_m2 - cnt * mu_m * mu_m).max(0.0);
        let num = sum_fm - cnt * mu_f * mu_m;

        (mu_f, mu_m, num, vi, vj, cnt as u32)
    }
}

#[cfg(test)]
mod tests_sat {
    use super::*;

    /// SAT `query_at` must produce results within 1e-9 of the two-pass
    /// `window_cc_stats` for interior voxels on [0,1]-normalized data.
    ///
    /// Evidence tier: differential equivalence (two independent algorithms
    /// computing the same statistical quantity).
    #[test]
    fn test_sat_matches_two_pass_interior() {
        use crate::diffeomorphic::local_cc::window_cc_stats;
        let dims = [8_usize, 8, 8];
        let n = 8 * 8 * 8;
        let i_w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.003) % 1.0).collect();
        let j_w: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.007) + 0.1) % 1.0).collect();
        let r = 3;
        let sats = CcSats::build(&i_w, &j_w, dims, r);

        // Check several interior voxels (at least r away from boundary).
        for &(iz, iy, ix) in &[(3, 3, 3), (4, 4, 4), (5, 3, 4)] {
            let (mu_i_s, mu_j_s, num_s, vi_s, vj_s, cnt_s) = sats.query_at(iz, iy, ix);
            let (mu_i_r, mu_j_r, num_r, vi_r, vj_r, cnt_r) =
                window_cc_stats(&i_w, &j_w, dims, iz, iy, ix, r as isize);

            assert_eq!(cnt_s, cnt_r, "count mismatch at ({iz},{iy},{ix})");
            assert!(
                (mu_i_s - mu_i_r).abs() < 1e-9,
                "mu_i diverges at ({iz},{iy},{ix}): sat={mu_i_s:.12}, ref={mu_i_r:.12}"
            );
            assert!(
                (mu_j_s - mu_j_r).abs() < 1e-9,
                "mu_j diverges at ({iz},{iy},{ix}): sat={mu_j_s:.12}, ref={mu_j_r:.12}"
            );
            assert!(
                (num_s - num_r).abs() < 1e-9,
                "num diverges at ({iz},{iy},{ix}): sat={num_s:.12}, ref={num_r:.12}"
            );
            assert!(
                (vi_s - vi_r).abs() < 1e-9,
                "vi diverges at ({iz},{iy},{ix}): sat={vi_s:.12}, ref={vi_r:.12}"
            );
            assert!(
                (vj_s - vj_r).abs() < 1e-9,
                "vj diverges at ({iz},{iy},{ix}): sat={vj_s:.12}, ref={vj_r:.12}"
            );
        }
    }

    /// SAT `query_at` must match two-pass for boundary voxels (replicate padding
    /// must reproduce the clamped-index semantics of `window_cc_stats`).
    #[test]
    fn test_sat_matches_two_pass_boundary() {
        use crate::diffeomorphic::local_cc::window_cc_stats;
        let dims = [8_usize, 8, 8];
        let n = 8 * 8 * 8;
        let i_w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.003) % 1.0).collect();
        let j_w: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.007) + 0.1) % 1.0).collect();
        let r = 3;
        let sats = CcSats::build(&i_w, &j_w, dims, r);

        // Corner and edge voxels.
        for &(iz, iy, ix) in &[(0, 0, 0), (7, 7, 7), (0, 7, 3)] {
            let (mu_i_s, mu_j_s, num_s, vi_s, vj_s, cnt_s) = sats.query_at(iz, iy, ix);
            let (mu_i_r, mu_j_r, num_r, vi_r, vj_r, cnt_r) =
                window_cc_stats(&i_w, &j_w, dims, iz, iy, ix, r as isize);

            assert_eq!(cnt_s, cnt_r, "count mismatch at ({iz},{iy},{ix})");
            assert!(
                (mu_i_s - mu_i_r).abs() < 1e-9,
                "mu_i boundary diverges at ({iz},{iy},{ix}): sat={mu_i_s:.12}, ref={mu_i_r:.12}"
            );
            assert!(
                (mu_j_s - mu_j_r).abs() < 1e-9,
                "mu_j boundary diverges at ({iz},{iy},{ix}): sat={mu_j_s:.12}, ref={mu_j_r:.12}"
            );
            assert!(
                (num_s - num_r).abs() < 1e-9,
                "num boundary diverges at ({iz},{iy},{ix}): sat={num_s:.12}, ref={num_r:.12}"
            );
            assert!(
                (vi_s - vi_r).abs() < 1e-9,
                "vi boundary diverges at ({iz},{iy},{ix}): sat={vi_s:.12}, ref={vi_r:.12}"
            );
            assert!(
                (vj_s - vj_r).abs() < 1e-9,
                "vj boundary diverges at ({iz},{iy},{ix}): sat={vj_s:.12}, ref={vj_r:.12}"
            );
        }
    }
}
