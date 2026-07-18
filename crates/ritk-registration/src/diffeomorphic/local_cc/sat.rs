//! Summed-area tables for O(1) local CC window statistics.
//!
//! # Numerical contract
//! Inputs must be approximately [0, 1]-normalized (typical for SyN after
//! intensity normalization). For f64 SATs on [0,1] data, the KÃ¶nigâ€“Huygens
//! cancellation error is bounded by 2Â·cntÂ·Îµ_f64 â‰ˆ 2Â·343Â·2.2Ã—10â»Â¹â¶ â‰ˆ 1.5Ã—10â»Â¹Â³,
//! well below the 1Ã—10â»Â¹â° force guard used by callers.

/// 3-D summed-area tables for local cross-correlation window queries.
///
/// Build once from `(i_w, j_w)` at construction; then query any window
/// in O(1) with `query_at`. Replaces the two-pass O(wÂ³) `window_cc_stats`.
///
/// # Memory
/// 5 Ã— (nz+2r)(ny+2r)(nx+2r) f64 values.
/// At r=3, 192Â³ input: ~315 MB. Iterative callers should allocate once and
/// call [`CcSats::rebuild`] for each new warped-image pair.
pub(crate) struct CcSats {
    sat_f: Vec<f64>,
    sat_m: Vec<f64>,
    sat_f2: Vec<f64>,
    sat_m2: Vec<f64>,
    sat_fm: Vec<f64>,
    /// SAT dimensions = (pnz+1, pny+1, pnx+1) where pnd = dims[d] + 2*r
    sdims: [usize; 3],
    r: usize,
    cnt: f64 }

impl CcSats {
    /// Allocate the five tables for one image shape and window radius.
    pub(crate) fn new(dims: [usize; 3], r: usize) -> Self {
        let [nz, ny, nx] = dims;
        let w = 2 * r + 1;
        let cnt = (w * w * w) as f64;
        let pnz = nz + 2 * r;
        let pny = ny + 2 * r;
        let pnx = nx + 2 * r;
        let snz = pnz + 1;
        let sny = pny + 1;
        let snx = pnx + 1;
        let sat_size = snz * sny * snx;

        Self {
            sat_f: vec![0.0_f64; sat_size],
            sat_m: vec![0.0_f64; sat_size],
            sat_f2: vec![0.0_f64; sat_size],
            sat_m2: vec![0.0_f64; sat_size],
            sat_fm: vec![0.0_f64; sat_size],
            sdims: [snz, sny, snx],
            r,
            cnt }
    }

    /// Build all five tables from `i_w` and `j_w`.
    #[cfg(test)]
    pub(crate) fn build(i_w: &[f32], j_w: &[f32], dims: [usize; 3], r: usize) -> Self {
        let mut tables = Self::new(dims, r);
        tables.rebuild(i_w, j_w, dims);
        tables
    }

    /// Rebuild previously allocated tables for a new image pair of the same shape.
    pub(crate) fn rebuild(&mut self, i_w: &[f32], j_w: &[f32], dims: [usize; 3]) {
        let [nz, ny, nx] = dims;
        debug_assert_eq!(i_w.len(), nz * ny * nx);
        debug_assert_eq!(j_w.len(), nz * ny * nx);
        let [snz, sny, snx] = self.sdims;
        let pnz = snz - 1;
        let pny = sny - 1;
        let pnx = snx - 1;
        debug_assert_eq!(pnz, nz + 2 * self.r);
        debug_assert_eq!(pny, ny + 2 * self.r);
        debug_assert_eq!(pnx, nx + 2 * self.r);

        self.sat_f.fill(0.0);
        self.sat_m.fill(0.0);
        self.sat_f2.fill(0.0);
        self.sat_m2.fill(0.0);
        self.sat_fm.fill(0.0);

        // Build all five 1-based SATs in four fused, contiguous traversals:
        // input copy followed by one prefix pass per axis. This eliminates five
        // padded volumes and replaces twenty channel-specific passes while
        // retaining replicate-clamp boundary values and summation order.
        let r = self.r;

        for z in 1..=pnz {
            let src_z = (z - 1).saturating_sub(r).min(nz - 1);
            for y in 1..=pny {
                let src_y = (y - 1).saturating_sub(r).min(ny - 1);
                for x in 1..=pnx {
                    let src_x = (x - 1).saturating_sub(r).min(nx - 1);
                    let src = src_z * ny * nx + src_y * nx + src_x;
                    let f = i_w[src] as f64;
                    let m = j_w[src] as f64;
                    let index = z * sny * snx + y * snx + x;
                    self.sat_f[index] = f;
                    self.sat_m[index] = m;
                    self.sat_f2[index] = f * f;
                    self.sat_m2[index] = m * m;
                    self.sat_fm[index] = f * m;
                }
            }
        }

        macro_rules! accumulate_channels {
            ($index:expr, $previous:expr) => {
                self.sat_f[$index] += self.sat_f[$previous];
                self.sat_m[$index] += self.sat_m[$previous];
                self.sat_f2[$index] += self.sat_f2[$previous];
                self.sat_m2[$index] += self.sat_m2[$previous];
                self.sat_fm[$index] += self.sat_fm[$previous];
            };
        }
        for z in 1..=pnz {
            for y in 1..=pny {
                for x in 1..=pnx {
                    let index = z * sny * snx + y * snx + x;
                    accumulate_channels!(index, index - 1);
                }
            }
        }
        for z in 1..=pnz {
            for y in 1..=pny {
                for x in 1..=pnx {
                    let index = z * sny * snx + y * snx + x;
                    accumulate_channels!(index, index - snx);
                }
            }
        }
        let plane = sny * snx;
        for z in 1..=pnz {
            for y in 1..=pny {
                for x in 1..=pnx {
                    let index = z * plane + y * snx + x;
                    accumulate_channels!(index, index - plane);
                }
            }
        }
    }

    /// O(1) query matching the `window_cc_stats` interface exactly.
    ///
    /// In padded space, original voxel `(iz, iy, ix)` maps to center
    /// `(iz+r, iy+r, ix+r)`. The window `[iz..=iz+2r] Ã— [iy..=iy+2r] Ã—
    /// [ix..=ix+2r]` in padded coordinates is always interior (no boundary
    /// checks needed), because of the `r`-wide replicate border.
    ///
    /// Returns `(mu_i, mu_j, cc_numerator, var_i, var_j, count)` â€” identical
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

        // KÃ¶nigâ€“Huygens form. .max(0.0) absorbs residual f64 rounding toward âˆ’Îµ.
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

    #[test]
    fn rebuilt_tables_match_fresh_construction() {
        let dims = [5, 6, 7];
        let n = dims.iter().product();
        let first_i: Vec<f32> = (0..n).map(|index| index as f32 / n as f32).collect();
        let first_j: Vec<f32> = first_i.iter().rev().copied().collect();
        let second_i: Vec<f32> = first_i.iter().map(|value| 1.0 - value).collect();
        let second_j: Vec<f32> = first_j.iter().map(|value| value * value).collect();
        let mut reused = CcSats::new(dims, 2);

        reused.rebuild(&first_i, &first_j, dims);
        reused.rebuild(&second_i, &second_j, dims);
        let fresh = CcSats::build(&second_i, &second_j, dims, 2);

        for z in 0..dims[0] {
            for y in 0..dims[1] {
                for x in 0..dims[2] {
                    assert_eq!(reused.query_at(z, y, x), fresh.query_at(z, y, x));
                }
            }
        }
    }
}
