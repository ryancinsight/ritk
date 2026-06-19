//! Level-set reinitialization to a signed distance function
//! (`itk::ReinitializeLevelSetImageFilter` / `sitk.ReinitializeLevelSet`).
//!
//! # Mathematical Specification
//!
//! Deterministic (NOT an iterative PDE). The zero level set of `φ` is located by
//! a neighbourhood crossing extractor: a voxel adjacent to a sign change of
//! `φ − level` is a fast-marching seed whose sub-pixel arrival time is
//!
//! ```text
//! distⱼ = centerVal / (centerVal − neighVal) · spacingⱼ   (nearest crossing along axis j)
//! trial = 1 / √( Σⱼ 1/distⱼ² )                            (multi-axis crossing distance)
//! ```
//!
//! Then [`FastMarchingFilter`](crate::FastMarchingFilter) propagates unit speed
//! from the *outside* seeds (`centerVal > 0`) and the *inside* seeds
//! (`centerVal ≤ 0`); the output is `+T_out` on outside voxels and `−T_in` on
//! inside voxels — a signed distance to the level set. Float-exact to SimpleITK.

use anyhow::Result;
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

use crate::FastMarchingFilter;

/// Reinitialize a level-set image to a signed distance function.
#[derive(Debug, Clone, Default)]
pub struct ReinitializeLevelSetFilter {
    /// Iso-value defining the zero level set (ITK/sitk default 0.0).
    pub level_set_value: f64,
}

impl ReinitializeLevelSetFilter {
    /// Construct for the given level-set value.
    pub fn new(level_set_value: f64) -> Self {
        Self { level_set_value }
    }

    /// Reinitialize the level set.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let sp = image.spacing(); // [sz, sy, sx]
        let level = self.level_set_value;
        let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;
        let cval = |i: usize| vals[i] as f64 - level;

        let (mut out_pts, mut out_vals) = (Vec::new(), Vec::new());
        let (mut in_pts, mut in_vals) = (Vec::new(), Vec::new());

        // Per-axis (dz, dy, dx) and its spacing index.
        let axes: [([i64; 3], usize); 3] = [([1, 0, 0], 0), ([0, 1, 0], 1), ([0, 0, 1], 2)];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = idx(z, y, x);
                    let cv = cval(i);
                    if cv == 0.0 {
                        in_pts.push([z, y, x]);
                        in_vals.push(0.0);
                        continue;
                    }
                    let inside = cv <= 0.0;
                    let mut node = [f64::MAX; 3];
                    for (j, (d, spj)) in axes.iter().enumerate() {
                        let mut best = f64::MAX;
                        for s in [-1i64, 1] {
                            let (zz, yy, xx) = (
                                z as i64 + s * d[0],
                                y as i64 + s * d[1],
                                x as i64 + s * d[2],
                            );
                            if zz < 0
                                || yy < 0
                                || xx < 0
                                || zz >= nz as i64
                                || yy >= ny as i64
                                || xx >= nx as i64
                            {
                                continue;
                            }
                            let nv = cval(idx(zz as usize, yy as usize, xx as usize));
                            if (nv > 0.0 && inside) || (nv < 0.0 && !inside) {
                                let dist = cv / (cv - nv) * sp[*spj];
                                if best > dist {
                                    best = dist;
                                }
                            }
                        }
                        node[j] = best;
                    }
                    node.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mut dsum = 0.0;
                    for &nd in &node {
                        if nd >= f64::MAX {
                            break;
                        }
                        dsum += 1.0 / (nd * nd);
                    }
                    if dsum == 0.0 {
                        continue; // not adjacent to a crossing
                    }
                    let trial = 1.0 / dsum.sqrt();
                    if inside {
                        in_pts.push([z, y, x]);
                        in_vals.push(trial);
                    } else {
                        out_pts.push([z, y, x]);
                        out_vals.push(trial);
                    }
                }
            }
        }

        // Unit-speed fast marching from each seed set.
        let speed = rebuild(vec![1.0f32; n], dims, image);
        let mut out = vec![0.0f32; n];
        if !out_pts.is_empty() {
            let t = FastMarchingFilter {
                trial_points: out_pts,
                initial_trial_values: out_vals,
                normalization_factor: 1.0,
                stopping_value: f64::MAX / 2.0,
            }
            .apply(&speed);
            let (tv, _) = extract_vec_infallible(&t);
            for i in 0..n {
                if cval(i) > 0.0 {
                    out[i] = tv[i];
                }
            }
        }
        if !in_pts.is_empty() {
            let t = FastMarchingFilter {
                trial_points: in_pts,
                initial_trial_values: in_vals,
                normalization_factor: 1.0,
                stopping_value: f64::MAX / 2.0,
            }
            .apply(&speed);
            let (tv, _) = extract_vec_infallible(&t);
            for i in 0..n {
                if cval(i) <= 0.0 {
                    out[i] = -tv[i];
                }
            }
        }
        Ok(rebuild(out, dims, image))
    }
}

#[cfg(test)]
#[path = "tests_reinitialize_level_set.rs"]
mod tests_reinitialize_level_set;
