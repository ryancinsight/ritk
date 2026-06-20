//! Anti-alias binary image filter via the ITK SparseField level-set solver.
//!
//! # Mathematical Specification
//!
//! Ports `sitk.AntiAliasBinaryImageFilter` / `itk::AntiAliasBinaryImageFilter`,
//! which derives from `itk::SparseFieldLevelSetImageFilter` with a
//! `CurvatureFlowFunction` difference function. The boundary of a binary object
//! is smoothed by evolving a narrow-band signed level-set under mean curvature
//! flow, constrained so the zero crossing never leaves the original boundary
//! band (the per-pixel sign is locked to the input binary).
//!
//! ## Algorithm (faithful SparseField port — bit-exact to sitk)
//!
//! 1. **CopyInputToOutput**: `shifted = input − iso` with `iso = (max+min)/2`;
//!    the active layer is the `ZeroCrossing(shifted)` set (the voxels on the
//!    near-zero side of the boundary, float-exact to `itk::ZeroCrossingImageFilter`).
//! 2. **Initialize**: active-layer values = `clamp(shifted / |∇shifted|_upwind, ±½)`
//!    (`InitializeActiveLayerValues`); outer layers ±1, ±2 are constructed
//!    (`ConstructLayer`) and value-propagated from the active seed
//!    (`PropagateLayerValues`, value = nearest-to-zero neighbour ∓ 1).
//! 3. **Iterate** (`ApplyUpdate`) until `rms < max_rms_error` or `number_of_iterations`:
//!    - `CalculateChange`: per active voxel, `CurvatureFlowFunction::ComputeUpdate`
//!      = `(Σ_i φ_i²·(Σ_{j≠i} φ_jj) − 2·Σ_{i<j} φ_i·φ_j·φ_ij) / |∇φ|²` (0 if `|∇φ|² < ε`).
//!    - `UpdateActiveLayerValues`: `new = constraint(φ + Δt·change)` with the
//!      AntiAlias constraint (foreground → `max(new,0)`, background → `min(new,0)`);
//!      voxels whose new value crosses ±½ are promoted/demoted, pulling their
//!      inner/outer neighbours into the active set.
//!    - `ProcessStatusList` cascade moves voxels between layers (consuming the
//!      work lists), `PropagateAllLayerValues` re-seeds the outer layers, and
//!      orphaned outer voxels are node-deleted (status → background).
//! 4. **PostProcessOutput**: background voxels → `±(NumberOfLayers+1)`.
//!
//! `NumberOfLayers` is 2 for 2-D inputs (`nz == 1`) and 3 for 3-D, matching ITK.
//! `Δt = 0.05` is the ITK `CurvatureFlowImageFilter` default time step.
//!
//! Validated bit-exact (max-err 0.0) against `sitk.AntiAliasBinary` across square,
//! circle, L-shape and jagged inputs over 1–27 iterations / varying `maxRMSError`.
//!
//! ## References
//! - Whitaker, R.T. (2000). "Reducing aliasing artifacts in iso-surfaces of binary
//!   volumes." *Proc. IEEE Vis. Symp. Vol. Vis.*, pp. 23–32.
//! - ITK `itkAntiAliasBinaryImageFilter.hxx`, `itkSparseFieldLevelSetImageFilter.hxx`,
//!   `itkCurvatureFlowFunction.hxx`, `itkZeroCrossingImageFilter.hxx`.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

// ── Constants ─────────────────────────────────────────────────────────────────

/// ITK `CurvatureFlowImageFilter` default explicit-Euler time step.
const DT: f32 = 0.05;
/// `m_ConstantGradientValue` (unit spacing).
const CGV: f32 = 1.0;
/// Gradient-magnitude-squared floor preventing 0/0 in flat regions.
const MSQ_EPS: f32 = 1e-9;

// Status sentinels (non-layer states are negative; layer indices are 0..num).
const ST_NULL: i32 = -1;
const ST_CHG: i32 = -2;
const ST_CUP: i32 = -3;
const ST_CDN: i32 = -4;

// ── Filter ────────────────────────────────────────────────────────────────────

/// Anti-alias binary image filter (faithful ITK SparseField solver).
///
/// Smooths the boundary of a binary object, returning the signed level-set φ
/// (negative inside the smoothed object, positive outside; the zero crossing is
/// the anti-aliased sub-voxel boundary). Bit-exact to `sitk.AntiAliasBinary`.
///
/// # Defaults
/// - `max_rms_error = 0.07` (ITK default)
/// - `number_of_iterations = 1000` (ITK default)
#[derive(Debug, Clone)]
pub struct AntiAliasBinaryImageFilter {
    /// Per-voxel RMS change threshold for early termination (ITK default 0.07).
    pub max_rms_error: f32,
    /// Maximum number of level-set evolution iterations (ITK default 1000).
    pub number_of_iterations: usize,
}

impl Default for AntiAliasBinaryImageFilter {
    fn default() -> Self {
        Self {
            max_rms_error: 0.07,
            number_of_iterations: 1000,
        }
    }
}

impl AntiAliasBinaryImageFilter {
    /// Evolve the binary boundary under the SparseField mean-curvature solver.
    ///
    /// `image`: binary float32 (foreground == max value, background == min),
    /// shape `[nz, ny, nx]` (`nz == 1` is treated as a 2-D image, matching sitk).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (binary, dims) = extract_vec_infallible(image);
        let out = self.run(&binary, dims);
        rebuild(out, dims, image)
    }

    fn run(&self, binary: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        if n == 0 {
            return Vec::new();
        }
        let ndim = if nz == 1 { 2 } else { 3 };
        let nl = ndim; // NumberOfLayers
        let num = 2 * nl + 1; // m_Layers.size()
        let bg_val = (nl + 1) as f32;

        // Face-neighbour offsets in m_NeighborList order. In-plane first so the
        // 2-D (nz==1) case matches the validated 2-D ordering; z last (out of
        // bounds and skipped when nz==1).
        let mut offsets: Vec<(isize, isize, isize)> =
            vec![(0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0)];
        if ndim == 3 {
            offsets.push((-1, 0, 0));
            offsets.push((1, 0, 0));
        }

        let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
        let decode = |f: usize| -> (usize, usize, usize) {
            let iz = f / (ny * nx);
            let r = f % (ny * nx);
            (iz, r / nx, r % nx)
        };
        // Neighbour flat index if in bounds.
        let neighbor = |f: usize, off: (isize, isize, isize)| -> Option<usize> {
            let (iz, iy, ix) = decode(f);
            let (z, y, x) = (iz as isize + off.0, iy as isize + off.1, ix as isize + off.2);
            if z >= 0
                && y >= 0
                && x >= 0
                && z < nz as isize
                && y < ny as isize
                && x < nx as isize
            {
                Some(idx(z as usize, y as usize, x as usize))
            } else {
                None
            }
        };

        // iso = (max+min)/2 (MinimumMaximumImageCalculator).
        let (mn, mx) = binary
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        let iso = mx - (mx - mn) / 2.0;
        let upper_bin = mx; // foreground value
        let shifted: Vec<f32> = binary.iter().map(|&v| v - iso).collect();

        // ZeroCrossing(shifted) → active set, float-exact to ITK.
        let sign_change = |a: f32, b: f32| (a * b < 0.0) || ((a == 0.0) != (b == 0.0));
        let mut is_active = vec![false; n];
        for f in 0..n {
            let v = shifted[f];
            let av = v.abs();
            let mut crosses = false;
            // forward (+) accept |v|<=|nv|; backward (−) strict |v|<|nv|.
            for &off in &offsets {
                // forward
                if let Some(g) = neighbor(f, off) {
                    let nv = shifted[g];
                    let forward = off.0 + off.1 + off.2 > 0;
                    if sign_change(v, nv) && (if forward { av <= nv.abs() } else { av < nv.abs() })
                    {
                        crosses = true;
                        break;
                    }
                }
            }
            is_active[f] = crosses;
        }

        // State.
        let mut status = vec![ST_NULL; n];
        let mut phi: Vec<f32> = shifted.iter().map(|&s| if s > 0.0 { bg_val } else { -bg_val }).collect();
        let mut layers: Vec<Vec<usize>> = vec![Vec::new(); num as usize];

        // add: set status + push-front into its layer list.
        macro_rules! push_layer {
            ($f:expr, $s:expr) => {{
                let s: i32 = $s;
                status[$f] = s;
                if s >= 0 {
                    layers[s as usize].insert(0, $f);
                }
            }};
        }

        // ConstructActiveLayer (raster scan): active voxels → layer 0; their
        // non-active neighbours → first inside(1, shifted<0) / outside(2) layer.
        for f in 0..n {
            if is_active[f] {
                push_layer!(f, 0);
                for &off in &offsets {
                    if let Some(g) = neighbor(f, off) {
                        if !is_active[g] && status[g] == ST_NULL {
                            let ln = if shifted[g] < 0.0 { 1 } else { 2 };
                            push_layer!(g, ln);
                        }
                    }
                }
            }
        }
        // ConstructLayer i → i+2.
        for i in 1..(num - 2) {
            let cur: Vec<usize> = layers[i as usize].clone();
            for f in cur {
                for &off in &offsets {
                    if let Some(g) = neighbor(f, off) {
                        if status[g] == ST_NULL {
                            push_layer!(g, i + 2);
                        }
                    }
                }
            }
        }
        // InitializeActiveLayerValues: clamp(shifted/upwind_len, ±CGV/2).
        let cf = CGV / 2.0;
        for &f in &layers[0].clone() {
            let c = shifted[f];
            let mut l2 = 0.0f32;
            for &off in &offsets {
                // forward/backward one-sided differences, larger magnitude wins.
                let fwd = neighbor(f, off).map(|g| shifted[g]).unwrap_or(c) - c;
                let back = c - neighbor(f, (-off.0, -off.1, -off.2)).map(|g| shifted[g]).unwrap_or(c);
                let d = if fwd.abs() > back.abs() { fwd } else { back };
                // Only count each axis once: the offsets list has +/- pairs; use
                // forward offsets (positive direction) to avoid double-counting.
                if off.0 + off.1 + off.2 > 0 {
                    l2 += d * d;
                }
            }
            let len = l2.sqrt() + 1e-6;
            phi[f] = (c / len).clamp(-cf, cf);
        }

        // PropagateLayerValues / PropagateAllLayerValues.
        let propagate_layer = |layers: &mut Vec<Vec<usize>>,
                                   phi: &mut [f32],
                                   status: &mut [i32],
                                   frm: i32,
                                   to: i32,
                                   promote: i32,
                                   inout: i32| {
            let delta = if inout == 1 { -CGV } else { CGV };
            let mut survivors: Vec<usize> = Vec::new();
            let cur: Vec<usize> = layers[to as usize].clone();
            for f in cur {
                if status[f] != to {
                    continue;
                }
                let mut val = 0.0f32;
                let mut found = false;
                for &off in &offsets {
                    if let Some(g) = neighbor(f, off) {
                        if status[g] == frm {
                            let vt = phi[g];
                            if !found {
                                val = vt;
                            } else if inout == 1 {
                                val = val.max(vt);
                            } else {
                                val = val.min(vt);
                            }
                            found = true;
                        }
                    }
                }
                if found {
                    phi[f] = val + delta;
                    survivors.push(f);
                } else if promote > num - 1 {
                    status[f] = ST_NULL;
                } else {
                    status[f] = promote;
                    layers[promote as usize].insert(0, f);
                }
            }
            layers[to as usize] = survivors;
        };
        macro_rules! propagate_all {
            () => {{
                propagate_layer(&mut layers, &mut phi, &mut status, 0, 1, 3, 1);
                propagate_layer(&mut layers, &mut phi, &mut status, 0, 2, 4, 2);
                for i in 1..(num - 2) {
                    propagate_layer(&mut layers, &mut phi, &mut status, i, i + 2, i + 4, (i + 2) % 2);
                }
            }};
        }
        propagate_all!();

        // CurvatureFlowFunction::ComputeUpdate at flat index f (clamped Neumann).
        let curvature = |phi: &[f32], f: usize| -> f32 {
            let (iz, iy, ix) = decode(f);
            let g = |dz: isize, dy: isize, dx: isize| -> f32 {
                let z = (iz as isize + dz).clamp(0, nz as isize - 1) as usize;
                let y = (iy as isize + dy).clamp(0, ny as isize - 1) as usize;
                let x = (ix as isize + dx).clamp(0, nx as isize - 1) as usize;
                phi[idx(z, y, x)]
            };
            let c = phi[f];
            // first derivatives, second derivatives, cross derivatives (axes y,x[,z]).
            let fx = 0.5 * (g(0, 0, 1) - g(0, 0, -1));
            let fy = 0.5 * (g(0, 1, 0) - g(0, -1, 0));
            let fxx = g(0, 0, 1) - 2.0 * c + g(0, 0, -1);
            let fyy = g(0, 1, 0) - 2.0 * c + g(0, -1, 0);
            let fxy = 0.25 * (g(0, -1, -1) - g(0, -1, 1) - g(0, 1, -1) + g(0, 1, 1));
            if ndim == 2 {
                let msq = fx * fx + fy * fy;
                if msq < MSQ_EPS {
                    return 0.0;
                }
                (fx * fx * fyy + fy * fy * fxx - 2.0 * fx * fy * fxy) / msq
            } else {
                let fz = 0.5 * (g(1, 0, 0) - g(-1, 0, 0));
                let fzz = g(1, 0, 0) - 2.0 * c + g(-1, 0, 0);
                let fxz = 0.25 * (g(-1, 0, -1) - g(-1, 0, 1) - g(1, 0, -1) + g(1, 0, 1));
                let fyz = 0.25 * (g(-1, -1, 0) - g(-1, 1, 0) - g(1, -1, 0) + g(1, 1, 0));
                let msq = fx * fx + fy * fy + fz * fz;
                if msq < MSQ_EPS {
                    return 0.0;
                }
                (fx * fx * (fyy + fzz)
                    + fy * fy * (fxx + fzz)
                    + fz * fz * (fxx + fyy)
                    - 2.0 * fx * fy * fxy
                    - 2.0 * fx * fz * fxz
                    - 2.0 * fy * fz * fyz)
                    / msq
            }
        };

        // ── ApplyUpdate loop ──
        for _ in 0..self.number_of_iterations {
            let al: Vec<usize> = layers[0].clone();
            let update: Vec<f32> = al.iter().map(|&f| curvature(&phi, f)).collect();
            let mut up: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            let mut dn: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            let mut keep: Vec<usize> = Vec::new();
            let mut rms_acc = 0.0f64;
            let mut cnt = 0usize;
            for (k, &f) in al.iter().enumerate() {
                let old = phi[f];
                let mut nv = old + DT * update[k];
                nv = if binary[f] == upper_bin { nv.max(0.0) } else { nv.min(0.0) };
                if nv >= cf {
                    if offsets.iter().any(|&o| neighbor(f, o).is_some_and(|g| status[g] == ST_CDN)) {
                        keep.push(f);
                        continue;
                    }
                    rms_acc += ((nv - old) as f64).powi(2);
                    cnt += 1;
                    let tv = nv - CGV;
                    for &off in &offsets {
                        if let Some(g) = neighbor(f, off) {
                            if status[g] == 1 && (phi[g] < -cf || tv.abs() < phi[g].abs()) {
                                phi[g] = tv;
                            }
                        }
                    }
                    // move f out of active (layers[0]) → CUP
                    status[f] = ST_CUP;
                    up[0].insert(0, f);
                } else if nv < -cf {
                    if offsets.iter().any(|&o| neighbor(f, o).is_some_and(|g| status[g] == ST_CUP)) {
                        keep.push(f);
                        continue;
                    }
                    rms_acc += ((nv - old) as f64).powi(2);
                    cnt += 1;
                    let tv = nv + CGV;
                    for &off in &offsets {
                        if let Some(g) = neighbor(f, off) {
                            if status[g] == 2 && (phi[g] >= cf || tv.abs() < phi[g].abs()) {
                                phi[g] = tv;
                            }
                        }
                    }
                    status[f] = ST_CDN;
                    dn[0].insert(0, f);
                } else {
                    rms_acc += ((nv - old) as f64).powi(2);
                    cnt += 1;
                    phi[f] = nv;
                    keep.push(f);
                }
            }
            layers[0] = keep;

            // move_to: remove f from its old layer, set status, push-front into
            // the new layer (if it is a layer). Mirrors ITK node relinking.
            let move_to = |layers: &mut Vec<Vec<usize>>, status: &mut [i32], f: usize, s: i32| {
                let o = status[f];
                if o >= 0 {
                    if let Some(p) = layers[o as usize].iter().position(|&x| x == f) {
                        layers[o as usize].remove(p);
                    }
                }
                status[f] = s;
                if s >= 0 {
                    layers[s as usize].insert(0, f);
                }
            };
            // ProcessStatusList: consume `inl` from the front (PopFront), move each
            // voxel to `ct`, and mark its `sr`-status neighbours as Changing into a
            // fresh output list (returned).
            let proc = |layers: &mut Vec<Vec<usize>>,
                        status: &mut [i32],
                        mut inl: Vec<usize>,
                        ct: i32,
                        sr: i32|
             -> Vec<usize> {
                let mut outl: Vec<usize> = Vec::new();
                while !inl.is_empty() {
                    let f = inl.remove(0);
                    move_to(layers, status, f, ct);
                    for &off in &offsets {
                        if let Some(g) = neighbor(f, off) {
                            if status[g] == sr {
                                move_to(layers, status, g, ST_CHG);
                                outl.insert(0, g);
                            }
                        }
                    }
                }
                outl
            };

            let mut u = proc(&mut layers, &mut status, std::mem::take(&mut up[0]), 2, 1);
            let mut d = proc(&mut layers, &mut status, std::mem::take(&mut dn[0]), 1, 2);
            let mut up_to = 0i32;
            let mut dn_to = 0i32;
            let mut us = 3i32;
            let mut ds = 4i32;
            while ds < num {
                u = proc(&mut layers, &mut status, u, up_to, us);
                d = proc(&mut layers, &mut status, d, dn_to, ds);
                up_to = if up_to == 0 { 1 } else { up_to + 2 };
                dn_to += 2;
                us += 2;
                ds += 2;
            }
            u = proc(&mut layers, &mut status, u, up_to, ST_NULL);
            d = proc(&mut layers, &mut status, d, dn_to, ST_NULL);
            // ProcessOutsideList: remaining work-list voxels → outermost layers.
            for f in u {
                move_to(&mut layers, &mut status, f, num - 2);
            }
            for f in d {
                move_to(&mut layers, &mut status, f, num - 1);
            }

            propagate_all!();

            let rms = if cnt > 0 {
                (rms_acc / cnt as f64).sqrt() as f32
            } else {
                0.0
            };
            if rms < self.max_rms_error {
                break;
            }
        }

        // PostProcessOutput: background voxels → ±(NL+1) by current sign.
        for f in 0..n {
            if status[f] == ST_NULL {
                phi[f] = if phi[f] > 0.0 { bg_val } else { -bg_val };
            }
        }
        phi
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_anti_alias_binary.rs"]
mod tests_anti_alias_binary;
