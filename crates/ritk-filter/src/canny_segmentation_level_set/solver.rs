//! The core SparseField solver loop for Canny segmentation level sets.

use super::{CannySegmentationLevelSet, CGV, GRAD_EPS, MIN_NORM, ST_CDN, ST_CHG, ST_CUP, ST_NULL};

impl CannySegmentationLevelSet {
    pub(crate) fn run(&self, shifted: &[f64], p: &[f64], adv: &[Vec<f64>], dims: [usize; 3]) -> Vec<f64> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let ndim = if nz == 1 { 2 } else { 3 };
        let nl: i32 = 2; // SparseField default (NOT overridden for segmentation).
        let num = 2 * nl + 1;
        let bg_val = (nl + 1) as f64;
        let cf = CGV / 2.0;
        let wave_dt = 1.0 / (2.0 * ndim as f64);

        let curv_w = self.curvature_scaling as f64;
        let prop_w = self.propagation_scaling as f64;
        let adv_w = self.advection_scaling as f64;

        // Face-neighbour offsets (in-plane first; z last for 3-D), matching the
        // validated AntiAlias ordering.
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
        let neighbor = |f: usize, off: (isize, isize, isize)| -> Option<usize> {
            let (iz, iy, ix) = decode(f);
            let (z, y, x) = (iz as isize + off.0, iy as isize + off.1, ix as isize + off.2);
            if z >= 0 && y >= 0 && x >= 0 && z < nz as isize && y < ny as isize && x < nx as isize {
                Some(idx(z as usize, y as usize, x as usize))
            } else {
                None
            }
        };
        // Clamped (Neumann) φ accessor for derivative stencils.
        let gphi = |phi: &[f64], iz: isize, iy: isize, ix: isize| -> f64 {
            let z = iz.clamp(0, nz as isize - 1) as usize;
            let y = iy.clamp(0, ny as isize - 1) as usize;
            let x = ix.clamp(0, nx as isize - 1) as usize;
            phi[idx(z, y, x)]
        };
        // Multilinear sample of a scalar field at continuous (cz, cy, cx).
        let interp = |arr: &[f64], cz: f64, cy: f64, cx: f64| -> f64 {
            let cl = |v: f64, hi: usize| v.clamp(0.0, hi as f64 - 1.0);
            let (cz, cy, cx) = (cl(cz, nz), cl(cy, ny), cl(cx, nx));
            let (z0, y0, x0) = (cz.floor() as usize, cy.floor() as usize, cx.floor() as usize);
            let (z1, y1, x1) = ((z0 + 1).min(nz - 1), (y0 + 1).min(ny - 1), (x0 + 1).min(nx - 1));
            let (fz, fy, fx) = (cz - z0 as f64, cy - y0 as f64, cx - x0 as f64);
            let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
            let c00 = lerp(arr[idx(z0, y0, x0)], arr[idx(z0, y0, x1)], fx);
            let c01 = lerp(arr[idx(z0, y1, x0)], arr[idx(z0, y1, x1)], fx);
            let c0 = lerp(c00, c01, fy);
            let c10 = lerp(arr[idx(z1, y0, x0)], arr[idx(z1, y0, x1)], fx);
            let c11 = lerp(arr[idx(z1, y1, x0)], arr[idx(z1, y1, x1)], fx);
            let c1 = lerp(c10, c11, fy);
            lerp(c0, c1, fz)
        };

        // ── ZeroCrossing(shifted) → active set (float-exact to ITK) ──────────
        let sign_change = |a: f64, b: f64| (a * b < 0.0) || ((a == 0.0) != (b == 0.0));
        let mut is_active = vec![false; n];
        for (f, &v) in shifted.iter().enumerate() {
            let av = v.abs();
            let mut crosses = false;
            for &off in &offsets {
                if let Some(g) = neighbor(f, off) {
                    let nv = shifted[g];
                    let forward = off.0 + off.1 + off.2 > 0;
                    if sign_change(v, nv) && (if forward { av <= nv.abs() } else { av < nv.abs() }) {
                        crosses = true;
                        break;
                    }
                }
            }
            is_active[f] = crosses;
        }

        let mut status = vec![ST_NULL; n];
        let mut phi: Vec<f64> = shifted.iter().map(|&s| if s > 0.0 { bg_val } else { -bg_val }).collect();
        let mut layers: Vec<Vec<usize>> = vec![Vec::new(); num as usize];

        macro_rules! push_layer {
            ($f:expr, $s:expr) => {{
                let s: i32 = $s;
                status[$f] = s;
                if s >= 0 {
                    layers[s as usize].insert(0, $f);
                }
            }};
        }

        // ConstructActiveLayer + initial neighbour layers (1 inside / 2 outside).
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
        // InitializeActiveLayerValues: clamp(shifted / upwind_len, ±½).
        for &f in &layers[0].clone() {
            let c = shifted[f];
            let mut l2 = 0.0f64;
            for &off in &offsets {
                let fwd = neighbor(f, off).map(|g| shifted[g]).unwrap_or(c) - c;
                let back = c - neighbor(f, (-off.0, -off.1, -off.2)).map(|g| shifted[g]).unwrap_or(c);
                let d = if fwd.abs() > back.abs() { fwd } else { back };
                if off.0 + off.1 + off.2 > 0 {
                    l2 += d * d;
                }
            }
            let len = l2.sqrt() + 1e-6;
            phi[f] = (c / len).clamp(-cf, cf);
        }

        // PropagateLayerValues / PropagateAllLayerValues.
        let propagate_layer = |layers: &mut Vec<Vec<usize>>,
                               phi: &mut [f64],
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
                let mut val = 0.0f64;
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

        // SegmentationLevelSetFunction::ComputeUpdate at active voxel f.
        // Returns (update, |weighted_curv|, |weighted_prop|, max|weighted_adv_i|).
        let seg_speed = |phi: &[f64], f: usize| -> (f64, f64, f64, f64) {
            let (iz, iy, ix) = decode(f);
            let (zi, yi, xi) = (iz as isize, iy as isize, ix as isize);
            let c = phi[f];
            let g = |dz: isize, dy: isize, dx: isize| gphi(phi, zi + dz, yi + dy, xi + dx);
            // φ derivatives (axis 0=x, 1=y, 2=z).
            let dxf = g(0, 0, 1) - c;
            let dxb = c - g(0, 0, -1);
            let dyf = g(0, 1, 0) - c;
            let dyb = c - g(0, -1, 0);
            let fx = 0.5 * (g(0, 0, 1) - g(0, 0, -1));
            let fy = 0.5 * (g(0, 1, 0) - g(0, -1, 0));
            let fxx = g(0, 0, 1) - 2.0 * c + g(0, 0, -1);
            let fyy = g(0, 1, 0) - 2.0 * c + g(0, -1, 0);
            let fxy = 0.25 * (g(0, -1, -1) - g(0, -1, 1) - g(0, 1, -1) + g(0, 1, 1));

            // ── InterpolateSurfaceLocation offset (idx − offset) ─────────────
            let off_axis = |fwd: f64, bwd: f64| -> f64 {
                if fwd * bwd >= 0.0 {
                    let df = fwd - c;
                    let db = c - bwd;
                    if df.abs() > db.abs() { df } else { db }
                } else if fwd * c < 0.0 {
                    fwd - c
                } else {
                    c - bwd
                }
            };
            // Sample P / A at the surface offset.
            let (cz, cy, cx);
            if c != 0.0 {
                let ox = off_axis(g(0, 0, 1), g(0, 0, -1));
                let oy = off_axis(g(0, 1, 0), g(0, -1, 0));
                let oz = if ndim == 3 { off_axis(g(1, 0, 0), g(-1, 0, 0)) } else { 0.0 };
                let norm = ox * ox + oy * oy + oz * oz + MIN_NORM;
                cx = ix as f64 - ox * c / norm;
                cy = iy as f64 - oy * c / norm;
                cz = iz as f64 - oz * c / norm;
            } else {
                cx = ix as f64;
                cy = iy as f64;
                cz = iz as f64;
            }
            let prop = interp(p, cz, cy, cx);
            let ax = interp(&adv[0], cz, cy, cx);
            let ay = interp(&adv[1], cz, cy, cx);
            let az = if ndim == 3 { interp(&adv[2], cz, cy, cx) } else { 0.0 };

            // ── Curvature term (ComputeCurvatureTerm) ────────────────────────
            let (curv, dzf, dzb);
            if ndim == 2 {
                let gm2 = fx * fx + fy * fy + GRAD_EPS;
                curv = (fxx * fy * fy + fyy * fx * fx - 2.0 * fx * fy * fxy) / gm2;
                dzf = 0.0;
                dzb = 0.0;
            } else {
                let fz = 0.5 * (g(1, 0, 0) - g(-1, 0, 0));
                let fzz = g(1, 0, 0) - 2.0 * c + g(-1, 0, 0);
                let fxz = 0.25 * (g(-1, 0, -1) - g(-1, 0, 1) - g(1, 0, -1) + g(1, 0, 1));
                let fyz = 0.25 * (g(-1, -1, 0) - g(-1, 1, 0) - g(1, -1, 0) + g(1, 1, 0));
                let gm2 = fx * fx + fy * fy + fz * fz + GRAD_EPS;
                curv = (fx * fx * (fyy + fzz) + fy * fy * (fxx + fzz) + fz * fz * (fxx + fyy)
                    - 2.0 * fx * fy * fxy
                    - 2.0 * fx * fz * fxz
                    - 2.0 * fy * fz * fyz)
                    / gm2;
                dzf = g(1, 0, 0) - c;
                dzb = c - g(-1, 0, 0);
            }
            let curv_term = curv * curv_w;

            // ── Propagation term (Godunov upwind in sign of P) ───────────────
            let prop_term = prop_w * prop;
            let pg = if prop_term > 0.0 {
                dxb.max(0.0).powi(2) + dxf.min(0.0).powi(2)
                    + dyb.max(0.0).powi(2) + dyf.min(0.0).powi(2)
                    + dzb.max(0.0).powi(2) + dzf.min(0.0).powi(2)
            } else {
                dxb.min(0.0).powi(2) + dxf.max(0.0).powi(2)
                    + dyb.min(0.0).powi(2) + dyf.max(0.0).powi(2)
                    + dzb.min(0.0).powi(2) + dzf.max(0.0).powi(2)
            };
            let propagation = prop_term * pg.sqrt();

            // ── Advection term (simple upwind per component) ─────────────────
            let mut adv_term = ax * (if ax > 0.0 { dxb } else { dxf })
                + ay * (if ay > 0.0 { dyb } else { dyf });
            if ndim == 3 {
                adv_term += az * (if az > 0.0 { dzb } else { dzf });
            }
            adv_term *= adv_w;

            let update = curv_term - propagation - adv_term;
            let max_adv = (adv_w * ax.abs()).max(adv_w * ay.abs()).max(adv_w * az.abs());
            (update, curv_term.abs(), prop_term.abs(), max_adv)
        };

        // ── ApplyUpdate loop ──
        for _ in 0..self.number_of_iterations {
            let al: Vec<usize> = layers[0].clone();
            // CalculateChange: per-voxel update + global maxima for the time step.
            let sp: Vec<(f64, f64, f64, f64)> = al.iter().map(|&f| seg_speed(&phi, f)).collect();
            let mut maxc = 0.0f64;
            let mut maxp = 0.0f64;
            let mut maxa = 0.0f64;
            for &(_, c, p, a) in &sp {
                maxc = maxc.max(c);
                maxp = maxp.max(p);
                maxa = maxa.max(a);
            }
            // ComputeGlobalTimeStep.
            let dt = if maxc > 0.0 {
                if maxa + maxp > 0.0 {
                    (wave_dt / (maxa + maxp)).min(wave_dt / maxc)
                } else {
                    wave_dt / maxc
                }
            } else if maxa + maxp > 0.0 {
                wave_dt / (maxa + maxp)
            } else {
                0.0
            };

            let mut up: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            let mut dn: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            let mut keep: Vec<usize> = Vec::new();
            let mut rms_acc = 0.0f64;
            let mut cnt = 0usize;
            for (k, &f) in al.iter().enumerate() {
                let old = phi[f];
                let nv = old + dt * sp[k].0;
                if nv >= cf {
                    if offsets.iter().any(|&o| neighbor(f, o).is_some_and(|g| status[g] == ST_CDN)) {
                        keep.push(f);
                        continue;
                    }
                    rms_acc += (nv - old).powi(2);
                    cnt += 1;
                    let tv = nv - CGV;
                    for &off in &offsets {
                        if let Some(g) = neighbor(f, off) {
                            if status[g] == 1 && (phi[g] < -cf || tv.abs() < phi[g].abs()) {
                                phi[g] = tv;
                            }
                        }
                    }
                    status[f] = ST_CUP;
                    up[0].insert(0, f);
                } else if nv < -cf {
                    if offsets.iter().any(|&o| neighbor(f, o).is_some_and(|g| status[g] == ST_CUP)) {
                        keep.push(f);
                        continue;
                    }
                    rms_acc += (nv - old).powi(2);
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
                    rms_acc += (nv - old).powi(2);
                    cnt += 1;
                    phi[f] = nv;
                    keep.push(f);
                }
            }
            layers[0] = keep;

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
            for f in u {
                move_to(&mut layers, &mut status, f, num - 2);
            }
            for f in d {
                move_to(&mut layers, &mut status, f, num - 1);
            }

            propagate_all!();

            let rms = if cnt > 0 {
                (rms_acc / cnt as f64).sqrt()
            } else {
                0.0
            };
            if rms < self.max_rms_error as f64 {
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
