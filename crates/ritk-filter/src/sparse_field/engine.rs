//! The band algorithm: active-set construction from the zero crossing, layer
//! construction, initial active values, the `ApplyUpdate` cascade and the RMS
//! convergence test — generic over the scalar and the step policy.

use std::collections::VecDeque;

use super::band::{process_status_list, propagate_all, ST_CDN, ST_CUP, ST_NULL};
use super::layers::{move_to, SparseFieldLayers};
use super::scalar::SparseScalar;
use super::topology::GridTopology;

/// ITK `SparseFieldLevelSetImageFilter` parameters, minus the physics.
pub(crate) struct SparseFieldConfig<T> {
    pub(crate) dims: [usize; 3],
    /// ITK `NumberOfLayers`: 2 for Canny, the image dimension for AntiAlias.
    pub(crate) number_of_layers: usize,
    /// ITK `m_ConstantGradientValue` (unit spacing).
    pub(crate) constant_gradient: T,
    pub(crate) iterations: usize,
    /// Convergence threshold on the active-layer RMS change.
    pub(crate) max_rms_error: f64,
}

/// The part of a SparseField filter that is not the band machinery: what one
/// active voxel does this iteration, and how the time step follows from it.
pub(crate) trait SparseFieldStep<T: SparseScalar> {
    /// Raw updates for the active layer, in iteration order, and the time step
    /// the caller's scheme assigns to them (`ComputeGlobalTimeStep` for Canny's
    /// variable scheme, the `CurvatureFlowImageFilter` constant for AntiAlias).
    fn stage(&self, phi: &[T], active: &[usize]) -> (Vec<T>, T);

    /// Correction applied to the new value before the band tests: AntiAlias
    /// locks the sign to the input binary here. Identity by default, and the
    /// monomorphiser removes the identity call.
    #[inline]
    fn clamp(&self, _f: usize, value: T) -> T {
        value
    }
}

/// Evolve `shifted` (the input level set with its iso-surface mapped to zero)
/// under `step`, returning the band values plus the `±(NL+1)` far field.
pub(crate) fn evolve<T: SparseScalar, S: SparseFieldStep<T>>(
    shifted: &[T],
    cfg: &SparseFieldConfig<T>,
    step: &S,
) -> Vec<T> {
    let topo = GridTopology::new(cfg.dims);
    let offsets = topo.face_offsets();
    let n = topo.len();
    let num = 2 * cfg.number_of_layers as i32 + 1;
    let cgv = cfg.constant_gradient;
    let half = T::from_f64(2.0);
    let bg = T::from_f64((cfg.number_of_layers + 1) as f64);
    let neg_bg = T::from_f64(-((cfg.number_of_layers + 1) as f64));
    let cf = cgv / half;

    // ── ZeroCrossing(shifted) → active set (float-exact to ITK) ──────────────
    let mut is_active = vec![false; n];
    for f in 0..n {
        let v = shifted[f];
        let av = v.abs();
        let mut crosses = false;
        for &off in offsets {
            if let Some(g) = topo.neighbor(f, off) {
                let nv = shifted[g];
                let forward = off.0 + off.1 + off.2 > 0;
                if sign_change(v, nv)
                    && (if forward {
                        av <= nv.abs()
                    } else {
                        av < nv.abs()
                    })
                {
                    crosses = true;
                    break;
                }
            }
        }
        is_active[f] = crosses;
    }

    let mut status = vec![ST_NULL; n];
    let mut phi: Vec<T> = shifted
        .iter()
        .map(|&s| if s > T::zero() { bg } else { neg_bg })
        .collect();
    let mut lists = SparseFieldLayers::new(n, num as usize);

    // ConstructActiveLayer + initial neighbour layers (1 inside / 2 outside).
    for f in 0..n {
        if is_active[f] {
            move_to(&mut lists, &mut status, f, 0);
            for &off in offsets {
                if let Some(g) = topo.neighbor(f, off) {
                    if !is_active[g] && status[g] == ST_NULL {
                        let ln = if shifted[g] < T::zero() { 1 } else { 2 };
                        move_to(&mut lists, &mut status, g, ln);
                    }
                }
            }
        }
    }
    // ConstructLayer i → i+2.
    for i in 1..(num - 2) {
        for f in lists.iter(i as usize).collect::<Vec<_>>() {
            for &off in offsets {
                if let Some(g) = topo.neighbor(f, off) {
                    if status[g] == ST_NULL {
                        move_to(&mut lists, &mut status, g, i + 2);
                    }
                }
            }
        }
    }
    // InitializeActiveLayerValues: clamp(shifted / upwind_len, ±½·CGV).
    for f in lists.iter(0).collect::<Vec<_>>() {
        let c = shifted[f];
        let mut l2 = T::zero();
        for &off in offsets {
            let fwd = topo.neighbor(f, off).map(|g| shifted[g]).unwrap_or(c) - c;
            let back = c - topo
                .neighbor(f, (-off.0, -off.1, -off.2))
                .map(|g| shifted[g])
                .unwrap_or(c);
            let d = if fwd.abs() > back.abs() { fwd } else { back };
            // Only count each axis once: the offsets list has +/- pairs, so use
            // the forward offsets (positive direction).
            if off.0 + off.1 + off.2 > 0 {
                l2 = l2 + d * d;
            }
        }
        let len = l2.sqrt() + T::from_f64(1e-6);
        phi[f] = clamp(c / len, -cf, cf);
    }

    propagate_all(&mut lists, &mut phi, &mut status, &topo, num, cgv);

    // ── ApplyUpdate ──────────────────────────────────────────────────────────
    for _ in 0..cfg.iterations {
        let active: Vec<usize> = lists.iter(0).collect();
        let (updates, dt) = step.stage(&phi, &active);

        let mut up: [VecDeque<usize>; 2] = [VecDeque::new(), VecDeque::new()];
        let mut dn: [VecDeque<usize>; 2] = [VecDeque::new(), VecDeque::new()];
        let mut keep: Vec<usize> = Vec::new();
        let mut rms_acc = 0.0f64;
        let mut cnt = 0usize;
        for (k, &f) in active.iter().enumerate() {
            let old = phi[f];
            let nv = step.clamp(f, old + dt * updates[k]);
            if nv >= cf {
                if offsets
                    .iter()
                    .any(|&o| topo.neighbor(f, o).is_some_and(|g| status[g] == ST_CDN))
                {
                    keep.push(f);
                    continue;
                }
                rms_acc += (nv - old).to_f64().powi(2);
                cnt += 1;
                let tv = nv - cgv;
                for &off in offsets {
                    if let Some(g) = topo.neighbor(f, off) {
                        if status[g] == 1 && (phi[g] < -cf || tv.abs() < phi[g].abs()) {
                            phi[g] = tv;
                        }
                    }
                }
                status[f] = ST_CUP;
                up[0].push_front(f);
            } else if nv < -cf {
                if offsets
                    .iter()
                    .any(|&o| topo.neighbor(f, o).is_some_and(|g| status[g] == ST_CUP))
                {
                    keep.push(f);
                    continue;
                }
                rms_acc += (nv - old).to_f64().powi(2);
                cnt += 1;
                let tv = nv + cgv;
                for &off in offsets {
                    if let Some(g) = topo.neighbor(f, off) {
                        if status[g] == 2 && (phi[g] >= cf || tv.abs() < phi[g].abs()) {
                            phi[g] = tv;
                        }
                    }
                }
                status[f] = ST_CDN;
                dn[0].push_front(f);
            } else {
                rms_acc += (nv - old).to_f64().powi(2);
                cnt += 1;
                phi[f] = nv;
                keep.push(f);
            }
        }
        lists.replace(0, &keep);

        // ProcessStatusList cascade, then ProcessOutsideList.
        let mut u = process_status_list(
            &mut lists,
            &mut status,
            &topo,
            std::mem::take(&mut up[0]),
            2,
            1,
        );
        let mut d = process_status_list(
            &mut lists,
            &mut status,
            &topo,
            std::mem::take(&mut dn[0]),
            1,
            2,
        );
        let mut up_to = 0i32;
        let mut dn_to = 0i32;
        let mut us = 3i32;
        let mut ds = 4i32;
        while ds < num {
            u = process_status_list(&mut lists, &mut status, &topo, u, up_to, us);
            d = process_status_list(&mut lists, &mut status, &topo, d, dn_to, ds);
            up_to = if up_to == 0 { 1 } else { up_to + 2 };
            dn_to += 2;
            us += 2;
            ds += 2;
        }
        u = process_status_list(&mut lists, &mut status, &topo, u, up_to, ST_NULL);
        d = process_status_list(&mut lists, &mut status, &topo, d, dn_to, ST_NULL);
        for f in u {
            move_to(&mut lists, &mut status, f, num - 2);
        }
        for f in d {
            move_to(&mut lists, &mut status, f, num - 1);
        }

        propagate_all(&mut lists, &mut phi, &mut status, &topo, num, cgv);

        let rms = if cnt > 0 {
            (rms_acc / cnt as f64).sqrt()
        } else {
            0.0
        };
        if rms < cfg.max_rms_error {
            break;
        }
    }

    // PostProcessOutput: background voxels → ±(NL+1) by current sign.
    for f in 0..n {
        if status[f] == ST_NULL {
            phi[f] = if phi[f] > T::zero() { bg } else { neg_bg };
        }
    }
    phi
}

/// `ZeroCrossingImageFilter`'s straddle test: opposite signs, with an exact zero
/// counting as crossed against any nonzero.
#[inline]
fn sign_change<T: SparseScalar>(a: T, b: T) -> bool {
    (a * b < T::zero()) || ((a == T::zero()) != (b == T::zero()))
}

/// `clamp` without the panic-on-empty-range of `Ord::clamp`, which `f32`/`f64`
/// have but a generic scalar does not.
#[inline]
fn clamp<T: SparseScalar>(v: T, lo: T, hi: T) -> T {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Noop;

    impl SparseFieldStep<f64> for Noop {
        fn stage(&self, _phi: &[f64], active: &[usize]) -> (Vec<f64>, f64) {
            (vec![0.0; active.len()], 0.0)
        }
    }

    impl SparseFieldStep<f32> for Noop {
        fn stage(&self, _phi: &[f32], active: &[usize]) -> (Vec<f32>, f32) {
            (vec![0.0; active.len()], 0.0)
        }
    }

    /// A step image: the boundary column is the active set, and the engine writes
    /// the signed band around it even with no iterations.
    #[test]
    fn zero_iterations_builds_the_signed_band() {
        let (nz, ny, nx) = (1usize, 4usize, 4usize);
        let mut shifted = vec![-1.0f64; nz * ny * nx];
        for iy in 0..ny {
            for ix in 2..nx {
                shifted[iy * nx + ix] = 1.0;
            }
        }
        let cfg = SparseFieldConfig {
            dims: [nz, ny, nx],
            number_of_layers: 2,
            constant_gradient: 1.0f64,
            iterations: 0,
            max_rms_error: 0.0,
        };
        let phi = evolve(&shifted, &cfg, &Noop);

        // Two layers of band on each side, the far field beyond: |φ| ≤ NL + 1 = 3.
        for (f, &v) in phi.iter().enumerate() {
            let ix = f % nx;
            assert!(v.abs() <= 3.0, "band value {v} out of range at {f}");
            if ix < 2 {
                assert!(v < 0.0, "inside voxel {f} must stay negative");
            } else {
                assert!(v > 0.0, "outside voxel {f} must stay positive");
            }
        }
    }

    /// The far field: a uniform image has no zero crossing, so every voxel ends
    /// at `-(NL+1)`.
    #[test]
    fn no_crossing_leaves_only_the_far_field() {
        let shifted = vec![-1.0f32; 8];
        let cfg = SparseFieldConfig {
            dims: [1, 2, 4],
            number_of_layers: 2,
            constant_gradient: 1.0f32,
            iterations: 3,
            max_rms_error: 0.0,
        };
        assert_eq!(evolve(&shifted, &cfg, &Noop), vec![-3.0f32; 8]);
    }
}
