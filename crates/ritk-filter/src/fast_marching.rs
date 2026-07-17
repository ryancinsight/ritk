//! Fast-marching arrival-time solver (Eikonal equation).
//!
//! # Mathematical Specification
//!
//! Ports `itk::FastMarchingImageFilter`. Given a non-negative speed image `F`
//! and a set of trial (seed) points with initial arrival times, it solves the
//! Eikonal equation `‖∇T‖·F = 1` by ordered upwind propagation: a min-heap pops
//! the smallest tentative arrival time (which is then final), marks it *alive*,
//! and re-solves each non-alive neighbour.
//!
//! At a voxel the upwind quadratic is solved over the per-axis smallest alive
//! neighbour `u_a`, taken in increasing order while the running solution exceeds
//! `u_a`:
//!
//! ```text
//! Σ_a (1/spacing_a²)·(T − u_a)² = (normalization_factor / F)²
//! ```
//!
//! Because the popped value is final, heap tie-ordering cannot change the
//! result, so the arrival-time field is deterministic and bit-exact to
//! `sitk.FastMarching`. Internal arithmetic is `f64` (ITK's solver precision).
//! Voxels never reached keep the large sentinel value.

use ritk_image::tensor::Backend;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Sentinel for unreached / not-yet-computed voxels (ITK `m_LargeValue`).
const LARGE_VALUE: f64 = f32::MAX as f64;

/// Fast-marching arrival-time filter (`itk::FastMarchingImageFilter`).
#[derive(Debug, Clone)]
pub struct FastMarchingFilter {
    /// Trial (seed) points as `[z, y, x]` indices.
    pub trial_points: Vec<[usize; 3]>,
    /// Initial arrival time per trial point; empty ⇒ all `0.0`.
    pub initial_trial_values: Vec<f64>,
    /// Speed normalization factor (ITK default `1.0`).
    pub normalization_factor: f64,
    /// Stop once the smallest tentative arrival time exceeds this (ITK default ∞).
    pub stopping_value: f64,
}

impl Default for FastMarchingFilter {
    fn default() -> Self {
        Self {
            trial_points: Vec::new(),
            initial_trial_values: Vec::new(),
            normalization_factor: 1.0,
            stopping_value: f64::MAX / 2.0,
        }
    }
}

impl FastMarchingFilter {
    /// Construct with trial points (other fields default).
    pub fn new(trial_points: Vec<[usize; 3]>) -> Self {
        Self {
            trial_points,
            ..Self::default()
        }
    }

    /// Solve the arrival-time field for the given speed image.
    pub fn apply<B: Backend>(&self, speed: &Image<B, 3>) -> Image<B, 3> {
        let (spd, dims) = extract_vec_infallible(speed);
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let sp = [speed.spacing()[0], speed.spacing()[1], speed.spacing()[2]];
        let nf = self.normalization_factor;

        let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;
        let coords = |i: usize| {
            let z = i / (ny * nx);
            let r = i % (ny * nx);
            (z, r / nx, r % nx)
        };

        // 0 = far, 1 = trial, 2 = alive.
        let mut label = vec![0u8; n];
        let mut t = vec![LARGE_VALUE; n];
        // Min-heap on (arrival-time bits, index); arrival times are ≥ 0 so the
        // f64 bit pattern is monotonic, giving a correct min-heap via Reverse.
        let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();

        for (k, &[z, y, x]) in self.trial_points.iter().enumerate() {
            if z >= nz || y >= ny || x >= nx {
                continue;
            }
            let v = self.initial_trial_values.get(k).copied().unwrap_or(0.0);
            let i = idx(z, y, x);
            t[i] = v;
            label[i] = 1;
            heap.push(Reverse((v.to_bits(), i)));
        }

        // Per-axis ±1 neighbour offsets in (z, y, x).
        let axis_neighbors = |z: usize, y: usize, x: usize, axis: usize| -> [Option<usize>; 2] {
            let (zi, yi, xi) = (z as isize, y as isize, x as isize);
            let step = |s: isize| -> Option<usize> {
                let (mut nz_, mut ny_, mut nx_) = (zi, yi, xi);
                match axis {
                    0 => nz_ += s,
                    1 => ny_ += s,
                    _ => nx_ += s,
                }
                if nz_ < 0
                    || ny_ < 0
                    || nx_ < 0
                    || nz_ >= nz as isize
                    || ny_ >= ny as isize
                    || nx_ >= nx as isize
                {
                    None
                } else {
                    Some(idx(nz_ as usize, ny_ as usize, nx_ as usize))
                }
            };
            [step(-1), step(1)]
        };

        // Upwind quadratic solve for the arrival time at (z, y, x).
        let solve = |z: usize, y: usize, x: usize, t: &[f64], label: &[u8]| -> f64 {
            // Smallest alive neighbour per axis (LARGE_VALUE if none).
            let mut nodes: [(f64, usize); 3] =
                [(LARGE_VALUE, 0), (LARGE_VALUE, 1), (LARGE_VALUE, 2)];
            for (axis, node) in nodes.iter_mut().enumerate() {
                let mut best = LARGE_VALUE;
                for ni in axis_neighbors(z, y, x, axis).into_iter().flatten() {
                    if label[ni] == 2 && t[ni] < best {
                        best = t[ni];
                    }
                }
                *node = (best, axis);
            }
            nodes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let speed_val = spd[idx(z, y, x)] as f64;
            let mut cc = speed_val / nf;
            cc = -(1.0 / cc).powi(2);
            let (mut aa, mut bb) = (0.0f64, 0.0f64);
            let mut solution = LARGE_VALUE;
            for &(value, axis) in &nodes {
                if solution >= value && value < LARGE_VALUE {
                    let sf = (1.0 / sp[axis]).powi(2);
                    aa += sf;
                    bb += value * sf;
                    cc += value * value * sf;
                    let discrim = bb * bb - aa * cc;
                    if discrim < 0.0 {
                        break;
                    }
                    solution = (discrim.sqrt() + bb) / aa;
                } else {
                    break;
                }
            }
            solution
        };

        while let Some(Reverse((bits, i))) = heap.pop() {
            if label[i] == 2 {
                continue; // outdated heap entry
            }
            let val = f64::from_bits(bits);
            if val > self.stopping_value {
                break;
            }
            label[i] = 2; // alive — `t[i]` is now final
            let (z, y, x) = coords(i);
            for axis in 0..3 {
                for ni in axis_neighbors(z, y, x, axis).into_iter().flatten() {
                    if label[ni] == 2 {
                        continue;
                    }
                    let (nz_, nny, nnx) = coords(ni);
                    let nv = solve(nz_, nny, nnx, &t, &label);
                    if nv < t[ni] {
                        t[ni] = nv;
                        label[ni] = 1;
                        heap.push(Reverse((nv.to_bits(), ni)));
                    }
                }
            }
        }

        let out: Vec<f32> = t.iter().map(|&v| v as f32).collect();
        rebuild(out, dims, speed)
    }
    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(
        &self,
        speed: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (spd, dims) = ritk_tensor_ops::native::extract_image_vec(speed)?;
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let sp = [speed.spacing()[0], speed.spacing()[1], speed.spacing()[2]];
        let nf = self.normalization_factor;

        let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;
        let coords = |i: usize| {
            let z = i / (ny * nx);
            let r = i % (ny * nx);
            (z, r / nx, r % nx)
        };

        // 0 = far, 1 = trial, 2 = alive.
        let mut label = vec![0u8; n];
        let mut t = vec![LARGE_VALUE; n];
        // Min-heap on (arrival-time bits, index); arrival times are ≥ 0 so the
        // f64 bit pattern is monotonic, giving a correct min-heap via Reverse.
        let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();

        for (k, &[z, y, x]) in self.trial_points.iter().enumerate() {
            if z >= nz || y >= ny || x >= nx {
                continue;
            }
            let v = self.initial_trial_values.get(k).copied().unwrap_or(0.0);
            let i = idx(z, y, x);
            t[i] = v;
            label[i] = 1;
            heap.push(Reverse((v.to_bits(), i)));
        }

        // Per-axis ±1 neighbour offsets in (z, y, x).
        let axis_neighbors = |z: usize, y: usize, x: usize, axis: usize| -> [Option<usize>; 2] {
            let (zi, yi, xi) = (z as isize, y as isize, x as isize);
            let step = |s: isize| -> Option<usize> {
                let (mut nz_, mut ny_, mut nx_) = (zi, yi, xi);
                match axis {
                    0 => nz_ += s,
                    1 => ny_ += s,
                    _ => nx_ += s,
                }
                if nz_ < 0
                    || ny_ < 0
                    || nx_ < 0
                    || nz_ >= nz as isize
                    || ny_ >= ny as isize
                    || nx_ >= nx as isize
                {
                    None
                } else {
                    Some(idx(nz_ as usize, ny_ as usize, nx_ as usize))
                }
            };
            [step(-1), step(1)]
        };

        // Upwind quadratic solve for the arrival time at (z, y, x).
        let solve = |z: usize, y: usize, x: usize, t: &[f64], label: &[u8]| -> f64 {
            // Smallest alive neighbour per axis (LARGE_VALUE if none).
            let mut nodes: [(f64, usize); 3] =
                [(LARGE_VALUE, 0), (LARGE_VALUE, 1), (LARGE_VALUE, 2)];
            for (axis, node) in nodes.iter_mut().enumerate() {
                let mut best = LARGE_VALUE;
                for ni in axis_neighbors(z, y, x, axis).into_iter().flatten() {
                    if label[ni] == 2 && t[ni] < best {
                        best = t[ni];
                    }
                }
                *node = (best, axis);
            }
            nodes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let speed_val = spd[idx(z, y, x)] as f64;
            let mut cc = speed_val / nf;
            cc = -(1.0 / cc).powi(2);
            let (mut aa, mut bb) = (0.0f64, 0.0f64);
            let mut solution = LARGE_VALUE;
            for &(value, axis) in &nodes {
                if solution >= value && value < LARGE_VALUE {
                    let sf = (1.0 / sp[axis]).powi(2);
                    aa += sf;
                    bb += value * sf;
                    cc += value * value * sf;
                    let discrim = bb * bb - aa * cc;
                    if discrim < 0.0 {
                        break;
                    }
                    solution = (discrim.sqrt() + bb) / aa;
                } else {
                    break;
                }
            }
            solution
        };

        while let Some(Reverse((bits, i))) = heap.pop() {
            if label[i] == 2 {
                continue; // outdated heap entry
            }
            let val = f64::from_bits(bits);
            if val > self.stopping_value {
                break;
            }
            label[i] = 2; // alive — `t[i]` is now final
            let (z, y, x) = coords(i);
            for axis in 0..3 {
                for ni in axis_neighbors(z, y, x, axis).into_iter().flatten() {
                    if label[ni] == 2 {
                        continue;
                    }
                    let (nz_, nny, nnx) = coords(ni);
                    let nv = solve(nz_, nny, nnx, &t, &label);
                    if nv < t[ni] {
                        t[ni] = nv;
                        label[ni] = 1;
                        heap.push(Reverse((nv.to_bits(), ni)));
                    }
                }
            }
        }

        let out: Vec<f32> = t.iter().map(|&v| v as f32).collect();
        crate::native_support::rebuild_image(out, dims, speed, backend)
    }
}

#[cfg(test)]
#[path = "tests_fast_marching.rs"]
mod tests_fast_marching;
