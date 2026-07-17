//! Colliding-fronts segmentation potential.
//!
//! # Mathematical Specification
//!
//! Ports `itk::CollidingFrontsImageFilter`. Two fast-marching fronts are
//! propagated through the speed image — one from `seeds1`, one from `seeds2` —
//! and their **upwind gradients** `∇T1`, `∇T2` are combined into the potential
//!
//! ```text
//! P = ∇T1 · ∇T2
//! ```
//!
//! which is strongly negative where the two fronts meet head-on (their gradients
//! oppose). Seed voxels are pinned to `negative_epsilon` (`−1e-6`). With
//! `apply_connectivity` the output is restricted to the connected region of
//! `P ≤ negative_epsilon` reachable from `seeds1` (the corridor between the seed
//! sets); elsewhere 0. Without it, `P` is returned directly.
//!
//! The upwind gradient matches `itk::FastMarchingUpwindGradientImageFilter`:
//! per axis, `dx_back = T(p) − T(p−e)` and `dx_fwd = T(p+e) − T(p)` over reached
//! neighbours, choosing `0` if `max(dx_back, −dx_fwd) < 0`, else the larger of
//! the two, divided by spacing. Seed voxels' gradients are irrelevant (they are
//! overwritten). This is float-exact to `sitk.CollidingFronts` (default
//! `stop_on_targets = false`, full march).

use std::collections::VecDeque;

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

use crate::fast_marching::FastMarchingFilter;

/// Threshold above which an arrival time counts as "unreached".
const REACHED_LIMIT: f32 = f32::MAX * 0.5;

/// Colliding-fronts filter (`itk::CollidingFrontsImageFilter`).
#[derive(Debug, Clone)]
pub struct CollidingFrontsFilter {
    /// First front's seed points, `[z, y, x]`.
    pub seeds1: Vec<[usize; 3]>,
    /// Second front's seed points, `[z, y, x]`.
    pub seeds2: Vec<[usize; 3]>,
    /// Restrict the output to the connected colliding corridor (ITK default `true`).
    pub apply_connectivity: bool,
    /// Value pinned at the seeds / connectivity threshold (ITK default `−1e-6`).
    pub negative_epsilon: f64,
}

impl Default for CollidingFrontsFilter {
    fn default() -> Self {
        Self {
            seeds1: Vec::new(),
            seeds2: Vec::new(),
            apply_connectivity: true,
            negative_epsilon: -1e-6,
        }
    }
}

impl CollidingFrontsFilter {
    /// Construct from the two seed sets (other fields default).
    pub fn new(seeds1: Vec<[usize; 3]>, seeds2: Vec<[usize; 3]>) -> Self {
        Self {
            seeds1,
            seeds2,
            ..Self::default()
        }
    }

    /// Compute the colliding-fronts potential for the given speed image.
    pub fn apply<B: Backend>(&self, speed: &Image<B, 3>) -> Image<B, 3> {
        let dims = speed.shape();
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let sp = [speed.spacing()[0], speed.spacing()[1], speed.spacing()[2]];

        let (t1, _) =
            extract_vec_infallible(&FastMarchingFilter::new(self.seeds1.clone()).apply(speed));
        let (t2, _) =
            extract_vec_infallible(&FastMarchingFilter::new(self.seeds2.clone()).apply(speed));
        let g1 = upwind_gradient(&t1, dims, sp);
        let g2 = upwind_gradient(&t2, dims, sp);

        let neg = self.negative_epsilon as f32;
        let mut mult: Vec<f32> = (0..n)
            .map(|i| ((g1[i][0] * g2[i][0]) + (g1[i][1] * g2[i][1]) + (g1[i][2] * g2[i][2])) as f32)
            .collect();
        let idx = |[z, y, x]: [usize; 3]| (z * ny + y) * nx + x;
        for &s in self.seeds1.iter().chain(self.seeds2.iter()) {
            mult[idx(s)] = neg;
        }

        if !self.apply_connectivity {
            return rebuild(mult, dims, speed);
        }

        // Flood from seeds1 over {P ≤ negative_epsilon}, copying P; rest 0.
        let mut out = vec![0.0f32; n];
        let mut visited = vec![false; n];
        let mut queue: VecDeque<usize> = VecDeque::new();
        for &s in &self.seeds1 {
            let i = idx(s);
            if mult[i] <= neg && !visited[i] {
                visited[i] = true;
                queue.push_back(i);
            }
        }
        let face = [
            (-1i64, 0i64, 0i64),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];
        while let Some(i) = queue.pop_front() {
            out[i] = mult[i];
            let (z, y, x) = (i / (ny * nx), (i % (ny * nx)) / nx, i % nx);
            for &(dz, dy, dx) in &face {
                let (zi, yi, xi) = (z as i64 + dz, y as i64 + dy, x as i64 + dx);
                if zi < 0
                    || yi < 0
                    || xi < 0
                    || zi >= nz as i64
                    || yi >= ny as i64
                    || xi >= nx as i64
                {
                    continue;
                }
                let ni = (zi as usize * ny + yi as usize) * nx + xi as usize;
                if !visited[ni] && mult[ni] <= neg {
                    visited[ni] = true;
                    queue.push_back(ni);
                }
            }
        }
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
        let dims = speed.shape();
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let sp = [speed.spacing()[0], speed.spacing()[1], speed.spacing()[2]];

        let t1 = FastMarchingFilter::new(self.seeds1.clone()).apply_native(speed, backend)?;
        let t2 = FastMarchingFilter::new(self.seeds2.clone()).apply_native(speed, backend)?;
        let (t1, _) = ritk_tensor_ops::native::extract_image_vec(&t1)?;
        let (t2, _) = ritk_tensor_ops::native::extract_image_vec(&t2)?;
        let g1 = upwind_gradient(&t1, dims, sp);
        let g2 = upwind_gradient(&t2, dims, sp);

        let neg = self.negative_epsilon as f32;
        let mut mult: Vec<f32> = (0..n)
            .map(|i| ((g1[i][0] * g2[i][0]) + (g1[i][1] * g2[i][1]) + (g1[i][2] * g2[i][2])) as f32)
            .collect();
        let idx = |[z, y, x]: [usize; 3]| (z * ny + y) * nx + x;
        for &s in self.seeds1.iter().chain(self.seeds2.iter()) {
            mult[idx(s)] = neg;
        }

        if !self.apply_connectivity {
            return crate::native_support::rebuild_image(mult, dims, speed, backend);
        }

        let mut out = vec![0.0f32; n];
        let mut visited = vec![false; n];
        let mut queue: VecDeque<usize> = VecDeque::new();
        for &s in &self.seeds1 {
            let i = idx(s);
            if mult[i] <= neg && !visited[i] {
                visited[i] = true;
                queue.push_back(i);
            }
        }
        let face = [
            (-1i64, 0i64, 0i64),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];
        while let Some(i) = queue.pop_front() {
            out[i] = mult[i];
            let (z, y, x) = (i / (ny * nx), (i % (ny * nx)) / nx, i % nx);
            for &(dz, dy, dx) in &face {
                let (zi, yi, xi) = (z as i64 + dz, y as i64 + dy, x as i64 + dx);
                if zi < 0
                    || yi < 0
                    || xi < 0
                    || zi >= nz as i64
                    || yi >= ny as i64
                    || xi >= nx as i64
                {
                    continue;
                }
                let ni = (zi as usize * ny + yi as usize) * nx + xi as usize;
                if !visited[ni] && mult[ni] <= neg {
                    visited[ni] = true;
                    queue.push_back(ni);
                }
            }
        }
        crate::native_support::rebuild_image(out, dims, speed, backend)
    }
}

/// Upwind gradient of an arrival-time field (`itk::FastMarchingUpwindGradient`).
fn upwind_gradient(t: &[f32], dims: [usize; 3], sp: [f64; 3]) -> Vec<[f64; 3]> {
    let [nz, ny, nx] = dims;
    let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;
    let mut grad = vec![[0.0f64; 3]; nz * ny * nx];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = idx(z, y, x);
                if t[i] >= REACHED_LIMIT {
                    continue;
                }
                let center = t[i] as f64;
                let (zi, yi, xi) = (z as i64, y as i64, x as i64);
                for axis in 0..3 {
                    let nb = |s: i64| -> Option<f64> {
                        let (mut az, mut ay, mut ax) = (zi, yi, xi);
                        match axis {
                            0 => az += s,
                            1 => ay += s,
                            _ => ax += s,
                        }
                        if az < 0
                            || ay < 0
                            || ax < 0
                            || az >= nz as i64
                            || ay >= ny as i64
                            || ax >= nx as i64
                        {
                            return None;
                        }
                        let v = t[idx(az as usize, ay as usize, ax as usize)];
                        if v >= REACHED_LIMIT {
                            None
                        } else {
                            Some(v as f64)
                        }
                    };
                    let dx_back = nb(-1).map_or(0.0, |b| center - b);
                    let dx_fwd = nb(1).map_or(0.0, |f| f - center);
                    let g = if dx_back.max(-dx_fwd) < 0.0 {
                        0.0
                    } else if dx_back > -dx_fwd {
                        dx_back
                    } else {
                        dx_fwd
                    };
                    grad[i][axis] = g / sp[axis];
                }
            }
        }
    }
    grad
}

#[cfg(test)]
#[path = "tests_colliding_fronts.rs"]
mod tests_colliding_fronts;
