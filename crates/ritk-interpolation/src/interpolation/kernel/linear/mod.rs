//! Linear interpolation implementation.

use super::BoundsPolicy;
use crate::interpolation::dispatch::dispatch_linear;
use crate::interpolation::shared::OutOfBoundsMode;
use ritk_core::interpolation::Interpolator;
use coeus_core::Backend;
use coeus_tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinearInterpolator {
    pub bounds_policy: BoundsPolicy,
}

impl LinearInterpolator {
    pub fn new() -> Self {
        Self {
            bounds_policy: BoundsPolicy::Extend,
        }
    }

    pub fn new_zero_pad() -> Self {
        Self {
            bounds_policy: BoundsPolicy::ZeroPad,
        }
    }

    pub fn with_bounds_policy(mut self, policy: BoundsPolicy) -> Self {
        self.bounds_policy = policy;
        self
    }
}

impl Default for LinearInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B> Interpolator<B> for LinearInterpolator
where
    B: Backend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    fn interpolate(&self, data: &Tensor<f32, B>, indices: Tensor<f32, B>) -> Tensor<f32, B> {
        dispatch_linear(data, indices, self.bounds_policy.as_out_of_bounds_mode())
    }
}

pub(crate) fn interpolate_linear_host<B>(
    data: &Tensor<f32, B>,
    indices: Tensor<f32, B>,
    mode: OutOfBoundsMode,
) -> Tensor<f32, B>
where
    B: Backend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    let backend = B::default();
    let shape = data.shape();
    let rank = shape.len();
    let idx_shape = indices.shape();
    assert_eq!(idx_shape.len(), 2, "indices must be rank 2");
    assert_eq!(idx_shape[1], rank, "indices trailing dim must equal tensor rank");

    let values = data.as_slice().to_vec();
    let coords = indices.as_slice().to_vec();
    let n_points = idx_shape[0];
    let mut out = Vec::with_capacity(n_points);

    for point in 0..n_points {
        let coord = &coords[point * rank..(point + 1) * rank];
        out.push(match rank {
            1 => linear_1d(&values, shape[0], coord[0], mode),
            2 => linear_2d(&values, shape[0], shape[1], coord[0], coord[1], mode),
            3 => linear_3d(
                &values, shape[0], shape[1], shape[2], coord[0], coord[1], coord[2], mode,
            ),
            4 => linear_4d(
                &values, shape[0], shape[1], shape[2], shape[3], coord[0], coord[1], coord[2],
                coord[3], mode,
            ),
            _ => unreachable!(),
        });
    }

    Tensor::<f32, B>::from_slice_on([n_points], &out, &backend)
}

#[inline]
fn in_bounds(coord: f32, size: usize) -> bool {
    coord >= 0.0 && coord <= (size.saturating_sub(1)) as f32
}

#[inline]
fn clamp_floor_pair(coord: f32, size: usize) -> (usize, usize, f32) {
    let max = size.saturating_sub(1) as f32;
    let clamped = coord.clamp(0.0, max);
    let lo_f = clamped.floor();
    let lo = lo_f as usize;
    let hi = (lo + 1).min(size.saturating_sub(1));
    (lo, hi, clamped - lo_f)
}

fn linear_1d(values: &[f32], d0: usize, x: f32, mode: OutOfBoundsMode) -> f32 {
    if matches!(mode, OutOfBoundsMode::ZeroPad) && !in_bounds(x, d0) {
        return 0.0;
    }
    let (x0, x1, wx) = clamp_floor_pair(x, d0);
    values[x0] * (1.0 - wx) + values[x1] * wx
}

fn linear_2d(values: &[f32], d0: usize, d1: usize, x: f32, y: f32, mode: OutOfBoundsMode) -> f32 {
    if matches!(mode, OutOfBoundsMode::ZeroPad) && (!in_bounds(x, d1) || !in_bounds(y, d0)) {
        return 0.0;
    }
    let (x0, x1, wx) = clamp_floor_pair(x, d1);
    let (y0, y1, wy) = clamp_floor_pair(y, d0);
    let at = |yy: usize, xx: usize| values[yy * d1 + xx];
    let v00 = at(y0, x0);
    let v01 = at(y0, x1);
    let v10 = at(y1, x0);
    let v11 = at(y1, x1);
    let v0 = v00 * (1.0 - wx) + v01 * wx;
    let v1 = v10 * (1.0 - wx) + v11 * wx;
    v0 * (1.0 - wy) + v1 * wy
}

fn linear_3d(
    values: &[f32],
    d0: usize,
    d1: usize,
    d2: usize,
    x: f32,
    y: f32,
    z: f32,
    mode: OutOfBoundsMode,
) -> f32 {
    if matches!(mode, OutOfBoundsMode::ZeroPad)
        && (!in_bounds(x, d2) || !in_bounds(y, d1) || !in_bounds(z, d0))
    {
        return 0.0;
    }
    let (x0, x1, wx) = clamp_floor_pair(x, d2);
    let (y0, y1, wy) = clamp_floor_pair(y, d1);
    let (z0, z1, wz) = clamp_floor_pair(z, d0);
    let at = |zz: usize, yy: usize, xx: usize| values[(zz * d1 + yy) * d2 + xx];
    let c000 = at(z0, y0, x0);
    let c001 = at(z0, y0, x1);
    let c010 = at(z0, y1, x0);
    let c011 = at(z0, y1, x1);
    let c100 = at(z1, y0, x0);
    let c101 = at(z1, y0, x1);
    let c110 = at(z1, y1, x0);
    let c111 = at(z1, y1, x1);
    let c00 = c000 * (1.0 - wx) + c001 * wx;
    let c01 = c010 * (1.0 - wx) + c011 * wx;
    let c10 = c100 * (1.0 - wx) + c101 * wx;
    let c11 = c110 * (1.0 - wx) + c111 * wx;
    let c0 = c00 * (1.0 - wy) + c01 * wy;
    let c1 = c10 * (1.0 - wy) + c11 * wy;
    c0 * (1.0 - wz) + c1 * wz
}

#[allow(clippy::too_many_arguments)]
fn linear_4d(
    values: &[f32],
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
    x: f32,
    y: f32,
    z: f32,
    w: f32,
    mode: OutOfBoundsMode,
) -> f32 {
    if matches!(mode, OutOfBoundsMode::ZeroPad)
        && (!in_bounds(x, d3) || !in_bounds(y, d2) || !in_bounds(z, d1) || !in_bounds(w, d0))
    {
        return 0.0;
    }
    let (x0, x1, wx) = clamp_floor_pair(x, d3);
    let (y0, y1, wy) = clamp_floor_pair(y, d2);
    let (z0, z1, wz) = clamp_floor_pair(z, d1);
    let (w0, w1, ww) = clamp_floor_pair(w, d0);
    let at = |wwi: usize, zzi: usize, yyi: usize, xxi: usize| {
        values[(((wwi * d1 + zzi) * d2) + yyi) * d3 + xxi]
    };
    let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let v000 = lerp(at(w0, z0, y0, x0), at(w0, z0, y0, x1), wx);
    let v001 = lerp(at(w0, z0, y1, x0), at(w0, z0, y1, x1), wx);
    let v010 = lerp(at(w0, z1, y0, x0), at(w0, z1, y0, x1), wx);
    let v011 = lerp(at(w0, z1, y1, x0), at(w0, z1, y1, x1), wx);
    let v100 = lerp(at(w1, z0, y0, x0), at(w1, z0, y0, x1), wx);
    let v101 = lerp(at(w1, z0, y1, x0), at(w1, z0, y1, x1), wx);
    let v110 = lerp(at(w1, z1, y0, x0), at(w1, z1, y0, x1), wx);
    let v111 = lerp(at(w1, z1, y1, x0), at(w1, z1, y1, x1), wx);
    let z00 = lerp(v000, v001, wy);
    let z01 = lerp(v010, v011, wy);
    let z10 = lerp(v100, v101, wy);
    let z11 = lerp(v110, v111, wy);
    let w0v = lerp(z00, z01, wz);
    let w1v = lerp(z10, z11, wz);
    lerp(w0v, w1v, ww)
}
