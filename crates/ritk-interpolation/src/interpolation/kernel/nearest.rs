//! Nearest-neighbor interpolation implementation.

use super::BoundsPolicy;
use crate::interpolation::dispatch::dispatch_nearest;
use crate::interpolation::shared::OutOfBoundsMode;
use ritk_core::interpolation::Interpolator;
use coeus_core::Backend;
use coeus_tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct NearestNeighborInterpolator {
    pub bounds_policy: BoundsPolicy,
}

impl NearestNeighborInterpolator {
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

impl Default for NearestNeighborInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B> Interpolator<B> for NearestNeighborInterpolator
where
    B: Backend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    fn interpolate(&self, data: &Tensor<f32, B>, indices: Tensor<f32, B>) -> Tensor<f32, B> {
        dispatch_nearest(data, indices, self.bounds_policy.as_out_of_bounds_mode())
    }
}

pub(crate) fn interpolate_nearest_host<B>(
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
            1 => nearest_1d(&values, shape[0], coord[0], mode),
            2 => nearest_2d(&values, shape[0], shape[1], coord[0], coord[1], mode),
            3 => nearest_3d(
                &values, shape[0], shape[1], shape[2], coord[0], coord[1], coord[2], mode,
            ),
            4 => nearest_4d(
                &values, shape[0], shape[1], shape[2], shape[3], coord[0], coord[1], coord[2],
                coord[3], mode,
            ),
            _ => unreachable!(),
        });
    }

    Tensor::<f32, B>::from_slice_on([n_points], &out, &backend)
}

#[inline]
fn nearest_index(coord: f32, size: usize) -> Option<usize> {
    if coord < 0.0 || coord > (size.saturating_sub(1)) as f32 {
        None
    } else {
        Some((coord + 0.5).floor() as usize)
    }
}

#[inline]
fn clamp_nearest_index(coord: f32, size: usize) -> usize {
    (coord + 0.5)
        .floor()
        .clamp(0.0, size.saturating_sub(1) as f32) as usize
}

fn nearest_1d(values: &[f32], d0: usize, x: f32, mode: OutOfBoundsMode) -> f32 {
    let ix = match mode {
        OutOfBoundsMode::Clamp => clamp_nearest_index(x, d0),
        OutOfBoundsMode::ZeroPad => match nearest_index(x, d0) {
            Some(v) => v,
            None => return 0.0,
        },
    };
    values[ix]
}

fn nearest_2d(values: &[f32], d0: usize, d1: usize, x: f32, y: f32, mode: OutOfBoundsMode) -> f32 {
    let (ix, iy) = match mode {
        OutOfBoundsMode::Clamp => (clamp_nearest_index(x, d1), clamp_nearest_index(y, d0)),
        OutOfBoundsMode::ZeroPad => {
            let Some(ix) = nearest_index(x, d1) else { return 0.0 };
            let Some(iy) = nearest_index(y, d0) else { return 0.0 };
            (ix, iy)
        }
    };
    values[iy * d1 + ix]
}

fn nearest_3d(
    values: &[f32],
    d0: usize,
    d1: usize,
    d2: usize,
    x: f32,
    y: f32,
    z: f32,
    mode: OutOfBoundsMode,
) -> f32 {
    let (ix, iy, iz) = match mode {
        OutOfBoundsMode::Clamp => (
            clamp_nearest_index(x, d2),
            clamp_nearest_index(y, d1),
            clamp_nearest_index(z, d0),
        ),
        OutOfBoundsMode::ZeroPad => {
            let Some(ix) = nearest_index(x, d2) else { return 0.0 };
            let Some(iy) = nearest_index(y, d1) else { return 0.0 };
            let Some(iz) = nearest_index(z, d0) else { return 0.0 };
            (ix, iy, iz)
        }
    };
    values[(iz * d1 + iy) * d2 + ix]
}

#[allow(clippy::too_many_arguments)]
fn nearest_4d(
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
    let (ix, iy, iz, iw) = match mode {
        OutOfBoundsMode::Clamp => (
            clamp_nearest_index(x, d3),
            clamp_nearest_index(y, d2),
            clamp_nearest_index(z, d1),
            clamp_nearest_index(w, d0),
        ),
        OutOfBoundsMode::ZeroPad => {
            let Some(ix) = nearest_index(x, d3) else { return 0.0 };
            let Some(iy) = nearest_index(y, d2) else { return 0.0 };
            let Some(iz) = nearest_index(z, d1) else { return 0.0 };
            let Some(iw) = nearest_index(w, d0) else { return 0.0 };
            (ix, iy, iz, iw)
        }
    };
    values[(((iw * d1 + iz) * d2) + iy) * d3 + ix]
}
