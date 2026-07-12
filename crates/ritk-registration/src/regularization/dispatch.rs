//! Rank dispatch and native spatial-operator kernels for regularizers.
//!
//! Routes regularizer loss computation to the correct rank-specific kernel
//! based on the displacement field's runtime rank. Only rank ∈ {4, 5} is
//! supported (2-D `[B, C, H, W]` and 3-D `[B, C, D, H, W]` displacement fields).
//!
//! The kernels are Coeus-native: they read the field as a contiguous host slice
//! and fuse the finite-difference stencil with the reduction, so no
//! intermediate gradient/Laplacian tensor is materialized. Boundary voxels use
//! forward differences padded with zero (gradient operators) or a zero border
//! (Laplacian operators), matching the reference AirLab formulation. All
//! arithmetic executes in the field's native precision `T` (no widen/narrow).

use coeus_core::{ComputeBackend, CpuAddressableStorage, Scalar};
use coeus_tensor::Tensor;

/// Run `f` over the field's contiguous host slice and shape.
///
/// The displacement fields regularizers receive are freshly constructed and
/// already contiguous; the `to_contiguous` branch keeps the kernels correct if
/// a non-contiguous view is ever passed, at the cost of one compaction copy.
#[inline]
fn with_field_data<T, B, R>(field: &Tensor<T, B>, f: impl FnOnce(&[T], &[usize]) -> R) -> R
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    if field.is_contiguous() {
        f(field.as_slice(), field.shape())
    } else {
        let contiguous = field.to_contiguous();
        f(contiguous.as_slice(), contiguous.shape())
    }
}

/// Rank assertion shared by every kernel entry point.
#[inline]
fn rank_dims<const N: usize>(shape: &[usize], op: &str) -> [usize; N] {
    assert_eq!(
        shape.len(),
        N,
        "{op}: displacement field must be rank {N}, got rank {} (shape {shape:?})",
        shape.len()
    );
    let mut dims = [0usize; N];
    dims.copy_from_slice(shape);
    dims
}

/// Dispatch bending-energy loss (mean squared Laplacian) to the field's rank.
#[inline]
pub fn dispatch_bending_energy<T, B>(field: &Tensor<T, B>, weight: f64) -> T
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    with_field_data(field, |data, shape| {
        laplacian_squared_mean(data, shape, "bending_energy") * T::from_f64(weight)
    })
}

/// Dispatch curvature loss to the field's rank.
///
/// Curvature and bending energy share the mean-squared-Laplacian kernel; they
/// differ only in their conventional default weight (see the regularizer types).
#[inline]
pub fn dispatch_curvature<T, B>(field: &Tensor<T, B>, weight: f64) -> T
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    with_field_data(field, |data, shape| {
        laplacian_squared_mean(data, shape, "curvature") * T::from_f64(weight)
    })
}

/// Dispatch diffusion loss (mean squared gradient) to the field's rank.
#[inline]
pub fn dispatch_diffusion<T, B>(field: &Tensor<T, B>, weight: f64) -> T
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    with_field_data(field, |data, shape| {
        gradient_squared_mean(data, shape, "diffusion") * T::from_f64(weight)
    })
}

/// Dispatch total-variation loss (mean gradient magnitude) to the field's rank.
#[inline]
pub fn dispatch_total_variation<T, B>(field: &Tensor<T, B>, weight: f64) -> T
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    with_field_data(field, |data, shape| {
        gradient_magnitude_mean(data, shape, "total_variation") * T::from_f64(weight)
    })
}

/// Dispatch elastic loss (membrane + divergence terms) to the field's rank.
#[inline]
pub fn dispatch_elastic<T, B>(field: &Tensor<T, B>, alpha: f64, beta: f64) -> T
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    with_field_data(field, |data, shape| {
        elastic(data, shape, T::from_f64(alpha), T::from_f64(beta))
    })
}

// ── Rank dispatch over the native kernels ────────────────────────────────────

#[inline]
fn laplacian_squared_mean<T: Scalar>(data: &[T], shape: &[usize], op: &str) -> T {
    match shape.len() {
        4 => {
            let [b, c, h, w] = rank_dims(shape, op);
            laplacian_squared_mean_planar(data, b * c, h, w)
        }
        5 => {
            let [b, c, d, h, w] = rank_dims(shape, op);
            laplacian_squared_mean_volumetric(data, b * c, d, h, w)
        }
        rank => panic!("{op}: displacement field must be rank 4 or 5, got rank {rank}"),
    }
}

#[inline]
fn gradient_squared_mean<T: Scalar>(data: &[T], shape: &[usize], op: &str) -> T {
    match shape.len() {
        4 => {
            let [b, c, h, w] = rank_dims(shape, op);
            gradient_squared_mean_planar(data, b * c, h, w)
        }
        5 => {
            let [b, c, d, h, w] = rank_dims(shape, op);
            gradient_squared_mean_volumetric(data, b * c, d, h, w)
        }
        rank => panic!("{op}: displacement field must be rank 4 or 5, got rank {rank}"),
    }
}

#[inline]
fn gradient_magnitude_mean<T: Scalar>(data: &[T], shape: &[usize], op: &str) -> T {
    match shape.len() {
        4 => {
            let [b, c, h, w] = rank_dims(shape, op);
            gradient_magnitude_mean_planar(data, b * c, h, w)
        }
        5 => {
            let [b, c, d, h, w] = rank_dims(shape, op);
            gradient_magnitude_mean_volumetric(data, b * c, d, h, w)
        }
        rank => panic!("{op}: displacement field must be rank 4 or 5, got rank {rank}"),
    }
}

#[inline]
fn elastic<T: Scalar>(data: &[T], shape: &[usize], alpha: T, beta: T) -> T {
    match shape.len() {
        4 => {
            let [b, c, h, w] = rank_dims(shape, "elastic");
            elastic_planar(data, b, c, h, w, alpha, beta)
        }
        5 => {
            let [b, c, d, h, w] = rank_dims(shape, "elastic");
            elastic_volumetric(data, b, c, d, h, w, alpha, beta)
        }
        rank => panic!("elastic: displacement field must be rank 4 or 5, got rank {rank}"),
    }
}

// ── Native finite-difference kernels (planar, `[BC, H, W]`) ───────────────────

/// Forward difference along an axis, zero-padded at the last index.
#[inline]
fn fwd_diff<T: Scalar>(data: &[T], offset: usize, local: usize, extent: usize, stride: usize) -> T {
    if local + 1 < extent {
        data[offset + stride] - data[offset]
    } else {
        T::zero()
    }
}

/// Mean over all `BC·H·W` voxels of the squared 5-point Laplacian.
///
/// Border voxels (`i∈{0,h-1}` or `j∈{0,w-1}`) contribute a zero Laplacian, so
/// the loop visits only the interior; the mean divides by the full voxel count.
fn laplacian_squared_mean_planar<T: Scalar>(data: &[T], bc: usize, h: usize, w: usize) -> T {
    let four = T::from_f64(4.0);
    let mut acc = T::zero();
    for plane in 0..bc {
        let base = plane * h * w;
        for i in 1..h.saturating_sub(1) {
            for j in 1..w.saturating_sub(1) {
                let o = base + i * w + j;
                let lap = data[o - 1] + data[o + 1] + data[o - w] + data[o + w] - four * data[o];
                acc += lap * lap;
            }
        }
    }
    acc / T::from_usize(bc * h * w)
}

/// Mean over all voxels of `|∇u|²` via forward differences (zero-padded edges).
fn gradient_squared_mean_planar<T: Scalar>(data: &[T], bc: usize, h: usize, w: usize) -> T {
    let mut acc = T::zero();
    for plane in 0..bc {
        let base = plane * h * w;
        for i in 0..h {
            for j in 0..w {
                let o = base + i * w + j;
                let gh = fwd_diff(data, o, i, h, w);
                let gw = fwd_diff(data, o, j, w, 1);
                acc += gh * gh + gw * gw;
            }
        }
    }
    acc / T::from_usize(bc * h * w)
}

/// Mean over all voxels of `|∇u| = √(g_h² + g_w²)` (isotropic total variation).
fn gradient_magnitude_mean_planar<T: Scalar>(data: &[T], bc: usize, h: usize, w: usize) -> T {
    let mut acc = T::zero();
    for plane in 0..bc {
        let base = plane * h * w;
        for i in 0..h {
            for j in 0..w {
                let o = base + i * w + j;
                let gh = fwd_diff(data, o, i, h, w);
                let gw = fwd_diff(data, o, j, w, 1);
                acc += (gh * gh + gw * gw).sqrt_val();
            }
        }
    }
    acc / T::from_usize(bc * h * w)
}

/// Elastic loss: `α·mean(|∇u|²) + β·mean((div u)²)`.
///
/// The divergence couples channel 0's height-gradient with channel 1's
/// width-gradient (`div u = ∂u_x/∂x + ∂u_y/∂y`), so the field must carry at
/// least two displacement channels; its mean is over `B·H·W` (one divergence
/// scalar per spatial site), distinct from the membrane mean over `B·C·H·W`.
fn elastic_planar<T: Scalar>(
    data: &[T],
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    alpha: T,
    beta: T,
) -> T {
    assert!(c >= 2, "elastic (2-D) requires ≥2 displacement channels, got {c}");
    let membrane_mean = gradient_squared_mean_planar(data, b * c, h, w);

    let mut div_acc = T::zero();
    let plane = h * w;
    for bb in 0..b {
        let ch0 = (bb * c) * plane;
        let ch1 = (bb * c + 1) * plane;
        for i in 0..h {
            for j in 0..w {
                let off = i * w + j;
                let gh0 = fwd_diff(data, ch0 + off, i, h, w);
                let gw1 = fwd_diff(data, ch1 + off, j, w, 1);
                let div = gh0 + gw1;
                div_acc += div * div;
            }
        }
    }
    let divergence_mean = div_acc / T::from_usize(b * h * w);
    membrane_mean * alpha + divergence_mean * beta
}

// ── Native finite-difference kernels (volumetric, `[BC, D, H, W]`) ────────────

fn laplacian_squared_mean_volumetric<T: Scalar>(
    data: &[T],
    bc: usize,
    d: usize,
    h: usize,
    w: usize,
) -> T {
    let six = T::from_f64(6.0);
    let hw = h * w;
    let dhw = d * hw;
    let mut acc = T::zero();
    for vol in 0..bc {
        let base = vol * dhw;
        for k in 1..d.saturating_sub(1) {
            for i in 1..h.saturating_sub(1) {
                for j in 1..w.saturating_sub(1) {
                    let o = base + k * hw + i * w + j;
                    let lap = data[o - 1]
                        + data[o + 1]
                        + data[o - w]
                        + data[o + w]
                        + data[o - hw]
                        + data[o + hw]
                        - six * data[o];
                    acc += lap * lap;
                }
            }
        }
    }
    acc / T::from_usize(bc * dhw)
}

fn gradient_squared_mean_volumetric<T: Scalar>(
    data: &[T],
    bc: usize,
    d: usize,
    h: usize,
    w: usize,
) -> T {
    let hw = h * w;
    let dhw = d * hw;
    let mut acc = T::zero();
    for vol in 0..bc {
        let base = vol * dhw;
        for k in 0..d {
            for i in 0..h {
                for j in 0..w {
                    let o = base + k * hw + i * w + j;
                    let gd = fwd_diff(data, o, k, d, hw);
                    let gh = fwd_diff(data, o, i, h, w);
                    let gw = fwd_diff(data, o, j, w, 1);
                    acc += gd * gd + gh * gh + gw * gw;
                }
            }
        }
    }
    acc / T::from_usize(bc * dhw)
}

fn gradient_magnitude_mean_volumetric<T: Scalar>(
    data: &[T],
    bc: usize,
    d: usize,
    h: usize,
    w: usize,
) -> T {
    let hw = h * w;
    let dhw = d * hw;
    let mut acc = T::zero();
    for vol in 0..bc {
        let base = vol * dhw;
        for k in 0..d {
            for i in 0..h {
                for j in 0..w {
                    let o = base + k * hw + i * w + j;
                    let gd = fwd_diff(data, o, k, d, hw);
                    let gh = fwd_diff(data, o, i, h, w);
                    let gw = fwd_diff(data, o, j, w, 1);
                    acc += (gd * gd + gh * gh + gw * gw).sqrt_val();
                }
            }
        }
    }
    acc / T::from_usize(bc * dhw)
}

fn elastic_volumetric<T: Scalar>(
    data: &[T],
    b: usize,
    c: usize,
    d: usize,
    h: usize,
    w: usize,
    alpha: T,
    beta: T,
) -> T {
    assert!(c >= 3, "elastic (3-D) requires ≥3 displacement channels, got {c}");
    let membrane_mean = gradient_squared_mean_volumetric(data, b * c, d, h, w);

    let hw = h * w;
    let dhw = d * hw;
    let mut div_acc = T::zero();
    for bb in 0..b {
        let ch0 = (bb * c) * dhw;
        let ch1 = (bb * c + 1) * dhw;
        let ch2 = (bb * c + 2) * dhw;
        for k in 0..d {
            for i in 0..h {
                for j in 0..w {
                    let off = k * hw + i * w + j;
                    let gd0 = fwd_diff(data, ch0 + off, k, d, hw);
                    let gh1 = fwd_diff(data, ch1 + off, i, h, w);
                    let gw2 = fwd_diff(data, ch2 + off, j, w, 1);
                    let div = gd0 + gh1 + gw2;
                    div_acc += div * div;
                }
            }
        }
    }
    let divergence_mean = div_acc / T::from_usize(b * dhw);
    membrane_mean * alpha + divergence_mean * beta
}

#[cfg(test)]
#[path = "tests_dispatch.rs"]
mod tests;
