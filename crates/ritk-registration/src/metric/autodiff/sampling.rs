//! Differentiable linear/trilinear image sampling on Coeus autograd `Var`s.
//!
//! Interpolates an image at continuous coordinates such that the gradient of
//! the sampled values flows back to the **coordinate** leaves. This is the step
//! that makes a registration loss a function of the transform parameters: the
//! transform produces continuous coordinates, sampling reads the moving image
//! at them, and the coordinate gradient drives the optimizer.
//!
//! # Mechanism (why the coordinate gradient is correct)
//!
//! Along one axis, linear interpolation is `s[i0]·(1 − f) + s[i1]·f` where
//! `i0 = clamp(⌊x⌋)`, `i1 = clamp(⌊x⌋+1)`, and `f = x − ⌊x⌋`. The corner
//! *indices* are piecewise-constant in `x` (Coeus `gather` treats its index as
//! non-differentiable — gradient flows only through the gathered values), so
//! the coordinate gradient flows entirely through the fractional weight
//! (`∂f/∂x = 1`). In 1-D this gives `∂out/∂x = s[i1] − s[i0]` (the local
//! slope); in 3-D the trilinear weight of each of the eight corners is the
//! product of the three per-axis weights, so the per-axis coordinate gradient
//! is the corresponding trilinear finite difference.
//!
//! Each per-axis weight `f` is built as `sub(coord_axis, ⌊coord_axis⌋)`, keeping
//! that coordinate on the autograd tape; `⌊·⌋` and the clamped integer corner
//! indices are materialized as constant (`requires_grad = false`) `Var`s.
//! Reading a coordinate's host values to compute those constants does not
//! detach it from the tape — the differentiable path `coord → f → out` is
//! unbroken (migration gate #3). `AxisInterp` is the single shared per-axis
//! computation both samplers use (DRY: 1-D and each of the three trilinear axes
//! are the same operation).

use coeus_autograd::{add, gather, mul, sub, Var};
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Per-axis linear-interpolation decomposition: the two differentiable corner
/// weights `w0 = 1 − f`, `w1 = f` (as `Var`s on the coordinate tape) and the
/// two independently-clamped integer corner indices (as plain `usize`s, ready
/// to be combined into flat gather indices).
struct AxisInterp<T: Scalar, B>
where
    B: ComputeBackend + BackendOps<T> + Default,
{
    w0: Var<T, B>,
    w1: Var<T, B>,
    idx0: Vec<usize>,
    idx1: Vec<usize>,
}

/// Decompose one coordinate axis (`coords`, shape `[N]`) against an `extent`
/// (`[0, extent-1]`) into its corner weights and clamped corner indices.
///
/// The weight derives from the unclamped floor (so the mapping stays
/// continuous), while the two corner indices are clamped independently
/// (matching the RITK trilinear reference's edge behavior). Reads `coords`'
/// host values to build the constant floor/index `Var`s; `coords` itself stays
/// on the tape via `w1 = coords − ⌊coords⌋`.
fn axis_interp<T, B>(coords: &Var<T, B>, extent: usize, backend: &B) -> AxisInterp<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let shape = coords.tensor.shape();
    assert_eq!(shape.len(), 1, "axis_interp: coords must be 1-D");
    let n = shape[0];
    assert!(extent > 0, "axis_interp: extent must be non-zero");
    let max_index = extent - 1;

    let coord_vals = coords.tensor.as_slice();
    let mut floor_vals = Vec::with_capacity(n);
    let mut idx0 = Vec::with_capacity(n);
    let mut idx1 = Vec::with_capacity(n);
    for &c in coord_vals {
        let floor = <T as Scalar>::to_f64(c).floor();
        floor_vals.push(T::from_f64(floor));
        idx0.push(clamp_index(floor, max_index));
        idx1.push(clamp_index(floor + 1.0, max_index));
    }

    let floor_const = Var::new(Tensor::from_slice_on([n], &floor_vals, backend), false);
    let ones = Var::new(Tensor::full_on([n], T::one(), backend), false);
    let w1 = sub(coords, &floor_const);
    let w0 = sub(&ones, &w1);

    AxisInterp { w0, w1, idx0, idx1 }
}

/// Build a constant index `Var` of shape `[N]` from flat `usize` indices.
fn index_var<T, B>(indices: &[usize], backend: &B) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
{
    let vals: Vec<T> = indices.iter().map(|&i| T::from_f64(i as f64)).collect();
    Var::new(Tensor::from_slice_on([indices.len()], &vals, backend), false)
}

/// Differentiable linear interpolation of a 1-D `signal` (shape `[L]`) at
/// continuous `coords` (shape `[N]`), returning sampled intensities (shape
/// `[N]`). Out-of-range coordinates clamp to the signal edges; the weight
/// derives from the unclamped floor, so the mapping is continuous.
///
/// The result's autograd graph links back to `coords` (through the fractional
/// weights) and to `signal` (through `gather`, if `signal` requires grad).
///
/// # Panics
///
/// Panics if `signal` or `coords` is not 1-D, if `signal` is empty, or if
/// either tensor is non-contiguous — all caller invariants, not input-data
/// errors.
pub fn sample_linear_1d<T, B>(signal: &Var<T, B>, coords: &Var<T, B>) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let signal_shape = signal.tensor.shape();
    assert_eq!(signal_shape.len(), 1, "sample_linear_1d: signal must be 1-D");
    let len = signal_shape[0];
    assert!(len > 0, "sample_linear_1d: signal must be non-empty");

    let axis = axis_interp(coords, len, &backend);
    let v0 = gather(signal, 0, &index_var(&axis.idx0, &backend));
    let v1 = gather(signal, 0, &index_var(&axis.idx1, &backend));
    add(&mul(&v0, &axis.w0), &mul(&v1, &axis.w1))
}

/// Differentiable trilinear interpolation of a flattened 3-D `signal` (shape
/// `[Z·Y·X]`, row-major with `dims = [Z, Y, X]`) at continuous per-axis
/// coordinates `coords_z`, `coords_y`, `coords_x` (each shape `[N]`, in voxel
/// index space), returning sampled intensities (shape `[N]`).
///
/// Coordinates are supplied per axis (rather than as an `[N, 3]` tensor) so the
/// gradient flows to three independent coordinate leaves without depending on a
/// differentiable column-slice op; a transform producing `[N, 3]` splits into
/// three columns at its boundary. Out-of-range coordinates clamp per axis
/// independently (matching the RITK trilinear reference); weights derive from
/// the unclamped floors.
///
/// The result's autograd graph links back to each coordinate axis (through the
/// per-axis fractional weights) and to `signal` (through `gather`).
///
/// # Panics
///
/// Panics if `signal` is not 1-D of length `Z·Y·X`, if any coordinate vector is
/// not 1-D, if the three coordinate vectors differ in length, or on
/// non-contiguous input — all caller invariants.
pub fn sample_trilinear<T, B>(
    signal: &Var<T, B>,
    dims: [usize; 3],
    coords_z: &Var<T, B>,
    coords_y: &Var<T, B>,
    coords_x: &Var<T, B>,
) -> Var<T, B>
where
    T: Scalar,
    B: ComputeBackend + BackendOps<T> + Default,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let [nz, ny, nx] = dims;
    let expected = nz * ny * nx;
    let signal_shape = signal.tensor.shape();
    assert_eq!(signal_shape.len(), 1, "sample_trilinear: signal must be flat 1-D");
    assert_eq!(
        signal_shape[0], expected,
        "sample_trilinear: signal length must equal Z·Y·X"
    );
    let n = coords_z.tensor.shape().first().copied().unwrap_or(0);
    assert_eq!(coords_y.tensor.shape(), [n], "coords_y length must match coords_z");
    assert_eq!(coords_x.tensor.shape(), [n], "coords_x length must match coords_z");

    let az = axis_interp(coords_z, nz, &backend);
    let ay = axis_interp(coords_y, ny, &backend);
    let ax = axis_interp(coords_x, nx, &backend);

    let stride_z = ny * nx;
    let stride_y = nx;

    // Accumulate the eight trilinear corners. Corner (cz, cy, cx) contributes
    // gathered_value · (wz · wy · wx); the flat gather index combines the three
    // clamped per-axis corner indices.
    let mut acc: Option<Var<T, B>> = None;
    for (iz, wz) in [(&az.idx0, &az.w0), (&az.idx1, &az.w1)] {
        for (iy, wy) in [(&ay.idx0, &ay.w0), (&ay.idx1, &ay.w1)] {
            for (ix, wx) in [(&ax.idx0, &ax.w0), (&ax.idx1, &ax.w1)] {
                let flat: Vec<usize> = (0..n)
                    .map(|k| iz[k] * stride_z + iy[k] * stride_y + ix[k])
                    .collect();
                let value = gather(signal, 0, &index_var(&flat, &backend));
                let weight = mul(&mul(wz, wy), wx);
                let term = mul(&value, &weight);
                acc = Some(match acc {
                    Some(prev) => add(&prev, &term),
                    None => term,
                });
            }
        }
    }

    acc.expect("trilinear always accumulates eight corners")
}

/// Clamp a floored coordinate to a valid `[0, max_index]` signal index.
#[inline]
fn clamp_index(floor: f64, max_index: usize) -> usize {
    if floor <= 0.0 {
        0
    } else if floor >= max_index as f64 {
        max_index
    } else {
        floor as usize
    }
}

#[cfg(test)]
#[path = "tests_sampling.rs"]
mod tests;
