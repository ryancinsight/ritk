//! Hessian operation family for the recursive Gaussian filter.

use super::iir::DericheCoefficients;
use super::{apply_deriche_1d, DerivativeOrder};

/// Compute all 6 independent Hessian components at every voxel using the
/// Deriche IIR recursion — matching ITK `HessianRecursiveGaussianImageFilter`.
///
/// For each axis `d` (z=0, y=1, x=2):
/// - H_{dd}: `second_order` Deriche along axis `d`, `zero_order` along the
///   other two, divided by `spacing[d]²`.
/// - H_{di} (d<i): `first_order` along `d`, `first_order` along `i`,
///   `zero_order` along the remaining axis, divided by `spacing[d]·spacing[i]`.
///
/// Output layout per voxel: `[Hzz, Hzy, Hzx, Hyy, Hyx, Hxx]` — identical to
/// `compute_hessian` in `vesselness/hessian/mod.rs` so callers are drop-in.
///
/// Evidence tier: matches ITK source `itkHessianRecursiveGaussianImageFilter.hxx`
/// structure; verified against LoG (sum of diagonal = Laplacian via IIR) and
/// differential tests against FD Hessian at σ=3.0 (large σ where both converge).
pub(crate) fn compute_hessian_iir(
    vals: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    sigma: f64,
) -> Vec<[f32; 6]> {
    let n = vals.len();

    // Helper: apply one IIR pass along `axis` with chosen `order`.
    let pass = |buf: &[f32], axis: usize, order: DerivativeOrder| -> Vec<f32> {
        let pixel_sigma = sigma / spacing[axis];
        let coeffs = match order {
            DerivativeOrder::Zero => DericheCoefficients::from_sigma(pixel_sigma),
            DerivativeOrder::First => DericheCoefficients::first_order(pixel_sigma),
            DerivativeOrder::Second => DericheCoefficients::second_order(pixel_sigma),
        };
        apply_deriche_1d(buf, dims, axis, &coeffs, pixel_sigma)
    };

    // Diagonal H_{dd}: second_order along d, zero_order along others.
    // Divided by spacing[d]² for physical units.
    let hessian_diag = |d: usize| -> Vec<f32> {
        let others: [usize; 2] = match d {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let mut buf = pass(vals, others[0], DerivativeOrder::Zero);
        buf = pass(&buf, others[1], DerivativeOrder::Zero);
        buf = pass(&buf, d, DerivativeOrder::Second);
        let inv = (1.0 / (spacing[d] * spacing[d])) as f32;
        for v in buf.iter_mut() {
            *v *= inv;
        }
        buf
    };

    // Cross H_{di} (d < i): first_order along d, first_order along i, zero_order along k.
    // Divided by spacing[d]·spacing[i] for physical units.
    let hessian_cross = |d: usize, i: usize| -> Vec<f32> {
        // k is the remaining axis: (0,1)→2, (0,2)→1, (1,2)→0.
        let k = 3 - d - i;
        let mut buf = pass(vals, k, DerivativeOrder::Zero);
        buf = pass(&buf, d, DerivativeOrder::First);
        buf = pass(&buf, i, DerivativeOrder::First);
        let inv = (1.0 / (spacing[d] * spacing[i])) as f32;
        for v in buf.iter_mut() {
            *v *= inv;
        }
        buf
    };

    let hzz = hessian_diag(0);
    let hzy = hessian_cross(0, 1);
    let hzx = hessian_cross(0, 2);
    let hyy = hessian_diag(1);
    let hyx = hessian_cross(1, 2);
    let hxx = hessian_diag(2);

    // Pack into [Hzz, Hzy, Hzx, Hyy, Hyx, Hxx] per voxel.
    let mut out = vec![[0.0_f32; 6]; n];
    for i in 0..n {
        out[i] = [hzz[i], hzy[i], hzx[i], hyy[i], hyx[i], hxx[i]];
    }
    out
}
