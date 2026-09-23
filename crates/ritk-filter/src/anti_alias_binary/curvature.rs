//! Curvature calculation for the Anti-Alias Binary filter.

use super::MSQ_EPS;
use crate::sparse_field::GridTopology;

/// `CurvatureFlowFunction::ComputeUpdate` at flat index f (clamped Neumann).
pub(crate) fn curvature(phi: &[f32], topo: &GridTopology, f: usize) -> f32 {
    let (iz, iy, ix) = topo.decode(f);
    let (zi, yi, xi) = (iz as isize, iy as isize, ix as isize);
    let g = |dz: isize, dy: isize, dx: isize| phi[topo.clamped_index(zi + dz, yi + dy, xi + dx)];
    let c = phi[f];
    // first derivatives, second derivatives, cross derivatives (axes y,x[,z]).
    let fx = 0.5 * (g(0, 0, 1) - g(0, 0, -1));
    let fy = 0.5 * (g(0, 1, 0) - g(0, -1, 0));
    let fxx = g(0, 0, 1) - 2.0 * c + g(0, 0, -1);
    let fyy = g(0, 1, 0) - 2.0 * c + g(0, -1, 0);
    let fxy = 0.25 * (g(0, -1, -1) - g(0, -1, 1) - g(0, 1, -1) + g(0, 1, 1));
    if topo.ndim() == 2 {
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
        (fx * fx * (fyy + fzz) + fy * fy * (fxx + fzz) + fz * fz * (fxx + fyy)
            - 2.0 * fx * fy * fxy
            - 2.0 * fx * fz * fxz
            - 2.0 * fy * fz * fyz)
            / msq
    }
}
