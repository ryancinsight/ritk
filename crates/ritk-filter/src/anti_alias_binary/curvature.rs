//! Curvature calculation for the Anti-Alias Binary filter.

use super::MSQ_EPS;

/// CurvatureFlowFunction::ComputeUpdate at flat index f (clamped Neumann).
pub(crate) fn curvature(phi: &[f32], f: usize, dims: [usize; 3], ndim: usize) -> f32 {
    let [nz, ny, nx] = dims;
    let iz = f / (ny * nx);
    let r = f % (ny * nx);
    let iy = r / nx;
    let ix = r % nx;

    let idx = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;
    let g = |dz: isize, dy: isize, dx: isize| -> f32 {
        let z = (iz as isize + dz).clamp(0, nz as isize - 1) as usize;
        let y = (iy as isize + dy).clamp(0, ny as isize - 1) as usize;
        let x = (ix as isize + dx).clamp(0, nx as isize - 1) as usize;
        phi[idx(z, y, x)]
    };
    let c = phi[f];
    // first derivatives, second derivatives, cross derivatives (axes y,x[,z]).
    let fx = 0.5 * (g(0, 0, 1) - g(0, 0, -1));
    let fy = 0.5 * (g(0, 1, 0) - g(0, -1, 0));
    let fxx = g(0, 0, 1) - 2.0 * c + g(0, 0, -1);
    let fyy = g(0, 1, 0) - 2.0 * c + g(0, -1, 0);
    let fxy = 0.25 * (g(0, -1, -1) - g(0, -1, 1) - g(0, 1, -1) + g(0, 1, 1));
    if ndim == 2 {
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
