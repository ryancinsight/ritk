//! Advection field construction for the Canny segmentation level set.

/// `A[axis][f] = P[f] · ∂P/∂axis` with central interior / one-sided boundary
/// differences (`numpy.gradient`, unit spacing). `axis` ∈ {0=x, 1=y, 2=z}.
pub(crate) fn advection_field(p: &[f64], dims: [usize; 3]) -> Vec<Vec<f64>> {
    let [nz, ny, nx] = dims;
    let idx = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;
    let ndim = if nz == 1 { 2 } else { 3 };
    let mut adv = vec![vec![0.0f64; nz * ny * nx]; ndim];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let f = idx(z, y, x);
                let pv = p[f];
                // x-derivative (axis 0).
                adv[0][f] = pv
                    * grad_1d(
                        p,
                        idx(z, y, x.saturating_sub(1)),
                        f,
                        idx(z, y, (x + 1).min(nx - 1)),
                        x,
                        nx,
                    );
                // y-derivative (axis 1).
                adv[1][f] = pv
                    * grad_1d(
                        p,
                        idx(z, y.saturating_sub(1), x),
                        f,
                        idx(z, (y + 1).min(ny - 1), x),
                        y,
                        ny,
                    );
                if ndim == 3 {
                    adv[2][f] = pv
                        * grad_1d(
                            p,
                            idx(z.saturating_sub(1), y, x),
                            f,
                            idx((z + 1).min(nz - 1), y, x),
                            z,
                            nz,
                        );
                }
            }
        }
    }
    adv
}

/// One-axis `numpy.gradient`: central `(next−prev)/2` in the interior,
/// one-sided `next−center` / `center−prev` at the boundaries.
#[inline]
fn grad_1d(p: &[f64], prev: usize, center: usize, next: usize, i: usize, len: usize) -> f64 {
    if i == 0 {
        p[next] - p[center]
    } else if i == len - 1 {
        p[center] - p[prev]
    } else {
        0.5 * (p[next] - p[prev])
    }
}
