//! `f64` difference and interpolation stencils over the shared [`GridTopology`].
//!
//! Index arithmetic, neighbour lookup and the Neumann clamp live in
//! [`crate::sparse_field::GridTopology`] because both SparseField filters need
//! them; what is left here is the `f64`-only stencil arithmetic of
//! `itkLevelSetFunction` (`CalculateChange` and its
//! `InterpolateSurfaceLocation` samples).

use super::MIN_NORM;
use crate::sparse_field::GridTopology;

#[derive(Clone, Copy, Debug)]
pub(crate) struct GridHelper {
    topo: GridTopology,
}

impl GridHelper {
    #[inline]
    pub fn new(dims: [usize; 3]) -> Self {
        Self {
            topo: GridTopology::new(dims),
        }
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.topo.ndim()
    }

    #[inline]
    pub fn decode(&self, f: usize) -> (usize, usize, usize) {
        self.topo.decode(f)
    }

    /// φ sampled at integer coordinates, each axis clamped into the volume.
    #[inline]
    pub fn gphi(&self, phi: &[f64], iz: isize, iy: isize, ix: isize) -> f64 {
        phi[self.topo.clamped_index(iz, iy, ix)]
    }

    /// Multilinear sample of a scalar field at continuous (cz, cy, cx).
    pub fn interp(&self, arr: &[f64], cz: f64, cy: f64, cx: f64) -> f64 {
        let (nz, ny, nx) = (self.topo.nz(), self.topo.ny(), self.topo.nx());
        let cl = |v: f64, hi: usize| v.clamp(0.0, hi as f64 - 1.0);
        let cz = cl(cz, nz);
        let cy = cl(cy, ny);
        let cx = cl(cx, nx);
        let z0 = cz.floor() as usize;
        let y0 = cy.floor() as usize;
        let x0 = cx.floor() as usize;
        let z1 = (z0 + 1).min(nz - 1);
        let y1 = (y0 + 1).min(ny - 1);
        let x1 = (x0 + 1).min(nx - 1);
        let fz = cz - z0 as f64;
        let fy = cy - y0 as f64;
        let fx = cx - x0 as f64;
        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let c00 = lerp(
            arr[self.topo.idx(z0, y0, x0)],
            arr[self.topo.idx(z0, y0, x1)],
            fx,
        );
        let c01 = lerp(
            arr[self.topo.idx(z0, y1, x0)],
            arr[self.topo.idx(z0, y1, x1)],
            fx,
        );
        let c0 = lerp(c00, c01, fy);
        let c10 = lerp(
            arr[self.topo.idx(z1, y0, x0)],
            arr[self.topo.idx(z1, y0, x1)],
            fx,
        );
        let c11 = lerp(
            arr[self.topo.idx(z1, y1, x0)],
            arr[self.topo.idx(z1, y1, x1)],
            fx,
        );
        let c1 = lerp(c10, c11, fy);
        lerp(c0, c1, fz)
    }

    /// InterpolateSurfaceLocation offset-sampled values.
    /// Returns (cz, cy, cx) continuous coordinates.
    pub fn surface_offset_coords(
        &self,
        phi: &[f64],
        f: usize,
        zi: isize,
        yi: isize,
        xi: isize,
    ) -> (f64, f64, f64) {
        let c = phi[f];
        if c == 0.0 {
            return (zi as f64, yi as f64, xi as f64);
        }
        let ox = self.off_axis(
            c,
            self.gphi(phi, zi, yi, xi + 1),
            self.gphi(phi, zi, yi, xi - 1),
        );
        let oy = self.off_axis(
            c,
            self.gphi(phi, zi, yi + 1, xi),
            self.gphi(phi, zi, yi - 1, xi),
        );
        let oz = if self.ndim() == 3 {
            self.off_axis(
                c,
                self.gphi(phi, zi + 1, yi, xi),
                self.gphi(phi, zi - 1, yi, xi),
            )
        } else {
            0.0
        };
        let norm = ox * ox + oy * oy + oz * oz + MIN_NORM;
        let cx = xi as f64 - ox * c / norm;
        let cy = yi as f64 - oy * c / norm;
        let cz = zi as f64 - oz * c / norm;
        (cz, cy, cx)
    }

    #[inline]
    fn off_axis(&self, c: f64, fwd: f64, bwd: f64) -> f64 {
        if fwd * bwd >= 0.0 {
            let df = fwd - c;
            let db = c - bwd;
            if df.abs() > db.abs() {
                df
            } else {
                db
            }
        } else if fwd * c < 0.0 {
            fwd - c
        } else {
            c - bwd
        }
    }
}
