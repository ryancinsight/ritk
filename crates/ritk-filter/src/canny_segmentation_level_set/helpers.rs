//! Helpers for index decoding, neighbor searching, and multilinear interpolation.

use super::MIN_NORM;

#[derive(Clone, Copy, Debug)]
pub(crate) struct GridHelper {
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
    pub ndim: usize,
}

impl GridHelper {
    #[inline]
    pub fn new(dims: [usize; 3]) -> Self {
        let [nz, ny, nx] = dims;
        let ndim = if nz == 1 { 2 } else { 3 };
        Self { nz, ny, nx, ndim }
    }

    #[inline]
    pub fn idx(&self, iz: usize, iy: usize, ix: usize) -> usize {
        iz * self.ny * self.nx + iy * self.nx + ix
    }

    #[inline]
    pub fn decode(&self, f: usize) -> (usize, usize, usize) {
        let iz = f / (self.ny * self.nx);
        let r = f % (self.ny * self.nx);
        (iz, r / self.nx, r % self.nx)
    }

    #[inline]
    pub fn neighbor(&self, f: usize, off: (isize, isize, isize)) -> Option<usize> {
        let (iz, iy, ix) = self.decode(f);
        let z = iz as isize + off.0;
        let y = iy as isize + off.1;
        let x = ix as isize + off.2;
        if z >= 0
            && y >= 0
            && x >= 0
            && z < self.nz as isize
            && y < self.ny as isize
            && x < self.nx as isize
        {
            Some(self.idx(z as usize, y as usize, x as usize))
        } else {
            None
        }
    }

    #[inline]
    pub fn gphi(&self, phi: &[f64], iz: isize, iy: isize, ix: isize) -> f64 {
        let z = iz.clamp(0, self.nz as isize - 1) as usize;
        let y = iy.clamp(0, self.ny as isize - 1) as usize;
        let x = ix.clamp(0, self.nx as isize - 1) as usize;
        phi[self.idx(z, y, x)]
    }

    /// Multilinear sample of a scalar field at continuous (cz, cy, cx).
    pub fn interp(&self, arr: &[f64], cz: f64, cy: f64, cx: f64) -> f64 {
        let cl = |v: f64, hi: usize| v.clamp(0.0, hi as f64 - 1.0);
        let cz = cl(cz, self.nz);
        let cy = cl(cy, self.ny);
        let cx = cl(cx, self.nx);
        let z0 = cz.floor() as usize;
        let y0 = cy.floor() as usize;
        let x0 = cx.floor() as usize;
        let z1 = (z0 + 1).min(self.nz - 1);
        let y1 = (y0 + 1).min(self.ny - 1);
        let x1 = (x0 + 1).min(self.nx - 1);
        let fz = cz - z0 as f64;
        let fy = cy - y0 as f64;
        let fx = cx - x0 as f64;
        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        let c00 = lerp(arr[self.idx(z0, y0, x0)], arr[self.idx(z0, y0, x1)], fx);
        let c01 = lerp(arr[self.idx(z0, y1, x0)], arr[self.idx(z0, y1, x1)], fx);
        let c0 = lerp(c00, c01, fy);
        let c10 = lerp(arr[self.idx(z1, y0, x0)], arr[self.idx(z1, y0, x1)], fx);
        let c11 = lerp(arr[self.idx(z1, y1, x0)], arr[self.idx(z1, y1, x1)], fx);
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
        let oz = if self.ndim == 3 {
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
