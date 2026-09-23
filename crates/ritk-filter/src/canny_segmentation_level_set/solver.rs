//! The Canny step: `CannySegmentationLevelSetFunction::ComputeUpdate` as an
//! engine policy. The band machinery it runs on lives in
//! [`crate::sparse_field`].

use super::{spatial::GridHelper, CannySegmentationLevelSet, CGV, GRAD_EPS};
use crate::sparse_field::{evolve, SparseFieldConfig, SparseFieldStep};

/// ITK SparseField default `NumberOfLayers` (Canny does not override it).
const NUMBER_OF_LAYERS: usize = 2;

/// `SegmentationLevelSetFunction::ComputeUpdate` over the prebuilt speed `P` and
/// advection `A` fields.
struct CannyStep<'a> {
    gh: GridHelper,
    p: &'a [f64],
    adv: &'a [Vec<f64>],
    curv_w: f64,
    prop_w: f64,
    adv_w: f64,
    wave_dt: f64,
}

impl SparseFieldStep<f64> for CannyStep<'_> {
    fn stage(&self, phi: &[f64], active: &[usize]) -> (Vec<f64>, f64) {
        let mut updates = Vec::with_capacity(active.len());
        let mut maxc = 0.0f64;
        let mut maxp = 0.0f64;
        let mut maxa = 0.0f64;
        for &f in active {
            let (update, curv, prop, max_adv) = self.speed(phi, f);
            maxc = maxc.max(curv);
            maxp = maxp.max(prop);
            maxa = maxa.max(max_adv);
            updates.push(update);
        }
        // ComputeGlobalTimeStep.
        let dt = if maxc > 0.0 {
            if maxa + maxp > 0.0 {
                (self.wave_dt / (maxa + maxp)).min(self.wave_dt / maxc)
            } else {
                self.wave_dt / maxc
            }
        } else if maxa + maxp > 0.0 {
            self.wave_dt / (maxa + maxp)
        } else {
            0.0
        };
        (updates, dt)
    }
}

impl CannyStep<'_> {
    /// Returns `(update, |weighted_curv|, |weighted_prop|, max|weighted_adv_i|)`.
    fn speed(&self, phi: &[f64], f: usize) -> (f64, f64, f64, f64) {
        let gh = self.gh;
        let (curv_w, prop_w, adv_w) = (self.curv_w, self.prop_w, self.adv_w);
        let (iz, iy, ix) = gh.decode(f);
        let (zi, yi, xi) = (iz as isize, iy as isize, ix as isize);
        let c = phi[f];
        let g = |dz: isize, dy: isize, dx: isize| gh.gphi(phi, zi + dz, yi + dy, xi + dx);
        // φ derivatives (axis 0=x, 1=y, 2=z).
        let dxf = g(0, 0, 1) - c;
        let dxb = c - g(0, 0, -1);
        let dyf = g(0, 1, 0) - c;
        let dyb = c - g(0, -1, 0);
        let fx = 0.5 * (g(0, 0, 1) - g(0, 0, -1));
        let fy = 0.5 * (g(0, 1, 0) - g(0, -1, 0));
        let fxx = g(0, 0, 1) - 2.0 * c + g(0, 0, -1);
        let fyy = g(0, 1, 0) - 2.0 * c + g(0, -1, 0);
        let fxy = 0.25 * (g(0, -1, -1) - g(0, -1, 1) - g(0, 1, -1) + g(0, 1, 1));

        // Sample P / A at the surface offset.
        let (cz, cy, cx) = gh.surface_offset_coords(phi, f, zi, yi, xi);
        let prop = gh.interp(self.p, cz, cy, cx);
        let ax = gh.interp(&self.adv[0], cz, cy, cx);
        let ay = gh.interp(&self.adv[1], cz, cy, cx);
        let az = if gh.ndim() == 3 {
            gh.interp(&self.adv[2], cz, cy, cx)
        } else {
            0.0
        };

        // ── Curvature term (ComputeCurvatureTerm) ────────────────────────────
        let (curv, dzf, dzb);
        if gh.ndim() == 2 {
            let gm2 = fx * fx + fy * fy + GRAD_EPS;
            curv = (fxx * fy * fy + fyy * fx * fx - 2.0 * fx * fy * fxy) / gm2;
            dzf = 0.0;
            dzb = 0.0;
        } else {
            let fz = 0.5 * (g(1, 0, 0) - g(-1, 0, 0));
            let fzz = g(1, 0, 0) - 2.0 * c + g(-1, 0, 0);
            let fxz = 0.25 * (g(-1, 0, -1) - g(-1, 0, 1) - g(1, 0, -1) + g(1, 0, 1));
            let fyz = 0.25 * (g(-1, -1, 0) - g(-1, 1, 0) - g(1, -1, 0) + g(1, 1, 0));
            let gm2 = fx * fx + fy * fy + fz * fz + GRAD_EPS;
            curv = (fx * fx * (fyy + fzz) + fy * fy * (fxx + fzz) + fz * fz * (fxx + fyy)
                - 2.0 * fx * fy * fxy
                - 2.0 * fx * fz * fxz
                - 2.0 * fy * fz * fyz)
                / gm2;
            dzf = g(1, 0, 0) - c;
            dzb = c - g(-1, 0, 0);
        }
        let curv_term = curv * curv_w;

        // ── Propagation term (Godunov upwind in sign of P) ───────────────────
        let prop_term = prop_w * prop;
        let pg = if prop_term > 0.0 {
            dxb.max(0.0).powi(2)
                + dxf.min(0.0).powi(2)
                + dyb.max(0.0).powi(2)
                + dyf.min(0.0).powi(2)
                + dzb.max(0.0).powi(2)
                + dzf.min(0.0).powi(2)
        } else {
            dxb.min(0.0).powi(2)
                + dxf.max(0.0).powi(2)
                + dyb.min(0.0).powi(2)
                + dyf.max(0.0).powi(2)
                + dzb.min(0.0).powi(2)
                + dzf.max(0.0).powi(2)
        };
        let propagation = prop_term * pg.sqrt();

        // ── Advection term (simple upwind per component) ─────────────────────
        let mut adv_term =
            ax * (if ax > 0.0 { dxb } else { dxf }) + ay * (if ay > 0.0 { dyb } else { dyf });
        if gh.ndim() == 3 {
            adv_term += az * (if az > 0.0 { dzb } else { dzf });
        }
        adv_term *= adv_w;

        let update = curv_term - propagation - adv_term;
        let max_adv = (adv_w * ax.abs())
            .max(adv_w * ay.abs())
            .max(adv_w * az.abs());
        (update, curv_term.abs(), prop_term.abs(), max_adv)
    }
}

impl CannySegmentationLevelSet {
    pub(crate) fn run(
        &self,
        shifted: &[f64],
        p: &[f64],
        adv: &[Vec<f64>],
        dims: [usize; 3],
    ) -> Vec<f64> {
        let gh = GridHelper::new(dims);
        let step = CannyStep {
            gh,
            p,
            adv,
            curv_w: self.curvature_scaling as f64,
            prop_w: self.propagation_scaling as f64,
            adv_w: self.advection_scaling as f64,
            wave_dt: 1.0 / (2.0 * gh.ndim() as f64),
        };
        evolve(
            shifted,
            &SparseFieldConfig {
                dims,
                number_of_layers: NUMBER_OF_LAYERS,
                constant_gradient: CGV,
                iterations: self.number_of_iterations,
                max_rms_error: self.max_rms_error as f64,
            },
            &step,
        )
    }
}
