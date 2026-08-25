//! Iterative (coordinate-descent) inversion of a displacement field
//! (`itk::IterativeInverseDisplacementFieldImageFilter` /
//! `sitk.IterativeInverseDisplacementField`).
//!
//! # Mathematical Specification
//!
//! Distinct from [`InvertDisplacementField`](crate::InvertDisplacementField)
//! (Chen et al. fixed point). Given a forward field `u`, it seeks `v` with
//! `(x + v) + u(x + v) = x`. The first guess warps the negated field by itself,
//! `v₀(x) = −u(x − u(x))`; then for each voxel a greedy line search over the
//! mapped point `p = x + v` minimizes `—–p + u(p) − x—–` by trying `p ± step` along
//! each physical axis (`step = spacing₀`, halved whenever a pass makes no move)
//! for `m_NumberOfIterations` iterations, stopping early if the error drops below
//! `stop_value`.
//!
//! `u` is sampled with the shared ITK-faithful vector linear interpolation
//! (`interp_component`); internal
//! arithmetic is `f64`, so the result is float-exact to
//! `sitk.IterativeInverseDisplacementField`.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

use crate::grid_geometry::CartesianGridGeometry;
use crate::invert_displacement::interp_component;

/// Parameters and entry point for iterative displacement-field inversion.
#[derive(Debug, Clone)]
pub struct IterativeInverseDisplacementField {
    /// Line-search iterations per voxel (ITK/sitk default 5).
    pub number_of_iterations: usize,
    /// Stop a voxel's search once its error falls below this (ITK default 0.0).
    pub stop_value: f64,
}

impl Default for IterativeInverseDisplacementField {
    fn default() -> Self {
        Self {
            number_of_iterations: 5,
            stop_value: 0.0,
        }
    }
}

impl IterativeInverseDisplacementField {
    /// Invert the field given as world components `(dx, dy, dz)`. Returns the
    /// inverted components `(dx, dy, dz)`.
    ///
    /// The search runs in physical space, so the grid's direction cosines
    /// participate in every index/world conversion: a voxel's physical position
    /// is `origin + D S index`, and a physical point's continuous index is
    /// `S^-1 D^-1 (point - origin)`. An oblique acquisition therefore inverts
    /// about its own axes rather than the index axes.
    ///
    /// # Errors
    ///
    /// Returns an error when `dx`'s coordinate map is not Cartesian, or when
    /// its direction matrix is singular.
    pub fn apply<B: Backend>(
        &self,
        dx: &Image<f32, B, 3>,
        dy: &Image<f32, B, 3>,
        dz: &Image<f32, B, 3>,
    ) -> anyhow::Result<crate::DisplacementComponents<B>> {
        let (ux, dims) = extract_vec_infallible(dx);
        let (uy, _) = extract_vec_infallible(dy);
        let (uz, _) = extract_vec_infallible(dz);
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let geometry = CartesianGridGeometry::new(
            dx.origin(),
            dx.spacing(),
            dx.direction(),
            dx.coordinate_map(),
        )?;
        // ITK uses the x-axis spacing as the search step; spacing is indexed by
        // tensor axis, so the x axis is index 2.
        let step0 = dx.spacing()[2];

        let uxd: Vec<f64> = ux.iter().map(|&v| v as f64).collect();
        let uyd: Vec<f64> = uy.iter().map(|&v| v as f64).collect();
        let uzd: Vec<f64> = uz.iter().map(|&v| v as f64).collect();
        let negx: Vec<f64> = uxd.iter().map(|&v| -v).collect();
        let negy: Vec<f64> = uyd.iter().map(|&v| -v).collect();
        let negz: Vec<f64> = uzd.iter().map(|&v| -v).collect();

        // Physical point of a voxel and its continuous index components. Both
        // honour the direction cosines; see `CartesianGridGeometry`.
        let phys = |z: usize, y: usize, x: usize| {
            let p = geometry.point([z as f64, y as f64, x as f64]);
            (p[0], p[1], p[2])
        };
        let idx = |px: f64, py: f64, pz: f64| {
            let i = geometry.index([px, py, pz]);
            (i[0], i[1], i[2])
        };
        // u(point) → world (cx-component, cy, cz) = (interp ux, interp uy, interp uz).
        let eval = |ud: (&[f64], &[f64], &[f64]), px: f64, py: f64, pz: f64| {
            let (cz, cy, cx) = idx(px, py, pz);
            (
                interp_component(ud.0, dims, cz, cy, cx),
                interp_component(ud.1, dims, cz, cy, cx),
                interp_component(ud.2, dims, cz, cy, cx),
            )
        };

        // First guess: v₀(voxel) = neg warped by neg = interp(neg, x + neg(x)).
        let mut vx = vec![0.0f64; n];
        let mut vy = vec![0.0f64; n];
        let mut vz = vec![0.0f64; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = (z * ny + y) * nx + x;
                    let (px, py, pz) = phys(z, y, x);
                    let (gx, gy, gz) = eval(
                        (&negx, &negy, &negz),
                        px + negx[i],
                        py + negy[i],
                        pz + negz[i],
                    );
                    vx[i] = gx;
                    vy[i] = gy;
                    vz[i] = gz;
                }
            }
        }

        // Per-voxel coordinate-descent refinement. `smallest_error` persists
        // across voxels (reset only when the initial mapped point is in-buffer),
        // matching ITK's out-of-loop declaration.
        let mut smallest_error = 0.0f64;
        let u = (uxd.as_slice(), uyd.as_slice(), uzd.as_slice());
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = (z * ny + y) * nx + x;
                    let (ax, ay, az) = phys(z, y, x); // original point
                                                      // mapped point p = orig + v.
                    let mut m = [ax + vx[i], ay + vy[i], az + vz[i]];
                    let mut new_p = m;
                    let mut step = step0;
                    let mut still_same = false;

                    let err_at = |m: [f64; 3]| -> Option<f64> {
                        let (cz, cy, cx) = idx(m[0], m[1], m[2]);
                        if cx < -0.5
                            || cx > nx as f64 - 0.5
                            || cy < -0.5
                            || cy > ny as f64 - 0.5
                            || cz < -0.5
                            || cz > nz as f64 - 0.5
                        {
                            return None;
                        }
                        let (fxv, fyv, fzv) = eval(u, m[0], m[1], m[2]);
                        Some(
                            ((m[0] + fxv - ax).powi(2)
                                + (m[1] + fyv - ay).powi(2)
                                + (m[2] + fzv - az).powi(2))
                            .sqrt(),
                        )
                    };

                    if let Some(e) = err_at(m) {
                        smallest_error = e;
                    }

                    for _ in 0..self.number_of_iterations {
                        if still_same {
                            step /= 2.0;
                        }
                        for k in 0..3 {
                            m[k] += step;
                            if let Some(t) = err_at(m) {
                                if t < smallest_error {
                                    smallest_error = t;
                                    new_p = m;
                                }
                            }
                            m[k] -= 2.0 * step;
                            if let Some(t) = err_at(m) {
                                if t < smallest_error {
                                    smallest_error = t;
                                    new_p = m;
                                }
                            }
                            m[k] += step;
                        }
                        still_same = true;
                        for j in 0..3 {
                            if new_p[j] != m[j] {
                                still_same = false;
                            }
                            m[j] = new_p[j];
                        }
                        if smallest_error < self.stop_value {
                            break;
                        }
                    }
                    vx[i] = m[0] - ax;
                    vy[i] = m[1] - ay;
                    vz[i] = m[2] - az;
                }
            }
        }

        let rx: Vec<f32> = vx.iter().map(|&v| v as f32).collect();
        let ry: Vec<f32> = vy.iter().map(|&v| v as f32).collect();
        let rz: Vec<f32> = vz.iter().map(|&v| v as f32).collect();
        Ok((
            rebuild(rx, dims, dx),
            rebuild(ry, dims, dy),
            rebuild(rz, dims, dz),
        ))
    }

    /// Coeus-native counterpart to the legacy application method.
    ///
    /// # Errors
    ///
    /// Returns an error when `dx`'s coordinate map is not Cartesian, when its
    /// direction matrix is singular, or when a device buffer cannot be read
    /// back or rebuilt.
    pub fn apply_native<B>(
        &self,
        dx: &ritk_image::Image<f32, B, 3>,
        dy: &ritk_image::Image<f32, B, 3>,
        dz: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<crate::NativeDisplacementField<B>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (ux, dims) = ritk_tensor_ops::native::extract_image_vec(dx)?;
        let (uy, _) = ritk_tensor_ops::native::extract_image_vec(dy)?;
        let (uz, _) = ritk_tensor_ops::native::extract_image_vec(dz)?;
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let geometry = CartesianGridGeometry::new(
            dx.origin(),
            dx.spacing(),
            dx.direction(),
            dx.coordinate_map(),
        )?;
        // ITK uses the x-axis spacing as the search step; spacing is indexed by
        // tensor axis, so the x axis is index 2.
        let step0 = dx.spacing()[2];

        let uxd: Vec<f64> = ux.iter().map(|&v| v as f64).collect();
        let uyd: Vec<f64> = uy.iter().map(|&v| v as f64).collect();
        let uzd: Vec<f64> = uz.iter().map(|&v| v as f64).collect();
        let negx: Vec<f64> = uxd.iter().map(|&v| -v).collect();
        let negy: Vec<f64> = uyd.iter().map(|&v| -v).collect();
        let negz: Vec<f64> = uzd.iter().map(|&v| -v).collect();

        // Physical point of a voxel and its continuous index components. Both
        // honour the direction cosines; see `CartesianGridGeometry`.
        let phys = |z: usize, y: usize, x: usize| {
            let p = geometry.point([z as f64, y as f64, x as f64]);
            (p[0], p[1], p[2])
        };
        let idx = |px: f64, py: f64, pz: f64| {
            let i = geometry.index([px, py, pz]);
            (i[0], i[1], i[2])
        };
        // u(point) → world (cx-component, cy, cz) = (interp ux, interp uy, interp uz).
        let eval = |ud: (&[f64], &[f64], &[f64]), px: f64, py: f64, pz: f64| {
            let (cz, cy, cx) = idx(px, py, pz);
            (
                interp_component(ud.0, dims, cz, cy, cx),
                interp_component(ud.1, dims, cz, cy, cx),
                interp_component(ud.2, dims, cz, cy, cx),
            )
        };

        // First guess: v₀(voxel) = neg warped by neg = interp(neg, x + neg(x)).
        let mut vx = vec![0.0f64; n];
        let mut vy = vec![0.0f64; n];
        let mut vz = vec![0.0f64; n];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = (z * ny + y) * nx + x;
                    let (px, py, pz) = phys(z, y, x);
                    let (gx, gy, gz) = eval(
                        (&negx, &negy, &negz),
                        px + negx[i],
                        py + negy[i],
                        pz + negz[i],
                    );
                    vx[i] = gx;
                    vy[i] = gy;
                    vz[i] = gz;
                }
            }
        }

        // Per-voxel coordinate-descent refinement. `smallest_error` persists
        // across voxels (reset only when the initial mapped point is in-buffer),
        // matching ITK's out-of-loop declaration.
        let mut smallest_error = 0.0f64;
        let u = (uxd.as_slice(), uyd.as_slice(), uzd.as_slice());
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = (z * ny + y) * nx + x;
                    let (ax, ay, az) = phys(z, y, x); // original point
                                                      // mapped point p = orig + v.
                    let mut m = [ax + vx[i], ay + vy[i], az + vz[i]];
                    let mut new_p = m;
                    let mut step = step0;
                    let mut still_same = false;

                    let err_at = |m: [f64; 3]| -> Option<f64> {
                        let (cz, cy, cx) = idx(m[0], m[1], m[2]);
                        if cx < -0.5
                            || cx > nx as f64 - 0.5
                            || cy < -0.5
                            || cy > ny as f64 - 0.5
                            || cz < -0.5
                            || cz > nz as f64 - 0.5
                        {
                            return None;
                        }
                        let (fxv, fyv, fzv) = eval(u, m[0], m[1], m[2]);
                        Some(
                            ((m[0] + fxv - ax).powi(2)
                                + (m[1] + fyv - ay).powi(2)
                                + (m[2] + fzv - az).powi(2))
                            .sqrt(),
                        )
                    };

                    if let Some(e) = err_at(m) {
                        smallest_error = e;
                    }

                    for _ in 0..self.number_of_iterations {
                        if still_same {
                            step /= 2.0;
                        }
                        for k in 0..3 {
                            m[k] += step;
                            if let Some(t) = err_at(m) {
                                if t < smallest_error {
                                    smallest_error = t;
                                    new_p = m;
                                }
                            }
                            m[k] -= 2.0 * step;
                            if let Some(t) = err_at(m) {
                                if t < smallest_error {
                                    smallest_error = t;
                                    new_p = m;
                                }
                            }
                            m[k] += step;
                        }
                        still_same = true;
                        for j in 0..3 {
                            if new_p[j] != m[j] {
                                still_same = false;
                            }
                            m[j] = new_p[j];
                        }
                        if smallest_error < self.stop_value {
                            break;
                        }
                    }
                    vx[i] = m[0] - ax;
                    vy[i] = m[1] - ay;
                    vz[i] = m[2] - az;
                }
            }
        }

        let rx: Vec<f32> = vx.iter().map(|&v| v as f32).collect();
        let ry: Vec<f32> = vy.iter().map(|&v| v as f32).collect();
        let rz: Vec<f32> = vz.iter().map(|&v| v as f32).collect();
        Ok(crate::NativeDisplacementField {
            x: crate::native_support::rebuild_image(rx, dims, dx, backend)?,
            y: crate::native_support::rebuild_image(ry, dims, dy, backend)?,
            z: crate::native_support::rebuild_image(rz, dims, dz, backend)?,
        })
    }
}

#[cfg(test)]
#[path = "tests_iterative_inverse_displacement.rs"]
mod tests_iterative_inverse_displacement;
