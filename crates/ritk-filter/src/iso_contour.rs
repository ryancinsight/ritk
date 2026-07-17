//! Iso-contour distance filter.
//!
//! # Mathematical Specification
//!
//! Ports `itk::IsoContourDistanceImageFilter`. It computes a narrow-band signed
//! distance to the `level_set_value` iso-contour of the input. The output is
//! initialized to `±far_value` (sign of `I − level`), and then for every pair of
//! adjacent voxels straddling the iso-surface a first-order distance estimate is
//! written to both, combined by **minimum absolute value** (so a voxel adjacent
//! to several crossings keeps the nearest):
//!
//! ```text
//! val0 = I(p) − level,   val1 = I(p+eₙ) − level    (sign change ⇒ crossing)
//! grad = (∇I(p)·½ + ∇I(p+eₙ)·½) / (2·spacing)       (central differences)
//! s    = |gradₙ|·spacingₙ / ‖grad‖ / (|val0 − val1|)
//! O(p)    ⊕= val0·s,   O(p+eₙ) ⊕= val1·s            (⊕ = keep smaller |·|)
//! ```
//!
//! The min-abs combine is order-independent, so the serial sweep is bitwise
//! identical to ITK's threaded result. Internal arithmetic is `f64` (ITK's
//! `PixelRealType` for a floating-point input). On a `z = 1` image the size-1
//! axis produces no crossings, reducing cleanly to the 2-D filter.
//!
//! # ITK parity
//!
//! Corresponds to `itk::IsoContourDistanceImageFilter` (`sitk.IsoContourDistance`,
//! default `level_set_value = 0`, `far_value = 10`), full (non-narrow-band) mode,
//! ZeroFluxNeumann boundary.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Iso-contour distance filter (`itk::IsoContourDistanceImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct IsoContourDistanceFilter {
    /// Iso-contour level. ITK default `0.0`.
    pub level_set_value: f64,
    /// Magnitude written to voxels away from the contour. ITK default `10.0`.
    pub far_value: f64,
}

impl Default for IsoContourDistanceFilter {
    fn default() -> Self {
        Self {
            level_set_value: 0.0,
            far_value: 10.0,
        }
    }
}

impl IsoContourDistanceFilter {
    /// Construct with explicit level and far value.
    pub fn new(level_set_value: f64, far_value: f64) -> Self {
        Self {
            level_set_value,
            far_value,
        }
    }

    /// Compute the narrow-band iso-contour signed distance.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let [nz, ny, nx] = dims;
        let sp = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        let level = self.level_set_value;
        let far = self.far_value;

        let cl = |i: isize, n: usize| -> usize { i.clamp(0, n as isize - 1) as usize };
        let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;
        // I(p) − level, ZeroFluxNeumann clamped.
        let gp = |z: isize, y: isize, x: isize| -> f64 {
            vals[idx(cl(z, nz), cl(y, ny), cl(x, nx))] as f64 - level
        };

        // Initialize: ±far away from the contour, 0 on it.
        let mut out: Vec<f32> = vals
            .iter()
            .map(|&v| {
                let d = v as f64 - level;
                if d > 0.0 {
                    far as f32
                } else if d < 0.0 {
                    -far as f32
                } else {
                    0.0
                }
            })
            .collect();

        let tiny = f64::MIN_POSITIVE;
        // Forward neighbour offset per axis n (0 = z, 1 = y, 2 = x).
        let e = [[1isize, 0, 0], [0, 1, 0], [0, 0, 1]];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (zi, yi, xi) = (z as isize, y as isize, x as isize);
                    let val0 = gp(zi, yi, xi);
                    let sign = val0 > 0.0;
                    // Central-difference gradient at p.
                    let grad0 = [
                        gp(zi + 1, yi, xi) - gp(zi - 1, yi, xi),
                        gp(zi, yi + 1, xi) - gp(zi, yi - 1, xi),
                        gp(zi, yi, xi + 1) - gp(zi, yi, xi - 1),
                    ];
                    for n in 0..3 {
                        let (nz_, ny_, nx_) = (zi + e[n][0], yi + e[n][1], xi + e[n][2]);
                        let val1 = gp(nz_, ny_, nx_);
                        if sign == (val1 > 0.0) {
                            continue;
                        }
                        // Central-difference gradient at the neighbour.
                        let grad1 = [
                            gp(nz_ + 1, ny_, nx_) - gp(nz_ - 1, ny_, nx_),
                            gp(nz_, ny_ + 1, nx_) - gp(nz_, ny_ - 1, nx_),
                            gp(nz_, ny_, nx_ + 1) - gp(nz_, ny_, nx_ - 1),
                        ];
                        let diff = if sign { val0 - val1 } else { val1 - val0 };
                        if diff < tiny {
                            continue;
                        }
                        let mut grad = [0.0f64; 3];
                        let mut norm = 0.0f64;
                        for a in 0..3 {
                            grad[a] = (grad0[a] * 0.5 + grad1[a] * 0.5) / (2.0 * sp[a]);
                            norm += grad[a] * grad[a];
                        }
                        norm = norm.sqrt();
                        if norm <= tiny {
                            continue;
                        }
                        let val = grad[n].abs() * sp[n] / norm / diff;
                        let new0 = val0 * val;
                        let new1 = val1 * val;
                        let ip = idx(z, y, x);
                        let iq = idx(cl(nz_, nz), cl(ny_, ny), cl(nx_, nx));
                        if new0.abs() < (out[ip] as f64).abs() {
                            out[ip] = new0 as f32;
                        }
                        if new1.abs() < (out[iq] as f64).abs() {
                            out[iq] = new1 as f32;
                        }
                    }
                }
            }
        }

        rebuild(out, dims, image)
    }    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(&self, image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B::default()) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let [nz, ny, nx] = dims;
        let sp = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        let level = self.level_set_value;
        let far = self.far_value;

        let cl = |i: isize, n: usize| -> usize { i.clamp(0, n as isize - 1) as usize };
        let idx = |z: usize, y: usize, x: usize| (z * ny + y) * nx + x;
        // I(p) − level, ZeroFluxNeumann clamped.
        let gp = |z: isize, y: isize, x: isize| -> f64 {
            vals[idx(cl(z, nz), cl(y, ny), cl(x, nx))] as f64 - level
        };

        // Initialize: ±far away from the contour, 0 on it.
        let mut out: Vec<f32> = vals
            .iter()
            .map(|&v| {
                let d = v as f64 - level;
                if d > 0.0 {
                    far as f32
                } else if d < 0.0 {
                    -far as f32
                } else {
                    0.0
                }
            })
            .collect();

        let tiny = f64::MIN_POSITIVE;
        // Forward neighbour offset per axis n (0 = z, 1 = y, 2 = x).
        let e = [[1isize, 0, 0], [0, 1, 0], [0, 0, 1]];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let (zi, yi, xi) = (z as isize, y as isize, x as isize);
                    let val0 = gp(zi, yi, xi);
                    let sign = val0 > 0.0;
                    // Central-difference gradient at p.
                    let grad0 = [
                        gp(zi + 1, yi, xi) - gp(zi - 1, yi, xi),
                        gp(zi, yi + 1, xi) - gp(zi, yi - 1, xi),
                        gp(zi, yi, xi + 1) - gp(zi, yi, xi - 1),
                    ];
                    for n in 0..3 {
                        let (nz_, ny_, nx_) = (zi + e[n][0], yi + e[n][1], xi + e[n][2]);
                        let val1 = gp(nz_, ny_, nx_);
                        if sign == (val1 > 0.0) {
                            continue;
                        }
                        // Central-difference gradient at the neighbour.
                        let grad1 = [
                            gp(nz_ + 1, ny_, nx_) - gp(nz_ - 1, ny_, nx_),
                            gp(nz_, ny_ + 1, nx_) - gp(nz_, ny_ - 1, nx_),
                            gp(nz_, ny_, nx_ + 1) - gp(nz_, ny_, nx_ - 1),
                        ];
                        let diff = if sign { val0 - val1 } else { val1 - val0 };
                        if diff < tiny {
                            continue;
                        }
                        let mut grad = [0.0f64; 3];
                        let mut norm = 0.0f64;
                        for a in 0..3 {
                            grad[a] = (grad0[a] * 0.5 + grad1[a] * 0.5) / (2.0 * sp[a]);
                            norm += grad[a] * grad[a];
                        }
                        norm = norm.sqrt();
                        if norm <= tiny {
                            continue;
                        }
                        let val = grad[n].abs() * sp[n] / norm / diff;
                        let new0 = val0 * val;
                        let new1 = val1 * val;
                        let ip = idx(z, y, x);
                        let iq = idx(cl(nz_, nz), cl(ny_, ny), cl(nx_, nx));
                        if new0.abs() < (out[ip] as f64).abs() {
                            out[ip] = new0 as f32;
                        }
                        if new1.abs() < (out[iq] as f64).abs() {
                            out[iq] = new1 as f32;
                        }
                    }
                }
            }
        }

        crate::native_support::rebuild_image(out, dims, image, backend)
    
    }

}

#[cfg(test)]
#[path = "tests_iso_contour.rs"]
mod tests_iso_contour;
