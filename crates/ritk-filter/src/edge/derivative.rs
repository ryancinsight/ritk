//! Directional derivative filter (central differences).
//!
//! Matches ITK `DerivativeImageFilter` / `sitk.Derivative`: applies the
//! `order`-th central-difference operator along one axis. With
//! `use_image_spacing = true` the result is divided by `spacing` **once**
//! (ITK's `DerivativeOperator` scales its coefficients by `1/spacing`
//! regardless of order — verified against sitk). Boundary handling is zero-flux
//! Neumann (edge-clamp), matching ITK's `NeighborhoodOperatorImageFilter`.
//!
//! Supported orders (the ITK `DerivativeOperator` coefficients):
//! - order 1: `[−½, 0, ½]`  →  `(f[i+1] − f[i−1]) / 2`
//! - order 2: `[1, −2, 1]`  →  `f[i+1] − 2f[i] + f[i−1]`

use anyhow::{bail, Result};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Directional derivative filter.
#[derive(Debug, Clone, Copy)]
pub struct DerivativeImageFilter {
    /// Axis in ritk tensor order: 0 = Z, 1 = Y, 2 = X.
    pub axis: usize,
    /// Derivative order (1 or 2).
    pub order: usize,
    /// Divide by `spacing^order` for a physical-unit derivative.
    pub use_image_spacing: bool,
}

impl DerivativeImageFilter {
    /// Construct a derivative filter.
    pub fn new(axis: usize, order: usize, use_image_spacing: bool) -> Self {
        Self {
            axis,
            order,
            use_image_spacing,
        }
    }

    /// Apply the derivative along the configured axis.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        if self.axis > 2 {
            bail!("derivative: axis must be 0, 1, or 2; got {}", self.axis);
        }
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;
        let spacing = image.spacing().to_array(); // [z, y, x]
        let h = if self.use_image_spacing {
            spacing[self.axis]
        } else {
            1.0
        };

        // Stride and length along the chosen axis.
        let (len, stride) = match self.axis {
            0 => (nz, ny * nx),
            1 => (ny, nx),
            _ => (nx, 1),
        };

        // Edge-clamped neighbour along the axis: returns f at the line position
        // `p` offset by `d`, clamped to [0, len-1].
        let at = |base: usize, p: usize, d: isize| -> f64 {
            let q = (p as isize + d).clamp(0, len as isize - 1) as usize;
            vals[base + q * stride] as f64
        };

        // ITK `DerivativeOperator` scales its coefficients by 1/spacing ONCE,
        // regardless of derivative order (verified against `sitk.Derivative`:
        // order-2 with spacing 2 divides by 2, not 4).
        let scale = match self.order {
            1 => 0.5 / h,
            2 => 1.0 / h,
            o => bail!("derivative: only order 1 and 2 are supported; got {o}"),
        };

        let mut out = vec![0.0_f32; vals.len()];
        // Iterate over every line parallel to the axis.
        let (n_outer, n_mid) = match self.axis {
            0 => (ny, nx),
            1 => (nz, nx),
            _ => (nz, ny),
        };
        for a in 0..n_outer {
            for b in 0..n_mid {
                // Base index of this line at axis position 0.
                let base = match self.axis {
                    0 => a * nx + b,           // (y=a, x=b), z varies
                    1 => a * ny * nx + b,      // (z=a, x=b), y varies
                    _ => a * ny * nx + b * nx, // (z=a, y=b), x varies
                };
                for p in 0..len {
                    let v = match self.order {
                        1 => at(base, p, 1) - at(base, p, -1),
                        _ => at(base, p, 1) - 2.0 * at(base, p, 0) + at(base, p, -1),
                    };
                    out[base + p * stride] = (v * scale) as f32;
                }
            }
        }
        Ok(rebuild(out, dims, image))
    }
    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        if self.axis > 2 {
            bail!("derivative: axis must be 0, 1, or 2; got {}", self.axis);
        }
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let [nz, ny, nx] = dims;
        let spacing = image.spacing().to_array(); // [z, y, x]
        let h = if self.use_image_spacing {
            spacing[self.axis]
        } else {
            1.0
        };

        // Stride and length along the chosen axis.
        let (len, stride) = match self.axis {
            0 => (nz, ny * nx),
            1 => (ny, nx),
            _ => (nx, 1),
        };

        // Edge-clamped neighbour along the axis: returns f at the line position
        // `p` offset by `d`, clamped to [0, len-1].
        let at = |base: usize, p: usize, d: isize| -> f64 {
            let q = (p as isize + d).clamp(0, len as isize - 1) as usize;
            vals[base + q * stride] as f64
        };

        // ITK `DerivativeOperator` scales its coefficients by 1/spacing ONCE,
        // regardless of derivative order (verified against `sitk.Derivative`:
        // order-2 with spacing 2 divides by 2, not 4).
        let scale = match self.order {
            1 => 0.5 / h,
            2 => 1.0 / h,
            o => bail!("derivative: only order 1 and 2 are supported; got {o}"),
        };

        let mut out = vec![0.0_f32; vals.len()];
        // Iterate over every line parallel to the axis.
        let (n_outer, n_mid) = match self.axis {
            0 => (ny, nx),
            1 => (nz, nx),
            _ => (nz, ny),
        };
        for a in 0..n_outer {
            for b in 0..n_mid {
                // Base index of this line at axis position 0.
                let base = match self.axis {
                    0 => a * nx + b,           // (y=a, x=b), z varies
                    1 => a * ny * nx + b,      // (z=a, x=b), y varies
                    _ => a * ny * nx + b * nx, // (z=a, y=b), x varies
                };
                for p in 0..len {
                    let v = match self.order {
                        1 => at(base, p, 1) - at(base, p, -1),
                        _ => at(base, p, 1) - 2.0 * at(base, p, 0) + at(base, p, -1),
                    };
                    out[base + p * stride] = (v * scale) as f32;
                }
            }
        }
        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_derivative.rs"]
mod tests_derivative;
