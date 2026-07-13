//! ITK-exact Canny edge detection (`itk::CannyEdgeDetectionImageFilter`).
//!
//! # Mathematical Specification
//!
//! Ports `sitk.CannyEdgeDetection` / `itk::CannyEdgeDetectionImageFilter`. Unlike
//! the gradient-non-maximum-suppression Canny in [`super::canny`], this is ITK's
//! zero-crossing-of-the-second-directional-derivative formulation:
//!
//! 1. **Gaussian smooth** the input with [`DiscreteGaussianFilter`] (`variance`,
//!    `maximum_error`).
//! 2. **Second directional derivative** of the smoothed image `I`:
//!    `D = (Σ_i I_i²·I_ii + Σ_{i<j} 2·I_i·I_j·I_ij) / (Σ_i I_i² + α²)`, `α² = 1e-4`,
//!    with `I_i` the central first derivative, `I_ii` the second, `I_ij` the 0.25
//!    diagonal cross derivative (ZeroFluxNeumann boundary).
//! 3. **Gradient-maximum mask × magnitude**: `U = ⟦∂D/∂n ≤ 0⟧ · |∇I|`, where
//!    `n = ∇I / |∇I|` is the gradient direction and `|∇I| = sqrt(α² + Σ I_i²)`.
//! 4. **Zero crossing** of `D` (via [`ZeroCrossingImageFilter`]).
//! 5. **Multiply**: `M = U · ZeroCross(D)`.
//! 6. **Hysteresis threshold**: edge voxels are the connected weak set
//!    (`M > lower_threshold`) reachable from any strong voxel (`M > upper_threshold`),
//!    flooded over face connectivity. Output is `1.0` at edges, `0.0` elsewhere.
//!
//! Validated bit-exact against `sitk.CannyEdgeDetection` across square, circle and
//! noisy inputs over a range of `variance` / threshold settings.
//!
//! ## References
//! - Canny, J. (1986). "A computational approach to edge detection." *IEEE TPAMI*.
//! - ITK `itkCannyEdgeDetectionImageFilter.hxx`.

use crate::discrete_gaussian::{discrete_gaussian_smooth_flat, SpacingMode};
use crate::intensity::zero_crossing::zero_crossing_vec;
use crate::DiscreteGaussianFilter;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// `alpha * alpha` floor on the squared gradient magnitude (ITK `alpha = 0.01`).
const ALPHA_SQ: f32 = 0.0001;

/// ITK-exact Canny edge detector.
///
/// Returns a binary edge image (`1.0` at edges, `0.0` elsewhere), bit-exact to
/// `sitk.CannyEdgeDetection`.
///
/// # Defaults
/// - `variance = 0.0`, `maximum_error = 0.01`
/// - `lower_threshold = 0.0`, `upper_threshold = 0.0`
#[derive(Debug, Clone)]
pub struct CannyEdgeDetectionImageFilter {
    /// Gaussian smoothing variance (σ², physical units).
    pub variance: f64,
    /// Discrete-Gaussian truncation error.
    pub maximum_error: f64,
    /// Lower hysteresis threshold on the edge-strength image.
    pub lower_threshold: f32,
    /// Upper hysteresis threshold on the edge-strength image.
    pub upper_threshold: f32,
}

impl Default for CannyEdgeDetectionImageFilter {
    fn default() -> Self {
        Self {
            variance: 0.0,
            maximum_error: 0.01,
            lower_threshold: 0.0,
            upper_threshold: 0.0,
        }
    }
}

impl CannyEdgeDetectionImageFilter {
    /// Detect edges in a 3-D image (`nz == 1` ⇒ 2-D).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        // 1. Gaussian smoothing (DiscreteGaussian, ITK discrete kernel).
        let sm_img = DiscreteGaussianFilter::<B>::new_isotropic(self.variance)
            .with_maximum_error(self.maximum_error)
            .apply(image);
        let (sm, dims) = extract_vec_infallible(&sm_img);
        let out = canny_edge_detection_flat(&sm, dims, self.lower_threshold, self.upper_threshold);
        rebuild(out, dims, image)
    }

    /// Coeus-native sister of [`CannyEdgeDetectionImageFilter::apply`].
    ///
    /// Smooths natively via the burn-free [`discrete_gaussian_smooth_flat`] core
    /// (same ITK discrete-Gaussian kernel and replicate-boundary convolution the
    /// Burn `DiscreteGaussianFilter::apply` uses) and runs the identical
    /// second-directional-derivative / zero-crossing / hysteresis pipeline via
    /// the shared [`canny_edge_detection_flat`] host core, so the result is
    /// bitwise-identical to the Burn path. No Burn tensor is constructed.
    /// Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let spacing = *image.spacing();
        let variance = vec![self.variance];
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            let sm = discrete_gaussian_smooth_flat(
                vals.to_vec(),
                dims,
                &spacing,
                &variance,
                self.maximum_error,
                SpacingMode::Physical,
            );
            canny_edge_detection_flat(&sm, dims, self.lower_threshold, self.upper_threshold)
        })
    }
}

/// Substrate-agnostic host core: ITK zero-crossing Canny stages 2–6 on the
/// already-smoothed flat z-major buffer `sm` — second directional derivative,
/// gradient-maximum masking, zero crossing of `D`, product, and hysteresis
/// flood. Single source of truth for the Burn
/// [`apply`](CannyEdgeDetectionImageFilter::apply) and Coeus-native
/// [`apply_native`](CannyEdgeDetectionImageFilter::apply_native) paths.
fn canny_edge_detection_flat(
    sm: &[f32],
    dims: [usize; 3],
    lower_threshold: f32,
    upper_threshold: f32,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    if n == 0 {
        return vec![0.0f32; 0];
    }
    let idx = |z: usize, y: usize, x: usize| z * ny * nx + y * nx + x;
    // Clamped (ZeroFluxNeumann) accessor.
    let at = |buf: &[f32], z: isize, y: isize, x: isize| -> f32 {
        let zc = z.clamp(0, nz as isize - 1) as usize;
        let yc = y.clamp(0, ny as isize - 1) as usize;
        let xc = x.clamp(0, nx as isize - 1) as usize;
        buf[idx(zc, yc, xc)]
    };

    // 2. Second directional derivative D and stored gradient components.
    let mut deriv = vec![0.0f32; n];
    let mut grad = vec![[0.0f32; 3]; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let (zi, yi, xi) = (z as isize, y as isize, x as isize);
                let c = sm[idx(z, y, x)];
                // first derivatives (central), second derivatives, cross (0.25 diag).
                let fx = 0.5 * (at(sm, zi, yi, xi + 1) - at(sm, zi, yi, xi - 1));
                let fy = 0.5 * (at(sm, zi, yi + 1, xi) - at(sm, zi, yi - 1, xi));
                let fz = 0.5 * (at(sm, zi + 1, yi, xi) - at(sm, zi - 1, yi, xi));
                let fxx = at(sm, zi, yi, xi + 1) - 2.0 * c + at(sm, zi, yi, xi - 1);
                let fyy = at(sm, zi, yi + 1, xi) - 2.0 * c + at(sm, zi, yi - 1, xi);
                let fzz = at(sm, zi + 1, yi, xi) - 2.0 * c + at(sm, zi - 1, yi, xi);
                let fxy = 0.25
                    * (at(sm, zi, yi - 1, xi - 1)
                        - at(sm, zi, yi - 1, xi + 1)
                        - at(sm, zi, yi + 1, xi - 1)
                        + at(sm, zi, yi + 1, xi + 1));
                let fxz = 0.25
                    * (at(sm, zi - 1, yi, xi - 1)
                        - at(sm, zi - 1, yi, xi + 1)
                        - at(sm, zi + 1, yi, xi - 1)
                        + at(sm, zi + 1, yi, xi + 1));
                let fyz = 0.25
                    * (at(sm, zi - 1, yi - 1, xi)
                        - at(sm, zi - 1, yi + 1, xi)
                        - at(sm, zi + 1, yi - 1, xi)
                        + at(sm, zi + 1, yi + 1, xi));
                let num = fx * fx * fxx
                    + fy * fy * fyy
                    + fz * fz * fzz
                    + 2.0 * (fx * fy * fxy + fx * fz * fxz + fy * fz * fyz);
                let den = fx * fx + fy * fy + fz * fz + ALPHA_SQ;
                deriv[idx(z, y, x)] = num / den;
                grad[idx(z, y, x)] = [fx, fy, fz];
            }
        }
    }

    // 3. U = ⟦∂D/∂n ≤ 0⟧ · |∇I|  (gradient-maximum mask × magnitude).
    let mut upd = vec![0.0f32; n];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let (zi, yi, xi) = (z as isize, y as isize, x as isize);
                let [fx, fy, fz] = grad[idx(z, y, x)];
                let grad_mag = (ALPHA_SQ + fx * fx + fy * fy + fz * fz).sqrt();
                let d1x = 0.5 * (at(&deriv, zi, yi, xi + 1) - at(&deriv, zi, yi, xi - 1));
                let d1y = 0.5 * (at(&deriv, zi, yi + 1, xi) - at(&deriv, zi, yi - 1, xi));
                let d1z = 0.5 * (at(&deriv, zi + 1, yi, xi) - at(&deriv, zi - 1, yi, xi));
                let deriv_pos =
                    d1x * (fx / grad_mag) + d1y * (fy / grad_mag) + d1z * (fz / grad_mag);
                upd[idx(z, y, x)] = if deriv_pos <= 0.0 { grad_mag } else { 0.0 };
            }
        }
    }

    // 4. Zero crossing of D (ITK default fg = 1.0, bg = 0.0).
    let zc = zero_crossing_vec(&deriv, dims, 1.0, 0.0);

    // 5. Multiply: edge strength M = U · ZeroCross(D).
    let mult: Vec<f32> = upd.iter().zip(zc.iter()).map(|(&u, &z)| u * z).collect();

    // 6. Hysteresis: flood the weak set (M > lower) from strong seeds
    //    (M > upper) over face connectivity (4-conn in 2-D, 6-conn in 3-D).
    let mut out = vec![0.0f32; n];
    let mut stack: Vec<usize> = Vec::new();
    for (f, &m) in mult.iter().enumerate() {
        if m > upper_threshold {
            out[f] = 1.0;
            stack.push(f);
        }
    }
    while let Some(f) = stack.pop() {
        let z = f / (ny * nx);
        let r = f % (ny * nx);
        let (y, x) = (r / nx, r % nx);
        let (zi, yi, xi) = (z as isize, y as isize, x as isize);
        for (dz, dy, dx) in [
            (-1isize, 0isize, 0isize),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ] {
            let (z2, y2, x2) = (zi + dz, yi + dy, xi + dx);
            if z2 >= 0
                && y2 >= 0
                && x2 >= 0
                && z2 < nz as isize
                && y2 < ny as isize
                && x2 < nx as isize
            {
                let g = idx(z2 as usize, y2 as usize, x2 as usize);
                if out[g] == 0.0 && mult[g] > lower_threshold {
                    out[g] = 1.0;
                    stack.push(g);
                }
            }
        }
    }

    out
}

#[cfg(test)]
#[path = "tests_canny_edge_detection.rs"]
mod tests_canny_edge_detection;

#[cfg(test)]
mod tests_native {
    use super::CannyEdgeDetectionImageFilter;
    use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};
    use coeus_core::SequentialBackend;

    fn filter() -> CannyEdgeDetectionImageFilter {
        CannyEdgeDetectionImageFilter {
            variance: 1.0,
            maximum_error: 0.01,
            lower_threshold: 2.0,
            upper_threshold: 5.0,
        }
    }

    #[test]
    fn matches_burn() {
        // A single axial slice with a bright half-plane step so the ITK
        // zero-crossing edge lands on the boundary. Both paths share the
        // discrete-Gaussian smoothing and stage-2–6 cores → bitwise-identical.
        let (nz, ny, nx) = (1usize, 7usize, 7usize);
        let mut vals = vec![0.0f32; nz * ny * nx];
        for y in 0..ny {
            for x in 0..nx {
                if x >= nx / 2 {
                    vals[y * nx + x] = 200.0;
                }
            }
        }
        assert_native_matches_burn(
            vals,
            [nz, ny, nx],
            |img| filter().apply(img),
            |img, backend| filter().apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_constant_has_no_edges() {
        // DiscreteGaussian replicate-padding preserves a constant field exactly,
        // so all derivatives vanish, D ≡ 0, its zero-crossing set is empty, and
        // no voxel is an edge.
        let img = make_native_image(vec![42.0f32; 125], [5, 5, 5]);
        let out = filter()
            .apply_native(&img, &SequentialBackend)
            .expect("native canny edge detection");
        for &v in &native_vals(&out) {
            assert_eq!(v, 0.0, "constant field must yield no edges, got {v}");
        }
    }
}
