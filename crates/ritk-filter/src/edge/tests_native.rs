//! Differential + analytical coverage for the Coeus-native edge-filter paths.
//!
//! Each native wrapper shares the exact substrate-agnostic host core its Burn
//! counterpart calls, so the differential assertion is bitwise-exact (via the
//! shared `native_support::assert_native_matches_burn` harness — no epsilon).
//! The analytical oracles pin the mathematical contract directly on the native
//! path: Laplacian of a linear field = 0, gradient magnitude of a constant = 0,
//! LoG of a constant = 0.

use crate::edge::gaussian_sigma::GaussianSigma;
use crate::edge::{
    GradientMagnitudeFilter, LaplacianFilter, LaplacianOfGaussianFilter, SobelFilter,
};
use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};

/// Deterministic non-trivial ramp-plus-ripple buffer for differential checks.
fn ramp(dims: [usize; 3]) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    (0..n)
        .map(|i| (i as f32) * 0.5 - (i % 7) as f32 * 1.3 + 2.0)
        .collect()
}

// ── GradientMagnitude ──────────────────────────────────────────────────────────

mod gradient_magnitude {
    use super::*;

    #[test]
    fn matches_burn() {
        let dims = [5, 6, 4];
        assert_native_matches_burn(
            ramp(dims),
            dims,
            |img| GradientMagnitudeFilter::unit().apply(img).expect("burn"),
            |img, _b| GradientMagnitudeFilter::unit().apply_native(img),
        );
    }

    /// |∇c| = 0 for a constant field (every central difference cancels).
    #[test]
    fn constant_field_is_zero() {
        let dims = [4, 4, 4];
        let img = make_native_image(vec![3.5_f32; dims[0] * dims[1] * dims[2]], dims);
        let out = GradientMagnitudeFilter::unit().apply_native(&img).unwrap();
        for v in native_vals(&out) {
            assert_eq!(v, 0.0, "gradient magnitude of a constant must be 0");
        }
    }
}

// ── Laplacian ───────────────────────────────────────────────────────────────────

mod laplacian {
    use super::*;

    #[test]
    fn matches_burn() {
        let dims = [5, 4, 6];
        assert_native_matches_burn(
            ramp(dims),
            dims,
            |img| LaplacianFilter::unit().apply(img).expect("burn"),
            |img, _b| LaplacianFilter::unit().apply_native(img),
        );
    }

    /// ∇²(a·z + b·y + c·x + d) = 0 in the interior (second differences of a
    /// linear field vanish). Boundary voxels are excluded: ZeroFluxNeumann
    /// clamps the out-of-range neighbour, yielding the one-sided slope there,
    /// not zero — matching ITK `LaplacianImageFilter`.
    #[test]
    fn linear_field_interior_is_zero() {
        let dims = [5, 5, 5];
        let [nz, ny, nx] = dims;
        let mut vals = vec![0.0_f32; nz * ny * nx];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    vals[iz * ny * nx + iy * nx + ix] =
                        2.0 * iz as f32 - 3.0 * iy as f32 + 1.5 * ix as f32 + 4.0;
                }
            }
        }
        let img = make_native_image(vals, dims);
        let out = LaplacianFilter::unit().apply_native(&img).unwrap();
        let result = native_vals(&out);
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let v = result[iz * ny * nx + iy * nx + ix];
                    assert!(
                        v.abs() < 1e-4,
                        "interior Laplacian of a linear field must be ~0, got {v} at ({iz},{iy},{ix})"
                    );
                }
            }
        }
    }
}

// ── Sobel ────────────────────────────────────────────────────────────────────────

mod sobel {
    use super::*;

    #[test]
    fn matches_burn() {
        let dims = [4, 5, 5];
        assert_native_matches_burn(
            ramp(dims),
            dims,
            |img| SobelFilter::unit().apply(img).expect("burn"),
            |img, _b| SobelFilter::unit().apply_native(img),
        );
    }

    /// Sobel gradient magnitude of a constant field is 0 (derivative kernel
    /// `[-1,0,1]` sums to zero over a constant neighbourhood).
    #[test]
    fn constant_field_is_zero() {
        let dims = [4, 4, 4];
        let img = make_native_image(vec![-7.0_f32; dims[0] * dims[1] * dims[2]], dims);
        let out = SobelFilter::unit().apply_native(&img).unwrap();
        for v in native_vals(&out) {
            assert_eq!(v, 0.0, "Sobel magnitude of a constant must be 0");
        }
    }
}

// ── Laplacian of Gaussian ────────────────────────────────────────────────────────

mod log {
    use super::*;

    fn filter() -> LaplacianOfGaussianFilter {
        LaplacianOfGaussianFilter::new(GaussianSigma::new_unchecked(1.5))
    }

    #[test]
    fn matches_burn() {
        let dims = [6, 6, 6];
        assert_native_matches_burn(
            ramp(dims),
            dims,
            |img| filter().apply(img).expect("burn"),
            |img, _b| filter().apply_native(img),
        );
    }

    /// ∇²(G_σ * c) = 0 for a constant field.
    #[test]
    fn constant_field_is_zero() {
        let dims = [8, 8, 8];
        let img = make_native_image(vec![5.0_f32; dims[0] * dims[1] * dims[2]], dims);
        let out = filter().apply_native(&img).unwrap();
        for v in native_vals(&out) {
            assert!(v.abs() < 1e-4, "LoG of a constant must be ~0, got {v}");
        }
    }
}
