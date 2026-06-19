use super::*;
use ritk_image::test_support as ts;

type B = burn_ndarray::NdArray<f32>;

/// For a linear ramp `f = x` (fastest-varying tensor axis), the central
/// difference is exactly 1 along x in the interior and 0 along y, z. Component
/// order is sitk `(∂/∂x, ∂/∂y, ∂/∂z)`.
#[test]
fn gradient_of_x_ramp_is_unit_along_x() {
    let (nz, ny, nx) = (3usize, 3, 4);
    let mut vals = vec![0.0_f32; nz * ny * nx];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                vals[z * ny * nx + y * nx + x] = x as f32;
            }
        }
    }
    let img = ts::make_image::<B, 3>(vals, [nz, ny, nx]);
    let grad = GradientImageFilter::new(true).apply(&img).unwrap();
    let comps = grad.into_component_buffers();
    assert_eq!(comps.len(), 3);

    // Interior x-index (1..nx-1): ∂/∂x = (f[x+1]-f[x-1])/2 = 1.
    for z in 0..nz {
        for y in 0..ny {
            for x in 1..nx - 1 {
                let i = z * ny * nx + y * nx + x;
                assert!(
                    (comps[0][i] - 1.0).abs() < 1e-6,
                    "dx interior: {}",
                    comps[0][i]
                );
            }
        }
    }
    // ∂/∂y and ∂/∂z are identically zero (ramp is constant along y, z).
    for (&dy, &dz) in comps[1].iter().zip(comps[2].iter()) {
        assert!(dy.abs() < 1e-6, "dy: {dy}");
        assert!(dz.abs() < 1e-6, "dz: {dz}");
    }
}

/// For a linear ramp `f = x`, the Gaussian-smoothed gradient is ≈1 along x in the
/// interior (a constant-slope ramp is invariant under smoothing) and ≈0 along
/// y, z. Component order is sitk `(∂/∂x, ∂/∂y, ∂/∂z)`.
#[test]
fn gradient_recursive_gaussian_of_x_ramp_is_unit_along_x() {
    let (nz, ny, nx) = (6usize, 6, 12);
    let mut vals = vec![0.0_f32; nz * ny * nx];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                vals[z * ny * nx + y * nx + x] = x as f32;
            }
        }
    }
    let img = ts::make_image::<B, 3>(vals, [nz, ny, nx]);
    let grad = GradientRecursiveGaussianImageFilter::new(1.0)
        .apply(&img)
        .unwrap();
    let comps = grad.into_component_buffers();
    // Interior (away from the IIR boundary transient): ∂/∂x ≈ 1, ∂/∂y, ∂/∂z ≈ 0.
    // The ~2e-3 residual on dx is the Deriche order-1 unit-DC-gain approximation
    // on a finite ramp (shared with sitk; the cmake test is the float-exact oracle).
    for z in 1..nz - 1 {
        for y in 1..ny - 1 {
            for x in 4..nx - 4 {
                let i = z * ny * nx + y * nx + x;
                assert!((comps[0][i] - 1.0).abs() < 5e-3, "dx: {}", comps[0][i]);
                assert!(comps[1][i].abs() < 5e-3, "dy: {}", comps[1][i]);
                assert!(comps[2][i].abs() < 5e-3, "dz: {}", comps[2][i]);
            }
        }
    }
}
