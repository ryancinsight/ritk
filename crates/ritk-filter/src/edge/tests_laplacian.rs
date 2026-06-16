use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
    ts::make_image_with_spacing::<B, 3>(vals, dims, spacing)
}

/// Uniform image → Laplacian = 0 everywhere.
#[test]
fn test_uniform_zero_laplacian() {
    let dims = [8, 8, 8];
    let vals = vec![7.0_f32; 8 * 8 * 8];
    let img = make_image(vals, dims, [1.0, 1.0, 1.0]);
    let filter = LaplacianFilter::unit();
    let lap = filter.apply(&img).unwrap();

    let (out, _) = extract_vec_infallible(&lap);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.abs() < 1e-4,
            "Laplacian[{i}] = {v} expected 0 for uniform image"
        );
    }
}

/// I[z,y,x] = x² (unit spacing) → ∂²I/∂x² = 2, other second derivatives = 0
/// → Laplacian = 2 at interior voxels.
#[test]
fn test_quadratic_x_laplacian() {
    let [nz, ny, nx] = [6usize, 6, 10];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let ix = (flat % nx) as f32;
            ix * ix
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let filter = LaplacianFilter::unit();
    let lap = filter.apply(&img).unwrap();

    let (out, _) = extract_vec_infallible(&lap);

    // Check interior voxels only (exclude boundary rows/columns).
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (out[flat] - 2.0).abs() < 1e-4,
                    "Laplacian[{iz},{iy},{ix}] = {} expected 2.0",
                    out[flat]
                );
            }
        }
    }
}

/// I = x² with spacing_x = 2.0 → ∂²I/∂x² = 2/4 = 0.5
#[test]
fn test_non_unit_spacing() {
    let [nz, ny, nx] = [4usize, 4, 8];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let ix = (flat % nx) as f32;
            ix * ix
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 2.0]);
    let filter = LaplacianFilter::new([1.0, 1.0, 2.0].into());
    let lap = filter.apply(&img).unwrap();

    let (out, _) = extract_vec_infallible(&lap);

    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                // d²(x²)/dx² = 2; divided by spacing² = 4 → 0.5
                assert!(
                    (out[flat] - 0.5).abs() < 1e-4,
                    "Laplacian[{iz},{iy},{ix}] = {} expected 0.5",
                    out[flat]
                );
            }
        }
    }
}

/// I = x² + y² + z² (unit spacing) → Laplacian = 6 at interior.
#[test]
fn test_isotropic_quadratic() {
    let [nz, ny, nx] = [8usize, 8, 8];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let ix = (flat % nx) as f32;
            let iy = ((flat / nx) % ny) as f32;
            let iz = (flat / (ny * nx)) as f32;
            ix * ix + iy * iy + iz * iz
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let filter = LaplacianFilter::unit();
    let lap = filter.apply(&img).unwrap();

    let (out, _) = extract_vec_infallible(&lap);

    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (out[flat] - 6.0).abs() < 1e-3,
                    "Laplacian[{iz},{iy},{ix}] = {} expected 6.0",
                    out[flat]
                );
            }
        }
    }
}

/// Linear field I = x + y + z → interior second derivatives = 0 → Laplacian = 0.
///
/// The boundary is intentionally excluded: under the ZeroFluxNeumann boundary
/// condition (ITK's convention, which this filter matches) a min-face voxel
/// evaluates `(I[1] − 2·I[0] + I[0])/h² = (I[1] − I[0])/h² = slope/h²`, which
/// is nonzero for a non-constant linear field. Asserting zero there would be
/// analytically incorrect for the boundary condition under test.
#[test]
fn test_linear_field_zero_laplacian_interior() {
    let [nz, ny, nx] = [6usize, 6, 6];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let ix = (flat % nx) as f32;
            let iy = ((flat / nx) % ny) as f32;
            let iz = (flat / (ny * nx)) as f32;
            ix + iy + iz
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let filter = LaplacianFilter::unit();
    let lap = filter.apply(&img).unwrap();

    let (out, _) = extract_vec_infallible(&lap);
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let v = out[iz * ny * nx + iy * nx + ix];
                assert!(
                    v.abs() < 1e-4,
                    "interior Laplacian[{iz},{iy},{ix}] = {v} expected 0 for linear field"
                );
            }
        }
    }
}
