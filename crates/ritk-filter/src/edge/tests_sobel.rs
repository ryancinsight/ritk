//! Tests for sobel
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

/// Construct a test image from flat data, dimensions, and spacing.
fn make_image(vals: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
    ts::make_image_with_spacing::<B, 3>(vals, dims, spacing)
}

/// Uniform image → gradient magnitude = 0.0 everywhere.
///
/// Proof: derivative of a constant is identically zero. The derivative
/// kernel [-1, 0, 1] applied to [c, c, c] yields 0 regardless of padding
/// strategy. All subsequent smoothing passes preserve zero.
#[test]
fn test_uniform_image_zero_gradient() {
    let dims = [8, 8, 8];
    let vals = vec![42.0_f32; 8 * 8 * 8];
    let img = make_image(vals, dims, [1.0, 1.0, 1.0]);
    let filter = SobelFilter::unit();
    let mag = filter.apply(&img).unwrap();

    let (out, _) = extract_vec_infallible(&mag);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "magnitude[{i}] = {v}, expected 0.0 for uniform image"
        );
    }
}

/// Linear ramp I(z,y,x) = x with unit spacing.
///
/// Expected at interior voxels:
///   gx = 1.0, gy = 0.0, gz = 0.0, magnitude = 1.0
///
/// Derivation:
///   Derivative along x: (x+1)−(x−1) = 2
///   Smooth y: [1,2,1]·[2,2,2] = 8
///   Smooth z: [1,2,1]·[8,8,8] = 32
///   Normalized: 32 / (32·1) = 1.0
///
///   Derivative along y of I=x: I(y+1,x)−I(y−1,x) = x−x = 0 → gy = 0
///   Derivative along z of I=x: analogous → gz = 0
#[test]
fn test_ramp_x_unit_spacing() {
    let [nz, ny, nx] = [8usize, 8, 12];
    let vals: Vec<f32> = (0..nz * ny * nx).map(|flat| (flat % nx) as f32).collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let filter = SobelFilter::unit();
    let (gz, gy, gx) = filter.apply_components(&img).unwrap();

    let (gz_vals, _) = extract_vec_infallible(&gz);
    let (gy_vals, _) = extract_vec_infallible(&gy);
    let (gx_vals, _) = extract_vec_infallible(&gx);

    // Check interior voxels only (1-voxel margin from each face).
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    gz_vals[flat].abs() < 1e-5,
                    "gz[{iz},{iy},{ix}] = {}, expected 0.0",
                    gz_vals[flat]
                );
                assert!(
                    gy_vals[flat].abs() < 1e-5,
                    "gy[{iz},{iy},{ix}] = {}, expected 0.0",
                    gy_vals[flat]
                );
                assert!(
                    (gx_vals[flat] - 1.0).abs() < 1e-5,
                    "gx[{iz},{iy},{ix}] = {}, expected 1.0",
                    gx_vals[flat]
                );
            }
        }
    }

    // Verify magnitude at interior.
    let mag = filter.apply(&img).unwrap();
    let (mag_vals, _) = extract_vec_infallible(&mag);
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (mag_vals[flat] - 1.0).abs() < 1e-5,
                    "magnitude[{iz},{iy},{ix}] = {}, expected 1.0",
                    mag_vals[flat]
                );
            }
        }
    }
}

/// Diagonal ramp I(z,y,x) = x + y + z with unit spacing.
///
/// Expected at interior: each component = 1.0, magnitude = √3.
///
/// Derivation (each axis identical by symmetry):
///   Derivative along a: (c_a+1)−(c_a−1) = 2
///   Two smoothing passes on constant 2 → 32
///   Normalized: 32 / (32·1) = 1.0
///   |∇I| = √(1²+1²+1²) = √3 ≈ 1.7320508
#[test]
fn test_diagonal_ramp_magnitude() {
    let [nz, ny, nx] = [10usize, 10, 10];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let ix = flat % nx;
            let iy = (flat / nx) % ny;
            let iz = flat / (ny * nx);
            (iz + iy + ix) as f32
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let filter = SobelFilter::unit();
    let (gz, gy, gx) = filter.apply_components(&img).unwrap();

    let (gz_vals, _) = extract_vec_infallible(&gz);
    let (gy_vals, _) = extract_vec_infallible(&gy);
    let (gx_vals, _) = extract_vec_infallible(&gx);
    let expected_mag = 3.0_f32.sqrt();
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (gz_vals[flat] - 1.0).abs() < 1e-5,
                    "gz[{iz},{iy},{ix}] = {}, expected 1.0",
                    gz_vals[flat]
                );
                assert!(
                    (gy_vals[flat] - 1.0).abs() < 1e-5,
                    "gy[{iz},{iy},{ix}] = {}, expected 1.0",
                    gy_vals[flat]
                );
                assert!(
                    (gx_vals[flat] - 1.0).abs() < 1e-5,
                    "gx[{iz},{iy},{ix}] = {}, expected 1.0",
                    gx_vals[flat]
                );
            }
        }
    }
    let mag = filter.apply(&img).unwrap();
    let (mag_vals, _) = extract_vec_infallible(&mag);
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (mag_vals[flat] - expected_mag).abs() < 1e-4,
                    "magnitude[{iz},{iy},{ix}] = {}, expected {expected_mag}",
                    mag_vals[flat]
                );
            }
        }
    }
}

/// Non-unit spacing: I(z,y,x) = x, spacing\_x = 2.0.
///
/// Expected at interior: gx = 1.0 / 2.0 = 0.5
///
/// Derivation:
///   Raw convolution is identical to unit-spacing case (index-space): 32
///   Normalized: 32 / (32 · 2.0) = 0.5
#[test]
fn test_non_unit_spacing() {
    let [nz, ny, nx] = [6usize, 6, 10];
    let vals: Vec<f32> = (0..nz * ny * nx).map(|flat| (flat % nx) as f32).collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 2.0]);
    let filter = SobelFilter::new([1.0, 1.0, 2.0].into());
    let (gz, gy, gx) = filter.apply_components(&img).unwrap();

    let (gz_vals, _) = extract_vec_infallible(&gz);
    let (gy_vals, _) = extract_vec_infallible(&gy);
    let (gx_vals, _) = extract_vec_infallible(&gx);
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    gz_vals[flat].abs() < 1e-5,
                    "gz[{iz},{iy},{ix}] = {}, expected 0.0",
                    gz_vals[flat]
                );
                assert!(
                    gy_vals[flat].abs() < 1e-5,
                    "gy[{iz},{iy},{ix}] = {}, expected 0.0",
                    gy_vals[flat]
                );
                assert!(
                    (gx_vals[flat] - 0.5).abs() < 1e-5,
                    "gx[{iz},{iy},{ix}] = {}, expected 0.5",
                    gx_vals[flat]
                );
            }
        }
    }
    // Verify magnitude = 0.5 at interior.
    let mag = filter.apply(&img).unwrap();
    let (mag_vals, _) = extract_vec_infallible(&mag);
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (mag_vals[flat] - 0.5).abs() < 1e-5,
                    "magnitude[{iz},{iy},{ix}] = {}, expected 0.5",
                    mag_vals[flat]
                );
            }
        }
    }
}

/// Metadata preservation: origin, spacing, and direction pass through
/// unmodified to all output images.
#[test]
fn test_metadata_preserved() {
    let dims = [4, 4, 4];
    let vals = vec![1.0_f32; 4 * 4 * 4];

    let origin = Point::new([10.0, -5.0, 3.5]);
    let spacing_val = Spacing::new([0.5, 1.5, 2.5]);
    let mut direction = Direction::identity();
    // 90-degree rotation around the z-axis (proper rotation, det = 1).
    direction[(0, 0)] = 0.0;
    direction[(0, 1)] = -1.0;
    direction[(1, 0)] = 1.0;
    direction[(1, 1)] = 0.0;

    let device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    let img = Image::new(tensor, origin, spacing_val, direction);

    let filter = SobelFilter::new([0.5, 1.5, 2.5].into());

    // Check apply (magnitude).
    let mag = filter.apply(&img).unwrap();
    assert_eq!(mag.origin(), &origin);
    assert_eq!(mag.spacing(), &spacing_val);
    assert_eq!(mag.direction(), &direction);
    assert_eq!(mag.shape(), dims);

    // Check apply_components.
    let (gz, gy, gx) = filter.apply_components(&img).unwrap();
    for (label, component) in [("gz", &gz), ("gy", &gy), ("gx", &gx)] {
        assert_eq!(component.origin(), &origin, "{label}: origin mismatch");
        assert_eq!(
            component.spacing(),
            &spacing_val,
            "{label}: spacing mismatch"
        );
        assert_eq!(
            component.direction(),
            &direction,
            "{label}: direction mismatch"
        );
        assert_eq!(component.shape(), dims, "{label}: shape mismatch");
    }
}

/// Y-axis ramp: I(z,y,x) = y with unit spacing.
///
/// Verifies axis separation: gy = 1.0, gx = gz = 0 at interior.
/// This confirms that the derivative and smoothing kernels are applied
/// along the correct axes.
#[test]
fn test_ramp_y_axis_separation() {
    let [nz, ny, nx] = [8usize, 10, 8];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let iy = (flat / nx) % ny;
            iy as f32
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let filter = SobelFilter::unit();
    let (gz, gy, gx) = filter.apply_components(&img).unwrap();

    let (gz_vals, _) = extract_vec_infallible(&gz);
    let (gy_vals, _) = extract_vec_infallible(&gy);
    let (gx_vals, _) = extract_vec_infallible(&gx);
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    gz_vals[flat].abs() < 1e-5,
                    "gz[{iz},{iy},{ix}] = {}, expected 0.0",
                    gz_vals[flat]
                );
                assert!(
                    (gy_vals[flat] - 1.0).abs() < 1e-5,
                    "gy[{iz},{iy},{ix}] = {}, expected 1.0",
                    gy_vals[flat]
                );
                assert!(
                    gx_vals[flat].abs() < 1e-5,
                    "gx[{iz},{iy},{ix}] = {}, expected 0.0",
                    gx_vals[flat]
                );
            }
        }
    }
}

/// Z-axis ramp: I(z,y,x) = z with unit spacing.
///
/// Verifies gz = 1.0, gy = gx = 0 at interior.
#[test]
fn test_ramp_z_axis_separation() {
    let [nz, ny, nx] = [10usize, 8, 8];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let iz = flat / (ny * nx);
            iz as f32
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let filter = SobelFilter::unit();
    let (gz, gy, gx) = filter.apply_components(&img).unwrap();

    let (gz_vals, _) = extract_vec_infallible(&gz);
    let (gy_vals, _) = extract_vec_infallible(&gy);
    let (gx_vals, _) = extract_vec_infallible(&gx);
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (gz_vals[flat] - 1.0).abs() < 1e-5,
                    "gz[{iz},{iy},{ix}] = {}, expected 1.0",
                    gz_vals[flat]
                );
                assert!(
                    gy_vals[flat].abs() < 1e-5,
                    "gy[{iz},{iy},{ix}] = {}, expected 0.0",
                    gy_vals[flat]
                );
                assert!(
                    gx_vals[flat].abs() < 1e-5,
                    "gx[{iz},{iy},{ix}] = {}, expected 0.0",
                    gx_vals[flat]
                );
            }
        }
    }
}

/// Anisotropic spacing: I = z + y + x, spacing = [0.5, 1.0, 2.0].
///
/// Expected interior:
///   gz = 1.0 / 0.5 = 2.0
///   gy = 1.0 / 1.0 = 1.0
///   gx = 1.0 / 2.0 = 0.5
///   |∇I| = √(4.0 + 1.0 + 0.25) = √5.25 ≈ 2.2912878
#[test]
fn test_anisotropic_spacing_diagonal() {
    let [nz, ny, nx] = [8usize, 8, 8];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let ix = flat % nx;
            let iy = (flat / nx) % ny;
            let iz = flat / (ny * nx);
            (iz + iy + ix) as f32
        })
        .collect();
    let sp = [0.5, 1.0, 2.0];
    let img = make_image(vals, [nz, ny, nx], sp);
    let filter = SobelFilter::new(sp.into());
    let (gz, gy, gx) = filter.apply_components(&img).unwrap();

    let (gz_vals, _) = extract_vec_infallible(&gz);
    let (gy_vals, _) = extract_vec_infallible(&gy);
    let (gx_vals, _) = extract_vec_infallible(&gx);

    let expected_gz = 1.0_f32 / 0.5;
    let expected_gy = 1.0_f32 / 1.0;
    let expected_gx = 1.0_f32 / 2.0;
    let expected_mag =
        (expected_gz * expected_gz + expected_gy * expected_gy + expected_gx * expected_gx).sqrt();

    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (gz_vals[flat] - expected_gz).abs() < 1e-4,
                    "gz[{iz},{iy},{ix}] = {}, expected {expected_gz}",
                    gz_vals[flat]
                );
                assert!(
                    (gy_vals[flat] - expected_gy).abs() < 1e-4,
                    "gy[{iz},{iy},{ix}] = {}, expected {expected_gy}",
                    gy_vals[flat]
                );
                assert!(
                    (gx_vals[flat] - expected_gx).abs() < 1e-4,
                    "gx[{iz},{iy},{ix}] = {}, expected {expected_gx}",
                    gx_vals[flat]
                );
            }
        }
    }

    let mag = filter.apply(&img).unwrap();
    let (mag_vals, _) = extract_vec_infallible(&mag);
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (mag_vals[flat] - expected_mag).abs() < 1e-4,
                    "magnitude[{iz},{iy},{ix}] = {}, expected {expected_mag}",
                    mag_vals[flat]
                );
            }
        }
    }
}
