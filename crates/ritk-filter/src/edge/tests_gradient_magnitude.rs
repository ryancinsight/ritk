use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
    ts::make_image_with_spacing::<B, 3>(
        vals,
        dims,
        spacing,
    )
}

/// Uniform image → gradient magnitude = 0 everywhere.
#[test]
fn test_uniform_image_zero_gradient() {
    let dims = [8, 8, 8];
    let vals = vec![5.0_f32; 8 * 8 * 8];
    let img = make_image(vals, dims, [1.0, 1.0, 1.0]);
    let filter = GradientMagnitudeFilter::unit();
    let mag = filter.apply(&img).unwrap();

    let td = mag.data().clone().into_data();
    let out = td.as_slice::<f32>().unwrap();
    for &v in out {
        assert!(v.abs() < 1e-5, "expected 0.0 for uniform image, got {v}");
    }
}

/// I[z,y,x] = x (unit spacing) → gx = 1.0 (interior), gy = gz = 0; magnitude = 1.0 (interior).
#[test]
fn test_ramp_x_gradient() {
    let [nz, ny, nx] = [6usize, 6, 10];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let ix = flat % nx;
            ix as f32
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let filter = GradientMagnitudeFilter::unit();
    let (gz, gy, gx) = filter.apply_components(&img).unwrap();

    let gz_data = gz.data().clone().into_data();
    let gz_vals = gz_data.as_slice::<f32>().unwrap();
    let gy_data = gy.data().clone().into_data();
    let gy_vals = gy_data.as_slice::<f32>().unwrap();
    let gx_data = gx.data().clone().into_data();
    let gx_vals = gx_data.as_slice::<f32>().unwrap();

    // Interior voxels: ix in 1..nx-1
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    gz_vals[flat].abs() < 1e-5,
                    "gz[{iz},{iy},{ix}] = {} expected 0.0",
                    gz_vals[flat]
                );
                assert!(
                    gy_vals[flat].abs() < 1e-5,
                    "gy[{iz},{iy},{ix}] = {} expected 0.0",
                    gy_vals[flat]
                );
                assert!(
                    (gx_vals[flat] - 1.0).abs() < 1e-5,
                    "gx[{iz},{iy},{ix}] = {} expected 1.0",
                    gx_vals[flat]
                );
            }
        }
    }

    // Magnitude image interior
    let mag = filter.apply(&img).unwrap();
    let mag_data = mag.data().clone().into_data();
    let mag_vals = mag_data.as_slice::<f32>().unwrap();
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (mag_vals[flat] - 1.0).abs() < 1e-5,
                    "magnitude[{iz},{iy},{ix}] = {} expected 1.0",
                    mag_vals[flat]
                );
            }
        }
    }
}

/// I[z,y,x] = x + y + z (unit spacing) → each component = 1.0 (interior);
/// magnitude = √3 ≈ 1.7320508 (interior).
#[test]
fn test_diagonal_ramp_gradient() {
    let [nz, ny, nx] = [8usize, 8, 8];
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let ix = flat % nx;
            let iy = (flat / nx) % ny;
            let iz = flat / (ny * nx);
            (iz + iy + ix) as f32
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);
    let filter = GradientMagnitudeFilter::unit();
    let (gz, gy, gx) = filter.apply_components(&img).unwrap();

    let gz_data = gz.data().clone().into_data();
    let gz_vals = gz_data.as_slice::<f32>().unwrap();
    let gy_data = gy.data().clone().into_data();
    let gy_vals = gy_data.as_slice::<f32>().unwrap();
    let gx_data = gx.data().clone().into_data();
    let gx_vals = gx_data.as_slice::<f32>().unwrap();

    let expected_mag = 3.0_f32.sqrt();
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (gz_vals[flat] - 1.0).abs() < 1e-5,
                    "gz interior expected 1.0, got {}",
                    gz_vals[flat]
                );
                assert!(
                    (gy_vals[flat] - 1.0).abs() < 1e-5,
                    "gy interior expected 1.0, got {}",
                    gy_vals[flat]
                );
                assert!(
                    (gx_vals[flat] - 1.0).abs() < 1e-5,
                    "gx interior expected 1.0, got {}",
                    gx_vals[flat]
                );
            }
        }
    }

    let mag = filter.apply(&img).unwrap();
    let mag_data = mag.data().clone().into_data();
    let mag_vals = mag_data.as_slice::<f32>().unwrap();
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (mag_vals[flat] - expected_mag).abs() < 1e-5,
                    "magnitude interior expected √3≈{expected_mag}, got {}",
                    mag_vals[flat]
                );
            }
        }
    }
}

/// Non-unit spacing: I[z,y,x] = x, spacing_x = 2.0 → gx = 0.5 (interior).
#[test]
fn test_non_unit_spacing() {
    let [nz, ny, nx] = [4usize, 4, 8];
    let vals: Vec<f32> = (0..nz * ny * nx).map(|flat| (flat % nx) as f32).collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 2.0]);
    let filter = GradientMagnitudeFilter::new([1.0, 1.0, 2.0].into());
    let (_, _, gx) = filter.apply_components(&img).unwrap();
    let gx_data = gx.data().clone().into_data();
    let gx_vals = gx_data.as_slice::<f32>().unwrap();
    // interior gx = 1 pixel / 2.0 mm = 0.5
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    (gx_vals[flat] - 0.5).abs() < 1e-5,
                    "gx with spacing=2.0 expected 0.5, got {}",
                    gx_vals[flat]
                );
            }
        }
    }
}

/// `apply_from_slice` must produce bit-identical output to `apply` for any input.
///
/// Mathematical justification: both methods share the same finite-difference
/// kernel and `rebuild` call; the only difference is how input data is obtained.
#[test]
fn test_apply_from_slice_matches_apply() {
    let [nz, ny, nx] = [8usize, 10, 12];
    // Non-trivial values to exercise all gradient branches.
    let vals: Vec<f32> = (0..nz * ny * nx)
        .map(|flat| {
            let ix = flat % nx;
            let iy = (flat / nx) % ny;
            let iz = flat / (ny * nx);
            (iz as f32) * 0.3 + (iy as f32) * 0.7 + (ix as f32) * 1.1
        })
        .collect();
    let spacing = [1.5, 2.0, 0.8];
    let img = make_image(vals.clone(), [nz, ny, nx], spacing);
    let filter = GradientMagnitudeFilter::new(spacing.into());

    // Reference path: apply() extracts data internally via clone().into_data().
    let mag_ref = filter.apply(&img).unwrap();
    let ref_data = mag_ref.data().clone().into_data();
    let ref_vals = ref_data.as_slice::<f32>().unwrap();

    // Zero-copy path: apply_from_slice() accepts pre-extracted &[f32].
    let mag_slice = filter.apply_from_slice(&vals, [nz, ny, nx], &img).unwrap();
    let slice_data = mag_slice.data().clone().into_data();
    let slice_vals = slice_data.as_slice::<f32>().unwrap();

    assert_eq!(ref_vals.len(), slice_vals.len(), "output length must match");
    for (i, (&r, &s)) in ref_vals.iter().zip(slice_vals.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            s.to_bits(),
            "apply vs apply_from_slice differ at voxel {i}: apply={r}, from_slice={s}"
        );
    }
}
