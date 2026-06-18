use super::warp_image;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, dims)
}

/// A linear ramp warped by a constant displacement equals the shifted ramp in
/// the interior. Trilinear interpolation is exact for affine functions, so the
/// expected value is the closed-form sample `f(x+dx, y+dy, z+dz)`.
#[test]
fn warp_constant_shift_of_ramp() {
    let (nz, ny, nx) = (6usize, 6, 6);
    // f(z,y,x) = x + 2y + 3z (axis-major ramp, exact under trilinear).
    let mut vals = vec![0.0f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                vals[iz * ny * nx + iy * nx + ix] = ix as f32 + 2.0 * iy as f32 + 3.0 * iz as f32;
            }
        }
    }
    let moving = make(vals.clone(), [nz, ny, nx]);
    // Constant physical displacement d = (dz=0, dy=0.5, dx=1.0); unit spacing.
    let dz = make(vec![0.0; nz * ny * nx], [nz, ny, nx]);
    let dy = make(vec![0.5; nz * ny * nx], [nz, ny, nx]);
    let dx = make(vec![1.0; nz * ny * nx], [nz, ny, nx]);

    let out = warp_image(&moving, &dz, &dy, &dx).unwrap();
    let (ov, _) = extract_vec_infallible(&out);
    // Interior (where x+1 and y+0.5 stay in-bounds): out = f(x+1, y+0.5, z) =
    // (x+1) + 2(y+0.5) + 3z = ramp + 2.
    for iz in 0..nz {
        for iy in 0..ny - 1 {
            for ix in 0..nx - 1 {
                let got = ov[iz * ny * nx + iy * nx + ix];
                let want = vals[iz * ny * nx + iy * nx + ix] + 2.0;
                assert!(
                    (got - want).abs() < 1e-4,
                    "warp[{iz},{iy},{ix}]: got {got}, want {want}"
                );
            }
        }
    }
}

/// A displacement that pushes the sample point outside the moving buffer yields
/// the edge-padding value (0), matching ITK's `IsInsideBuffer` gate.
#[test]
fn warp_out_of_bounds_is_zero() {
    let dims = [3usize, 3, 3];
    let moving = make(vec![5.0f32; 27], dims); // constant non-zero image
                                               // Large positive displacement pushes every sample far past the upper border.
    let dz = make(vec![100.0; 27], dims);
    let dy = make(vec![100.0; 27], dims);
    let dx = make(vec![100.0; 27], dims);
    let out = warp_image(&moving, &dz, &dy, &dx).unwrap();
    let (ov, _) = extract_vec_infallible(&out);
    assert!(
        ov.iter().all(|&v| v == 0.0),
        "out-of-bounds samples must be 0"
    );
}

/// Zero displacement is the identity (every voxel samples itself exactly).
#[test]
fn warp_zero_displacement_is_identity() {
    let dims = [4usize, 5, 3];
    let n: usize = dims.iter().product();
    let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 1.7).sin()).collect();
    let moving = make(vals.clone(), dims);
    let zero = make(vec![0.0; n], dims);
    let out = warp_image(&moving, &zero, &zero, &zero).unwrap();
    let (ov, _) = extract_vec_infallible(&out);
    for (i, (&got, &want)) in ov.iter().zip(vals.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "voxel {i}: got {got}, want {want}"
        );
    }
}

/// Mismatched component shapes are a typed error, not a panic.
#[test]
fn warp_mismatched_field_shapes_errors() {
    let moving = make(vec![0.0; 8], [2, 2, 2]);
    let dz = make(vec![0.0; 8], [2, 2, 2]);
    let dy = make(vec![0.0; 27], [3, 3, 3]);
    let dx = make(vec![0.0; 8], [2, 2, 2]);
    assert!(warp_image(&moving, &dz, &dy, &dx).is_err());
}
