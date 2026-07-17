use super::warp_image;
use coeus_core::SequentialBackend;
use ritk_image::native::Image;
use ritk_image::test_support as ts;

type B = SequentialBackend;

fn make(data: Vec<f32>, dims: [usize; 3]) -> Image<f32, B, 3> {
    ts::burn_compat::make_image::<B, 3>(data, dims)
}

/// Warping a constant image by any (in-bounds) displacement returns the same
/// constant, since every interpolated sample of a constant image equals that
/// constant — independent of the axis/geometry convention. (The exact value and
/// displacement-direction parity vs `sitk.Warp` is verified in the Python cmake
/// test on loaded images, which carry ritk.io's canonical Direction.)
#[test]
fn warp_constant_image_is_preserved() {
    let (nz, ny, nx) = (6usize, 6, 6);
    let n = nz * ny * nx;
    let moving = make(vec![7.0f32; n], [nz, ny, nx]);
    // Small spatially-varying displacement that stays within the buffer interior.
    let mut dxv = vec![0.0f32; n];
    for (i, v) in dxv.iter_mut().enumerate() {
        *v = 0.3 * ((i % 3) as f32 - 1.0);
    }
    let dz = make(vec![0.2; n], [nz, ny, nx]);
    let dy = make(vec![-0.4; n], [nz, ny, nx]);
    let dx = make(dxv, [nz, ny, nx]);

    let out = warp_image(&moving, &dz, &dy, &dx, &B::default()).unwrap();
    let ov = out.data().as_slice();
    // Interior voxels (displacement < 0.5, so samples stay in-bounds) equal 7.0.
    for iz in 1..nz - 1 {
        for iy in 1..ny - 1 {
            for ix in 1..nx - 1 {
                let got = ov[iz * ny * nx + iy * nx + ix];
                assert!(
                    (got - 7.0).abs() < 1e-4,
                    "warp[{iz},{iy},{ix}]: got {got}, want 7.0"
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
    let out = warp_image(&moving, &dz, &dy, &dx, &B::default()).unwrap();
    let ov = out.data().as_slice();
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
    let out = warp_image(&moving, &zero, &zero, &zero, &B::default()).unwrap();
    let ov = out.data().as_slice();
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
    assert!(warp_image(&moving, &dz, &dy, &dx, &B::default()).is_err());
}
