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

fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// LoG of a constant image is zero everywhere.
///
/// **Proof**: Let I(x) = c for all x.
///   G_σ * I = c  (Gaussian integrates to 1).
///   ∇²(c) = 0.
/// Therefore LoG(I) = 0. ∎
#[test]
fn test_constant_image_zero_log() {
    let dims = [16, 16, 16];
    let c = 42.0_f32;
    let vals = vec![c; dims[0] * dims[1] * dims[2]];
    let img = make_image(vals, dims, [1.0, 1.0, 1.0]);

    let filter = LaplacianOfGaussianFilter::new(GaussianSigma::new_unchecked(1.5));
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    // Check interior voxels (boundary may have artefacts from the
    // Gaussian conv1d padding propagating into the Laplacian stencil)
    let margin = 6;
    let [nz, ny, nx] = dims;
    for iz in margin..nz - margin {
        for iy in margin..ny - margin {
            for ix in margin..nx - margin {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    out[flat].abs() < 0.1,
                    "LoG of constant image should be zero, but voxel ({iz},{iy},{ix}) = {}",
                    out[flat]
                );
            }
        }
    }
}

/// LoG response at the centre of a bright Gaussian blob is negative.
///
/// **Derivation**: A bright isotropic Gaussian blob I(r) = A·exp(−r²/(2σ_b²))
/// has positive curvature (concave down) at its peak. The Laplacian of a
/// Gaussian-smoothed version of this blob is negative at the centre for
/// σ_smooth near σ_blob, because:
///
///   ∇²(G_σ * I)(0) < 0
///
/// when I is a bright bump. This is the foundation of LoG-based blob
/// detection (Lindeberg 1994).
#[test]
fn test_gaussian_blob_negative_centre() {
    let [nz, ny, nx] = [32usize, 32, 32];
    let n = nz * ny * nx;
    let cz = nz as f64 / 2.0;
    let cy = ny as f64 / 2.0;
    let cx = nx as f64 / 2.0;
    let sigma_blob = 3.0_f64;
    let two_sigma2 = 2.0 * sigma_blob * sigma_blob;
    let amplitude = 100.0_f64;

    let vals: Vec<f32> = (0..n)
        .map(|flat| {
            let ix = flat % nx;
            let iy = (flat / nx) % ny;
            let iz = flat / (ny * nx);
            let dz = iz as f64 - cz;
            let dy = iy as f64 - cy;
            let dx = ix as f64 - cx;
            let r2 = dz * dz + dy * dy + dx * dx;
            (amplitude * (-r2 / two_sigma2).exp()) as f32
        })
        .collect();

    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

    // Use a sigma close to the blob scale for maximum LoG response
    let filter = LaplacianOfGaussianFilter::new(GaussianSigma::new_unchecked(3.0));
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    // The centre voxel should have a negative LoG response
    let centre_flat = (nz / 2) * ny * nx + (ny / 2) * nx + (nx / 2);
    assert!(
        out[centre_flat] < 0.0,
        "LoG at centre of bright Gaussian blob should be negative, got {}",
        out[centre_flat]
    );

    // Verify the response is substantially negative (not just rounding noise)
    assert!(
        out[centre_flat] < -0.1,
        "LoG response at blob centre should be substantially negative, got {}",
        out[centre_flat]
    );
}

/// LoG of a linear field I = x + y + z is zero (∇² of a linear function = 0).
///
/// **Proof**: G_σ * (ax + by + cz + d) = ax + by + cz + d (linear functions
/// are invariant under Gaussian smoothing except at boundaries).
/// ∇²(ax + by + cz + d) = 0. ∎
#[test]
fn test_linear_field_zero_log() {
    let [nz, ny, nx] = [16usize, 16, 16];
    let n = nz * ny * nx;
    let vals: Vec<f32> = (0..n)
        .map(|flat| {
            let ix = (flat % nx) as f32;
            let iy = ((flat / nx) % ny) as f32;
            let iz = (flat / (ny * nx)) as f32;
            ix + iy + iz
        })
        .collect();
    let img = make_image(vals, [nz, ny, nx], [1.0, 1.0, 1.0]);

    let filter = LaplacianOfGaussianFilter::new(GaussianSigma::new_unchecked(1.5));
    let result = filter.apply(&img).unwrap();
    let out = extract_vals(&result);

    // Interior voxels should be near zero
    let margin = 5;
    for iz in margin..nz - margin {
        for iy in margin..ny - margin {
            for ix in margin..nx - margin {
                let flat = iz * ny * nx + iy * nx + ix;
                assert!(
                    out[flat].abs() < 0.5,
                    "LoG of linear field should be ~0 at interior, but voxel ({iz},{iy},{ix}) = {}",
                    out[flat]
                );
            }
        }
    }
}
