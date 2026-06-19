//! Gaussian pre-smooth + stride-subsample and trilinear displacement upsampling
//! for the multi-resolution Demons pyramid.

/// Gaussian pre-smooth then stride-subsample an image.
///
/// Returns (downsampled_image, coarse_dims) where
/// `coarse_dims[i] = max(1, dims[i] / factor)`.
///
/// # Invariants
/// - Gaussian blur with sigma = 0.5 * factor before subsampling (anti-aliasing).
/// - Coarse voxel (iz, iy, ix) <- fine voxel (iz*factor, iy*factor, ix*factor).
pub(super) fn downsample(image: &[f32], dims: [usize; 3], factor: usize) -> (Vec<f32>, [usize; 3]) {
    let factor = factor.max(1);
    let mut smoothed = image.to_vec();
    let sigma = 0.5 * factor as f64;
    crate::deformable_field_ops::gaussian_smooth_inplace(&mut smoothed, dims.into(), sigma);

    let [nz, ny, nx] = dims;
    let ncz = (nz / factor).max(1);
    let ncy = (ny / factor).max(1);
    let ncx = (nx / factor).max(1);

    let mut out = vec![0.0f32; ncz * ncy * ncx];
    for iz in 0..ncz {
        for iy in 0..ncy {
            for ix in 0..ncx {
                let fz = (iz * factor).min(nz - 1);
                let fy = (iy * factor).min(ny - 1);
                let fx = (ix * factor).min(nx - 1);
                out[iz * ncy * ncx + iy * ncx + ix] = smoothed[fz * ny * nx + fy * nx + fx];
            }
        }
    }
    (out, [ncz, ncy, ncx])
}

/// Trilinear upsample a single displacement field component from coarse to fine dims.
///
/// Each value is multiplied by scale to convert coarse-voxel displacement units
/// into fine-voxel displacement units.
///
/// # Invariants
/// - `scale = fine_dims[i] / coarse_dims[i]` for the corresponding axis.
/// - Boundary conditions: clamp-to-border via trilinear_interpolate.
pub(super) fn upsample_displacement(
    coarse: &[f32],
    coarse_dims: [usize; 3],
    fine_dims: [usize; 3],
    scale: f32,
) -> Vec<f32> {
    let [nfz, nfy, nfx] = fine_dims;
    let mut out = vec![0.0f32; nfz * nfy * nfx];

    for iz in 0..nfz {
        for iy in 0..nfy {
            for ix in 0..nfx {
                let cz = iz as f32 * coarse_dims[0] as f32 / fine_dims[0] as f32;
                let cy = iy as f32 * coarse_dims[1] as f32 / fine_dims[1] as f32;
                let cx = ix as f32 * coarse_dims[2] as f32 / fine_dims[2] as f32;
                let val = crate::deformable_field_ops::trilinear_interpolate(
                    coarse,
                    coarse_dims.into(),
                    cz,
                    cy,
                    cx,
                );
                out[iz * nfy * nfx + iy * nfx + ix] = scale * val;
            }
        }
    }
    out
}
