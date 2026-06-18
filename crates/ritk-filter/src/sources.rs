//! Procedural image sources (generate an image from parameters, no input).

/// Generate a Gaussian blob image (`itk::GaussianImageSource`).
///
/// Each voxel takes the value of an axis-aligned Gaussian evaluated at the
/// voxel's **physical** position `p_d = origin_d + index_d · spacing_d`:
///
/// ```text
/// out(index) = scale · exp(−½ · Σ_d ((p_d − mean_d) / sigma_d)²)
/// ```
///
/// (the non-normalised form, ITK default `Normalized = false`, so the peak value
/// is `scale`). Parameters are in **sitk axis order** `(x, y, z)`; the returned
/// buffer is in ritk tensor order `[z, y, x]` with dims `[nz, ny, nx]`. The
/// reduction runs in `f64` to match ITK's `RealType`, then narrows to `f32`.
///
/// Verified float-exact (~3e-6) against `sitk.GaussianSource(sigma, mean, scale,
/// origin)`.
pub fn gaussian_image_source(
    size_xyz: [usize; 3],
    sigma_xyz: [f64; 3],
    mean_xyz: [f64; 3],
    scale: f64,
    origin_xyz: [f64; 3],
    spacing_xyz: [f64; 3],
) -> (Vec<f32>, [usize; 3]) {
    let [nx, ny, nz] = size_xyz;
    let mut out = vec![0.0f32; nx * ny * nz];
    for z in 0..nz {
        let pz = origin_xyz[2] + z as f64 * spacing_xyz[2];
        let az = ((pz - mean_xyz[2]) / sigma_xyz[2]).powi(2);
        for y in 0..ny {
            let py = origin_xyz[1] + y as f64 * spacing_xyz[1];
            let ay = ((py - mean_xyz[1]) / sigma_xyz[1]).powi(2);
            let row = (z * ny + y) * nx;
            for x in 0..nx {
                let px = origin_xyz[0] + x as f64 * spacing_xyz[0];
                let ax = ((px - mean_xyz[0]) / sigma_xyz[0]).powi(2);
                out[row + x] = (scale * (-0.5 * (ax + ay + az)).exp()) as f32;
            }
        }
    }
    (out, [nz, ny, nx])
}

#[cfg(test)]
#[path = "tests_sources.rs"]
mod tests_sources;
