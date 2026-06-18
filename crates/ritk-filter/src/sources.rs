//! Procedural image sources (generate an image from parameters, no input).

use std::f64::consts::PI;

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

/// Generate a grid-pattern image (`itk::GridImageSource`).
///
/// Dark periodic Gaussian lines on a bright background. For each **selected**
/// dimension `d` (`which_dimensions[d]`), the per-axis line profile at physical
/// position `p_d = origin_d + index_d · spacing_d` is the sum of Gaussians
/// centred on every grid line `offset_d + k·grid_spacing_d`:
///
/// ```text
/// profile_d(p) = Σ_k exp(−(p − (offset_d + k·grid_spacing_d))² / (2·sigma_d²))
/// out(index)   = scale · Π_{d selected} (1 − profile_d(p_d))
/// ```
///
/// so the value is `0` on a grid line and `≈ scale` between lines, combining
/// across selected axes by the product rule (verified against `sitk.GridSource`:
/// 1-D `x=2` → `scale·(1 − 2·e⁻⁸)`; 2-D `(2,2)` → `scale·(1−p)²`). Grid lines
/// beyond `8σ` of `p` are below `f32` epsilon and omitted. Parameters are in
/// sitk `(x, y, z)` order; output buffer is `[z, y, x]`.
#[allow(clippy::too_many_arguments)]
pub fn grid_image_source(
    size_xyz: [usize; 3],
    spacing_xyz: [f64; 3],
    origin_xyz: [f64; 3],
    sigma_xyz: [f64; 3],
    grid_spacing_xyz: [f64; 3],
    grid_offset_xyz: [f64; 3],
    scale: f64,
    which_dims_xyz: [bool; 3],
) -> (Vec<f32>, [usize; 3]) {
    // Per-axis line profile at physical position `p`.
    let profile = |d: usize, p: f64| -> f64 {
        if !which_dims_xyz[d] {
            return 0.0;
        }
        let (sig, gs, off) = (sigma_xyz[d], grid_spacing_xyz[d], grid_offset_xyz[d]);
        let win = 8.0 * sig; // lines beyond 8σ are < f32 epsilon
        let klo = ((p - win - off) / gs).floor() as i64;
        let khi = ((p + win - off) / gs).ceil() as i64;
        let mut s = 0.0;
        for k in klo..=khi {
            let line = off + k as f64 * gs;
            let t = (p - line) / sig;
            s += (-0.5 * t * t).exp();
        }
        s
    };

    let [nx, ny, nz] = size_xyz;
    let mut out = vec![0.0f32; nx * ny * nz];
    for z in 0..nz {
        let fz = 1.0 - profile(2, origin_xyz[2] + z as f64 * spacing_xyz[2]);
        for y in 0..ny {
            let fy = 1.0 - profile(1, origin_xyz[1] + y as f64 * spacing_xyz[1]);
            let row = (z * ny + y) * nx;
            for x in 0..nx {
                let fx = 1.0 - profile(0, origin_xyz[0] + x as f64 * spacing_xyz[0]);
                out[row + x] = (scale * fz * fy * fx) as f32;
            }
        }
    }
    (out, [nz, ny, nx])
}

/// Generate a physical-point vector image (`itk::PhysicalPointImageSource`).
///
/// Each voxel holds its own physical coordinate as a 3-component vector:
/// component `d` at index `i` is `origin_d + index_d · spacing_d` (identity
/// direction). Returns the three channel buffers in sitk axis order `(x, y, z)`
/// and dims `[nz, ny, nx]`; matches `sitk.PhysicalPointSource`.
pub fn physical_point_image_source(
    size_xyz: [usize; 3],
    origin_xyz: [f64; 3],
    spacing_xyz: [f64; 3],
) -> ([Vec<f32>; 3], [usize; 3]) {
    let [nx, ny, nz] = size_xyz;
    let n = nx * ny * nz;
    let (mut cx, mut cy, mut cz) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    for z in 0..nz {
        let pz = (origin_xyz[2] + z as f64 * spacing_xyz[2]) as f32;
        for y in 0..ny {
            let py = (origin_xyz[1] + y as f64 * spacing_xyz[1]) as f32;
            let row = (z * ny + y) * nx;
            for x in 0..nx {
                let i = row + x;
                cx[i] = (origin_xyz[0] + x as f64 * spacing_xyz[0]) as f32;
                cy[i] = py;
                cz[i] = pz;
            }
        }
    }
    ([cx, cy, cz], [nz, ny, nx])
}

/// Generate a Gabor-wavelet image (`itk::GaborImageSource`).
///
/// A Gaussian envelope modulated by a cosine along the **x** axis (the real
/// part; sitk exposes no imaginary toggle):
///
/// ```text
/// out(index) = exp(−½·Σ_d ((p_d − mean_d)/sigma_d)²) · cos(2π·frequency·(p_x − mean_x))
/// ```
///
/// where `p_d = origin_d + index_d · spacing_d`. The peak (at `mean`) is `1`.
/// Parameters are in sitk `(x, y, z)` order; output buffer is `[z, y, x]`.
/// Verified float-exact against `sitk.GaborImageSource`.
pub fn gabor_image_source(
    size_xyz: [usize; 3],
    spacing_xyz: [f64; 3],
    origin_xyz: [f64; 3],
    sigma_xyz: [f64; 3],
    mean_xyz: [f64; 3],
    frequency: f64,
) -> (Vec<f32>, [usize; 3]) {
    let [nx, ny, nz] = size_xyz;
    let mut out = vec![0.0f32; nx * ny * nz];
    let two_pi_f = 2.0 * PI * frequency;
    for z in 0..nz {
        let pz = origin_xyz[2] + z as f64 * spacing_xyz[2];
        let ez = ((pz - mean_xyz[2]) / sigma_xyz[2]).powi(2);
        for y in 0..ny {
            let py = origin_xyz[1] + y as f64 * spacing_xyz[1];
            let ey = ((py - mean_xyz[1]) / sigma_xyz[1]).powi(2);
            let row = (z * ny + y) * nx;
            for x in 0..nx {
                let px = origin_xyz[0] + x as f64 * spacing_xyz[0];
                let ex = ((px - mean_xyz[0]) / sigma_xyz[0]).powi(2);
                let env = (-0.5 * (ex + ey + ez)).exp();
                let modulation = (two_pi_f * (px - mean_xyz[0])).cos();
                out[row + x] = (env * modulation) as f32;
            }
        }
    }
    (out, [nz, ny, nx])
}

#[cfg(test)]
#[path = "tests_sources.rs"]
mod tests_sources;
