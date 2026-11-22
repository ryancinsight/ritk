//! Patch weights and ordered image-boundary regions.

/// `InitializePatchWeightsSmoothDisc` (isotropic unit spacing), squared, in
/// patch-offset order matching the `(dz, dy, dx)` triple loop.
pub(super) fn smooth_disc_weights_sq(patch_radius: usize, ndim: usize) -> Vec<f64> {
    let radius = patch_radius as f32;
    let radius_plus_one = radius + 1.0;
    let disc = (patch_radius / 2) as f32;
    let interval = (patch_radius + 1) - patch_radius / 2;
    let interval = interval as f64;
    let rr = patch_radius as i64;
    let zr = if ndim == 3 { rr } else { 0 };
    let mut w: Vec<f64> = Vec::new();
    for dz in -zr..=zr {
        for dy in -rr..=rr {
            for dx in -rr..=rr {
                let distance = ((dz * dz + dy * dy + dx * dx) as f64).sqrt() as f32;
                let value = if distance >= radius_plus_one {
                    0.0f32
                } else if distance <= disc {
                    1.0f32
                } else {
                    let delta = radius_plus_one - distance;
                    // ITK's expression uses double `pow` for the interval
                    // denominator. The unqualified C++ numerator overload is
                    // host-toolchain dependent: MSVC returns float for the
                    // float arguments, while the GNU/Clang Linux build used
                    // by SimpleITK resolves the global `pow` to double.
                    // Preserve the oracle's host-specific rounding boundary.
                    let delta_cubed = spline_power(delta, 3.0);
                    let delta_squared = spline_power(delta, 2.0);
                    let weight = ((-2.0 / interval.powf(3.0)) * delta_cubed
                        + (3.0 / interval.powf(2.0)) * delta_squared)
                        as f32;
                    weight.clamp(0.0, 1.0)
                };
                let value = f64::from(value);
                w.push(value * value);
            }
        }
    }
    w
}

/// Evaluate one smooth-disc spline power with ITK's host overload.
#[inline]
fn spline_power(base: f32, exponent: f32) -> f64 {
    #[cfg(any(target_env = "msvc", target_os = "macos"))]
    let promoted = f64::from(base.powf(exponent));
    #[cfg(not(any(target_env = "msvc", target_os = "macos")))]
    let promoted = f64::from(base).powf(f64::from(exponent));
    promoted
}

// ── ImageBoundaryFacesCalculator visitation order ────────────────────────────────

/// Pixel visitation order (ITK `(x, y, z)` index space) reproducing
/// `itk::NeighborhoodAlgorithm::ImageBoundaryFacesCalculator`: the interior
/// (non-boundary) region first, then each boundary face, each raster-scanned
/// (x fastest). `radius` is the patch radius; `ndim` ∈ {2, 3}.
pub(super) fn face_calculator_order(
    sizes: [i64; 3],
    radius: i64,
    ndim: usize,
) -> Vec<(i64, i64, i64)> {
    // Each face is a region [start, start+size) in ITK (x, y, z).
    let mut faces: Vec<([i64; 3], [i64; 3])> = Vec::new();
    let mut vr_start = [0i64; 3];
    let mut vr_size = [sizes[0], sizes[1], sizes[2]];
    if ndim == 2 {
        vr_size[2] = 1;
    }
    let mut nb_start = vr_start;
    let mut nb_size = vr_size;

    for i in 0..ndim {
        // Low face (overlapLow = -radius < 0).
        let mut f_start = vr_start;
        let mut f_size = vr_size;
        f_size[i] = radius;
        nb_start[i] += radius;
        nb_size[i] = nb_size[i].saturating_sub(radius);
        vr_start[i] += radius;
        vr_size[i] -= radius;
        faces.push((f_start, f_size));

        // High face (overlapHigh = -radius < 0).
        f_start = vr_start;
        f_size = vr_size;
        f_start[i] = sizes[i] - radius;
        f_size[i] = radius;
        nb_size[i] = nb_size[i].saturating_sub(radius);
        vr_size[i] -= radius;
        faces.push((f_start, f_size));
    }

    // Order: interior (non-boundary) first, then the boundary faces.
    let mut regions: Vec<([i64; 3], [i64; 3])> = Vec::with_capacity(faces.len() + 1);
    regions.push((nb_start, nb_size));
    regions.extend(faces);

    let mut order: Vec<(i64, i64, i64)> = Vec::new();
    for (start, size) in regions {
        if size[0] <= 0 || size[1] <= 0 || size[2] <= 0 {
            continue;
        }
        // Raster scan: x fastest, then y, then z.
        for z in start[2]..start[2] + size[2] {
            for y in start[1]..start[1] + size[1] {
                for x in start[0]..start[0] + size[0] {
                    order.push((x, y, z));
                }
            }
        }
    }
    order
}
