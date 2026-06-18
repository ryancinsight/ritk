use super::gaussian_image_source;

/// Peak at the mean voxel, value = scale; symmetric falloff matching
/// `exp(−½·((p−mean)/sigma)²)`. 1-D along x: size 9, sigma 6, mean 8, scale 100,
/// origin 2, spacing 1 → physical p = 2+x, peak at x=6 (p=8). Pinned by sitk.
#[test]
fn gaussian_source_matches_physical_formula() {
    let (buf, dims) = gaussian_image_source(
        [9, 1, 1],
        [6.0, 1.0, 1.0],
        [8.0, 0.0, 0.0],
        100.0,
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    );
    assert_eq!(dims, [1, 1, 9]);
    // Peak at x=6 (physical 8 == mean).
    assert_eq!(buf.iter().cloned().fold(f32::MIN, f32::max), buf[6]);
    assert!((buf[6] - 100.0).abs() < 1e-3, "peak {}", buf[6]);
    for (x, &got) in buf.iter().enumerate() {
        let p = 2.0 + x as f64;
        let exp = 100.0 * (-0.5 * ((p - 8.0) / 6.0_f64).powi(2)).exp();
        assert!((got as f64 - exp).abs() < 1e-3, "x={x}: {got} vs {exp}");
    }
}

/// 3-D separability: the value at the centre voxel equals the product of the
/// per-axis Gaussians (a constant 1 here since each axis peaks at its mean).
#[test]
fn gaussian_source_3d_peak_is_scale() {
    let (buf, dims) = gaussian_image_source(
        [5, 5, 5],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0], // mean at index 2 each axis (origin 0, spacing 1)
        50.0,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    );
    assert_eq!(dims, [5, 5, 5]);
    let center = (2 * 5 + 2) * 5 + 2;
    assert!((buf[center] - 50.0).abs() < 1e-3, "center {}", buf[center]);
}
